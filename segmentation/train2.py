
import os
import shutil
import tempfile
from time import time
import random
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pathlib
import yaml
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import one_hot

import monai
from monai.metrics import DiceMetric
from monai.handlers import CheckpointSaver, CheckpointLoader
from monai.metrics import ROCAUCMetric, compute_meandice
from monai.data import Dataset, DataLoader, decollate_batch, CacheDataset
from monai.data.nifti_saver import NiftiSaver
from monai.data.png_saver import PNGSaver
from monai.transforms import Activations, Compose, EnsureType, AsDiscrete, SaveImage
from monai.utils import set_determinism
from monai.utils.misc import first
from monai.transforms.utils_pytorch_numpy_unification import ravel, concatenate

from data.load_data import read_split_file, initialize_transform, dataset_prep, save_img
from segmentation.module import BILSTM_SC_UNET, Generator_VGG16_mini_SE
from segmentation.loss import Loss_function
from segmentation.eval_plot import save_plots


generators={"VGG16_mini_SE" : Generator_VGG16_mini_SE}

set_determinism(seed=0)
class seq_by_seq(object):

    def __init__(self, config):
        
        split_file = os.path.join(pathlib.Path(config['experiment_dir']), pathlib.Path(config['split_file']))
        self.experiment_folder = os.path.join(pathlib.Path(config['experiment_dir']), pathlib.Path(config['output_dir']))

        split_data = read_split_file(split_file)
        
        self.train_list = dataset_prep(split_data['training'])
        self.validate_list = dataset_prep(split_data['validate'])
         
        image_keys=('ct', 'pet', 'gtv')

        self.train_tranforms = initialize_transform(image_keys, config['direction'], ct_norm="z_norm", pet_norm="z_norm", test=False)
        self.test_tranforms = initialize_transform(image_keys, config['direction'], ct_norm="z_norm", pet_norm="z_norm", test=True)
        
        self.checkpoint_dir = os.path.join(pathlib.Path(self.experiment_folder), pathlib.Path(config['checkpoint_dir']))
        self.sample_dir = os.path.join(pathlib.Path(self.experiment_folder), pathlib.Path(config['sample_dir']))
        self.test_dir = os.path.join(pathlib.Path(self.experiment_folder), pathlib.Path(config['test_dir']))
        self.output_dir = pathlib.Path(self.experiment_folder)
        
        self.batch_size = config['batch_size'] #batch size indicates the number of slices which we are considering in a sequence
        
        self.direction = config['direction']
        self.seq_size = config['seq_size']
        self.image_size = config['fine_size']
        
        self.input_c_dim = config['input_nc']
        self.output_c_dim = config['output_nc']
        
        self.epoch = config['epoch']
        self.epoch_step = config['epoch_step']
        
        self.model_type = config['model_type']
        self.encoder_type = config['encoder_type']

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        
        if self.model_type=="BILSTM_SC_UNET":
            self.net=BILSTM_SC_UNET(dims=3, in_channels=2, out_channels=1, kernel_size=3, pool_kernel=2, 
                                   act="relu", generator_model=generators[self.encoder_type]).to(self.device)
        else:
            assert False, 'Unknown encoder type'
        
        self.lr = config['lr']
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
        self.loss_type = config['loss_type'] #loss_function = torch.nn.CrossEntropyLoss()
        self.loss_function = Loss_function(self.loss_type)

        self.save_freq = config['save_freq']
        self.num_workers = config['num_workers']
        
        self.image_saver_png = SaveImage(output_dir=self.sample_dir,output_postfix="",output_ext=".png",separate_folder=False, scale=255)
        self.image_saver_nifty = SaveImage(output_dir=self.test_dir,output_postfix="",output_ext=".nii.gz",separate_folder=False, scale=255, squeeze_end_dims=True)
        
        self.continue_ = config['continue_train']
        self.loading_checkpoint = config['load_checkpoint']
        
        self.saver = CheckpointSaver(self.checkpoint_dir,
                                     save_dict={
                                         'network': self.net, 'optimizer': self.optimizer
                                               },
                                     name=None, file_prefix='check_point',
                                     save_final=True, final_filename='last_check_point',
                                     key_metric_n_saved=5,
                                     key_metric_filename="best_model",
                                     key_metric_greater_or_equal=True,
                                     epoch_level=True, n_saved=8)



    def train(self, config):
        
        pin_memory = True

        self.net = torch.nn.DataParallel(self.net).to(self.device)
        ##### self.lr_decay = torch.optim.lr_scheduler.MultiStepLR(self.opt, [400])
        
        train_ds = CacheDataset(data=self.train_list, transform=self.train_tranforms,cache_rate=1.0, num_workers=4)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
        
        validate_ds = CacheDataset(data=self.validate_list, transform=self.train_tranforms,cache_rate=1.0, num_workers=4)
        validate_loader = DataLoader(validate_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
        
        post_pred = Compose([EnsureType(), Activations(sigmoid=True)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

        lr= self.lr

        best_metric = -1
        best_metric_dice = -1
        best_metric_epoch = -1
        
        metric_values = list()
        auc_metric = ROCAUCMetric()
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        
        init=0
        epoch_loss_values=list()
        dice_val=list()
        
        loss_saver_train = list()
        loss_saver_val = list()

        start = time()
        #writer = SummaryWriter()
        
        for epoch in range(init+1,self.epoch+1):
            
            print("Starting epoch: " + str(epoch))
            
            if self.continue_:
                self.load(self.checkpoint_dir, which_check="last")
            
            loss_step = []
            self.net.train()
            
            if epoch >= self.epoch_step and np.mod(epoch,10)==0:
                lr = lr/2
            
            #self.generator_optimizer.learning_rate.assign(lr) 
            step=0
            epoch_loss=0
            
            for batch_data in train_loader:
                
                step += 1
                
                print(" - step: " + str(step))

                input_, gtv = batch_data['input'].to(self.device), batch_data['gtv'].to(self.device)
                
                input_=Variable(input_)
                gtv=Variable(gtv)
                
                self.optimizer.zero_grad()
                
                output = self.net(input_)
                
                loss = self.loss_function(output, gtv) 
                
                loss.backward()
                self.optimizer.step()
                
                loss_step.append(loss.item())
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                
                print(f"{step}/{len(train_ds) // train_loader.batch_size + 1}, training_loss: {loss.item():.4f}")
                
                if np.mod(epoch*step, self.save_freq) == 0: #1
                    self.sample_model(epoch, step, validate_loader)
                    
                #writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
                
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            
            loss_mean = (sum(loss_step)/len(loss_step))

            print("epoch / training loss")
            print(epoch,loss_mean)

            loss_saver_train.append(loss_mean)
            
            torch.save(self.net.state_dict(), os.path.join(self.checkpoint_dir, "last_model.pth"))
            
            print("Start validation:")
            self.net.eval()
            
            preds = list()
            labels = list()
            loss_step_val = []
            #auc_metric = list()
            #dice_metric=list()

            with torch.no_grad():
            
                counter = 0
                
                for batch_data_val in validate_loader:
                    
                    input_, gtv = batch_data_val['input'].to(self.device), batch_data_val['gtv'].to(self.device)
                    output = self.net(input_)
                    output_=output[2]
                    
                    if counter == 0:
                      preds_tensor = ravel(output_)
                      labels_tensor = ravel(gtv)
                    else:
                      preds_tensor = concatenate([preds_tensor, ravel(output_)])
                      labels_tensor = concatenate([labels_tensor, ravel(gtv)])
                      
                    loss = self.loss_function(output, gtv) #### implemn
                    loss_step_val.append(loss.item())
                    
                    counter=counter+1
                    
                    if counter == 50:
                      preds.append(preds_tensor)
                      labels.append(labels_tensor)
                      counter = 0
                    
                    dice_metric(y_pred=output_, y=gtv)
                    
                loss_mean_val = (sum(loss_step_val)/len(loss_step_val))
                
                metric = dice_metric.aggregate().item() #np.mean(dice_metric)
                dice_val.append(metric) 
                dice_metric.reset()
                
                print("epoch / validation loss")
                print(epoch,loss_mean_val)

                loss_saver_val.append(loss_mean_val)
                
                #y_onehot = [to_onehot(i) for i in decollate_batch(y)]        
                #y_pred_act = [act(i) for i in decollate_batch(y_pred)]
                #auc_metric(y_pred_act, y_onehot)
                
                y_onehot = [post_label(i) for i in decollate_batch(labels)]
                y_pred_act = [post_pred(i) for i in decollate_batch(preds)]
                
                for y_, z_ in zip(y_pred_act, y_onehot):
                  #x = metrics.compute_roc_auc(y_, z_)
                  auc_metric(y_, z_)
                           
                #auc_metric(y_pred_act, y_onehot)
                #auc_metric(y_pred=preds, y= labels)
                auc_value = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(auc_value)
                
                df = pd.DataFrame(list(zip(loss_saver_train, loss_saver_val, dice_val, metric_values)),
                                       columns =['Loss Train', 'Loss Val', 'Dice Val', 'AUC Val'])
                df.to_excel(os.path.join(self.output_dir, 'training_metrics.xlsx'))
                
                if epoch > 1:
                    save_plots(loss_saver_train, loss_saver_val,  dice_val, metric_values, self.output_dir)
                
                if auc_value > best_metric:
                    best_metric = auc_value
                    best_metric_epoch = epoch
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_dir, "best_auc_model.pth"))
                    print("saved new best metric network for AUC")
                    
                if metric > best_metric_dice:
                    best_metric_dice = metric
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_dir, "best_dice_model.pth"))
                    print("saved new best metric network for DICE")

                print(
                    f"current epoch: {epoch} current AUC: {auc_value:.4f} current Dice: {metric:.4f} /"
                    f" at epoch: {best_metric_epoch}"
                )

        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

        #writer.add_scalar("val_mean_dice", metric, epoch)
        #writer.close()

        total_time = time() - start
        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
        
    def load(self, checkpoint_dir, which_check):
        
        print(" [*] Reading checkpoint...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.output_dir)
        
        if which_check=="best_dice":
            print(" [*] Best dice checkpoint loading!")
            self.net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_dice_model.pth")))
            return True
        elif which_check=="best_auc":
            print(" [*] Best AUC checkpoint loading!")
            self.net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_auc_model.pth")))
            return True
        elif which_check=="last":
            print(" [*] Last checkpoint loading!")
            self.net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "last_model.pth")))
            return True
        else: 
            False
        
    def sample_model(self, epoch, idx, validate_loader):
        
        pin_memory = True
        self.net.eval()
        
        with torch.no_grad():
        
            element = random.choices(self.validate_list, k=3)
    
            validate_ds = Dataset(data=element, transform=self.train_tranforms)
            validate_loader = DataLoader(validate_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=pin_memory)
            
            batch_data_val =  first(validate_loader)  
            input_, gtv = batch_data_val['input'].to(self.device), batch_data_val['gtv'].to(self.device)
    
            output = self.net(input_)
            
            save_img(gtv, self.image_saver_png,'GTOriginal_{:02d}_{:04d}.png'.format(epoch, idx))
            save_img(output[2], self.image_saver_png,'GTPredicted_{:02d}_{:04d}.png'.format(epoch, idx))
        
    def test(self, config):
        
        pin_memory = True
        
        if config['phase']=="train":
            ds = Dataset(data=self.train_list, transform=self.train_tranforms)
        elif config['phase']=="validate":
            ds = Dataset(data=self.validate_list, transform=self.test_tranforms)
        
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=pin_memory)
        checkpoint_dir = os.path.join(pathlib.Path(config['output_dir']), pathlib.Path(config['checkpoint_dir']))
        
        if self.load(checkpoint_dir, which_check=self.loading_checkpoint):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        self.net.eval()
        with torch.no_grad():
            for batch_data_val in loader:
                input_, slice_, id_ = batch_data_val['input'].to(self.device), batch_data_val['slice'].to(self.device), batch_data_val['ID']
                output = self.net(input_)
                save_img(output[2], self.image_saver_nifty, '{0}.nii.gz'.format(id_[0] + '_seq_' + str(slice_[0].cpu().numpy())))

      