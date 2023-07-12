
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
import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import one_hot
import wandb
import monai
from monai.metrics import DiceMetric, ROCAUCMetric,ConfusionMatrixMetric, compute_confusion_matrix_metric
from monai.handlers import CheckpointSaver, CheckpointLoader
from monai.metrics import ROCAUCMetric, compute_meandice
from monai.data import Dataset, DataLoader, decollate_batch, CacheDataset
from monai.data.nifti_saver import NiftiSaver
from monai.data.png_saver import PNGSaver
from monai.transforms import Activations, Compose, EnsureType, AsDiscrete, SaveImage
from monai.utils import set_determinism
from monai.utils.misc import first
from monai.transforms.utils_pytorch_numpy_unification import ravel, concatenate
from data.load_data import read_split_file, initialize_transform, dataset_prep, save_img, combine_slices
from segmentation.module import BILSTM_SC_UNET, Generator_VGG16_mini_SE
from segmentation.loss import Loss_function, tversky
from segmentation.eval_plot import save_plots_pt


generators={"VGG16_mini_SE" : Generator_VGG16_mini_SE}

set_determinism(seed=0)


class seq_by_seq(object):

    def __init__(self, config):
        
        split_file = os.path.join(pathlib.Path(config['experiment_dir']), pathlib.Path(config['split_file']))
        self.experiment_folder = os.path.join(pathlib.Path(config['experiment_dir']), pathlib.Path(config['output_dir']))

        split_data = read_split_file(split_file)
        
        self.train_list = dataset_prep(split_data['training'])
        self.validate_list = dataset_prep(split_data['validate'])
         
        self.image_size = int(config['fine_size'])
        
        image_keys=('ct', 'pet', 'gtv')

        self.train_tranforms = initialize_transform(image_keys, config['direction'], level=60, window=350, image_size = self.image_size, ct_norm="z_norm", pet_norm="z_norm", test=False)
        self.test_tranforms = initialize_transform(image_keys, config['direction'], level=60, window=350, image_size = self.image_size, ct_norm="z_norm", pet_norm="z_norm", test=True)
        
        self.checkpoint_dir = os.path.join(pathlib.Path(self.experiment_folder), pathlib.Path(config['checkpoint_dir']))
        self.sample_dir = os.path.join(pathlib.Path(self.experiment_folder), pathlib.Path(config['sample_dir']))
        self.test_dir = os.path.join(pathlib.Path(self.experiment_folder), pathlib.Path(config['test_dir']))
        self.output_dir = pathlib.Path(self.experiment_folder)
        self.data_folder = pathlib.Path(config['data_folder'])
        
        self.batch_size = config['batch_size'] #batch size indicates the number of slices which we are considering in a sequence
        
        self.direction = config['direction']
        self.seq_size = config['seq_size']
        
        self.input_c_dim = config['input_nc']
        self.output_c_dim = config['output_nc']
        
        self.epoch = config['epoch']
        self.epoch_step = config['epoch_step']
        
        self.model_type = config['model_type']
        self.encoder_type = config['encoder_type']

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #torch.cuda.set_device(self.device)
        
        if self.model_type=="BILSTM_SC_UNET":
            self.net=BILSTM_SC_UNET(dims=3, in_channels=2, out_channels=1, kernel_size=3, pool_kernel=2, 
                                   act="relu", generator_model=generators[self.encoder_type]).to(self.device)
        else:
            assert False, 'Unknown encoder type'
        
        self.net = torch.nn.DataParallel(self.net).to(self.device)
        
        self.lr = config['lr']
        
        if config['optimizer']=="SGD":
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        elif config['optimizer']=="Adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,  milestones=self.epoch_step, gamma=0.1)  #30 failed, 60 did not try
        
        self.loss_type = config['loss_type'] #loss_function = torch.nn.CrossEntropyLoss()
        self.loss_function = Loss_function(self.loss_type)

        self.save_freq = config['save_freq']
        self.num_workers = config['num_workers']
        
        self.image_saver_png = SaveImage(output_dir=self.sample_dir,output_postfix="",output_ext=".png",separate_folder=False, scale=255)
        self.image_saver_nifty = SaveImage(output_dir=self.test_dir,output_postfix="",output_ext=".nii.gz",separate_folder=False, scale=255)
        
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
        
        #wandb.init(project="Tumour-segmentation", config=config)
        print(config)
        
        pin_memory = True

        ##### self.lr_decay = torch.optim.lr_scheduler.MultiStepLR(self.opt, [400])
        
        train_ds = CacheDataset(data=self.train_list, transform=self.train_tranforms,cache_rate=1.0, num_workers=self.num_workers)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=pin_memory)
        
        validate_ds = CacheDataset(data=self.validate_list, transform=self.train_tranforms,cache_rate=1.0, num_workers=self.num_workers)
        validate_loader = DataLoader(validate_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=pin_memory)
        
        post_pred = Compose([EnsureType(), AsDiscrete(threshold=0.5, to_onehot=2)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
        
        #lr= self.lr
        
        saver_epochs={}
        lowest_val_loss = 100
        best_pt_dice = -1
        best_metric_dice = -1
        best_metric_dice2 = -1
        best_metric_epoch = -1
        best_npv = -1
        best_tversky = -1
        
        metric_values = list()
        
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        
        recall = ConfusionMatrixMetric(include_background=False, metric_name="sensitivity", reduction="mean", compute_sample=False, get_not_nans=False)
        precision = ConfusionMatrixMetric(include_background=False, metric_name="precision", reduction="mean", compute_sample=False, get_not_nans=False)
        npv = ConfusionMatrixMetric(include_background=False, metric_name="negative_predictive_value", reduction="mean", compute_sample=False, get_not_nans=False)
        
        init=0
        epoch_loss_values=list()
        dice_val=list()
        npv_val=list()
        recall_val=list()
        precision_val=list()
        tversky_val=list()
        
        metric_values_neg = list()
        metric_values_pt = list()
        
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
            
            #if epoch >= self.epoch_step and np.mod(epoch,10)==0:
                #lr = lr/2
            
            #self.generator_optimizer.learning_rate.assign(lr) 
            step=0
            epoch_loss=0
            
            #wandb.watch(self.net)
            
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
                    
                
            print(" - lr: " + str(self.optimizer.param_groups[0]["lr"]))
            
            self.lr_scheduler.step()  
               
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            
            loss_mean = (sum(loss_step)/len(loss_step))

            print("epoch / training loss")
            print(epoch,loss_mean)

            loss_saver_train.append(loss_mean)
            
            torch.save(self.net.state_dict(), os.path.join(self.checkpoint_dir, "last_model.pth"))
            
            print("Start validation:")
            self.net.eval()
            
            loss_step_val = []
            tversky_step_val = []

            with torch.no_grad():
            
                counter = 0
                
                for batch_data_val in validate_loader:
                    
                    input_, gtv = batch_data_val['input'].to(self.device), batch_data_val['gtv'].to(self.device)
                    
                    output = self.net(input_)
                    output_=output[2]
                    
                    loss = self.loss_function(output, gtv) 
                    loss_step_val.append(loss.item())
                    
                    val_outputs = [post_pred(val_output) for val_output in decollate_batch(output_)]
                    labels = [post_label(label) for label in decollate_batch(gtv)]
                    
                    dice_metric(y_pred=val_outputs, y=labels)
                    dice_metric_batch(y_pred=val_outputs, y=labels)
                    
                    tversky_step_val.append(tversky(pred=val_outputs, target=labels).cpu().detach().numpy())
                    recall(y_pred=val_outputs, y=labels)
                    precision(y_pred=val_outputs, y=labels)
                    npv(y_pred=val_outputs, y=labels)
                    
                    
                loss_mean_val = (sum(loss_step_val)/len(loss_step_val))
                
                metric = dice_metric.aggregate().item()
                
                npv_value = npv.aggregate()[0].cpu().detach().numpy()[0]
                tversky_step_value=np.mean(tversky_step_val)
                
                dice_val.append(metric) 
                recall_val.append(recall.aggregate()[0].cpu().detach().numpy()[0])
                precision_val.append(precision.aggregate()[0].cpu().detach().numpy()[0])
                npv_val.append(npv_value)
                tversky_val.append(tversky_step_value)
                 
                metric_batch = dice_metric_batch.aggregate()
                
                metric_neg = metric_batch[0].item()
                metric_values_neg.append(metric_neg)
                metric_pt = metric_batch[1].item()
                metric_values_pt.append(metric_pt)
                
                dice_metric_batch.reset()
            
                
                #wandb.log({"epoch": epoch, "training loss": loss_mean, "validation loss": loss_mean_val, "dice val": metric, "recall val": recall.aggregate()[0].cpu().detach().numpy()[0], "precision val": precision.aggregate()[0].cpu().detach().numpy()[0], "npv val": npv_value, "tversky val": tversky_step_value})
            
                dice_metric.reset()
                recall.reset()
                precision.reset()
                npv.reset()
                
                print("epoch / validation loss")
                print(epoch,loss_mean_val)

                loss_saver_val.append(loss_mean_val)
                
                df = pd.DataFrame(list(zip(loss_saver_train, loss_saver_val, dice_val, metric_values_neg, metric_values_pt, recall_val, precision_val, npv_val, tversky_val)),
                                       columns =['Loss Train', 'Loss Val', 'Dice Val', 'Dice 0', 'Dice 1','Recall Val', 'Precision Val', 'Negative Predictive Val','Tversky Index Val'])
                
                df.to_excel(os.path.join(self.output_dir, 'training_metrics.xlsx'))
                
                if epoch > 1:
                    save_plots_pt(loss_saver_train, loss_saver_val,  dice_val, metric_values_pt, recall_val, self.output_dir) #save plot only pt dice
                
                if loss_mean_val < lowest_val_loss and epoch >= 100:
                    saver_epochs['loss val']=[loss_mean_val,epoch]
                    lowest_val_loss = loss_mean_val
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_dir, "lowest_val_loss_model.pth"))
                    print("saved new best metric network for val loss")
                
                if npv_value > best_npv and epoch >= 100:
                    saver_epochs['npv val']=[npv_value,epoch]
                    best_npv = npv_value
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_dir, "best_npv_model.pth"))
                    print("saved new best metric network for NPV")
                    
                if tversky_step_value > best_tversky and epoch >= 100:
                    saver_epochs['tversky val']=[tversky_step_value,epoch]
                    best_tversky = tversky_step_value
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_dir, "best_tversky_model.pth"))
                    print("saved new best metric network for tversky index")
                
                if metric_pt > best_pt_dice:
                    saver_epochs['best dice pt']=[metric_pt,epoch]
                    best_pt_dice = metric_pt
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_dir, "best_ptDice.pth"))
                    print("saved new best metric network for pt DICE")
                
                if metric > best_metric_dice:
                    saver_epochs['best dice val']=[metric,epoch]
                    best_metric_dice = metric
                    best_metric_epoch = epoch
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_dir, "best_dice_model.pth"))
                    print("saved new best metric network for DICE")
                    
                if metric > best_metric_dice2 and metric < best_metric_dice and epoch >= 100:
                    saver_epochs['best dice val 2']=[metric,epoch]
                    best_metric_dice2 = metric
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_dir, "best_dice_model2.pth"))
                    print("saved new best metric 2 network for DICE")

                print(
                    f"current epoch: {epoch} current Dice: {metric:.4f}"
                )
                
                overview = pd.DataFrame(saver_epochs)
                overview.to_excel(os.path.join(self.output_dir, 'summary_metrics.xlsx'))
                
                total_time = time() - start
                #wandb.log({"time": total_time})

        print(f"train completed, best_metric: {best_metric_dice:.4f} at epoch: {best_metric_epoch}")

        total_time = time() - start
        #wandb.log({"time": total_time})
        #wandb.finish()
        
        print(f"train completed, best_metric: {best_metric_dice:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
        
        
    def load(self, checkpoint_dir, which_check):
        
        print(" [*] Reading checkpoint...")
        
        if which_check=="best_dice":
            print(" [*] Best dice checkpoint loading!")
            self.net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_dice_model.pth")))
            return True
        elif which_check=="best_dice2":
            print(" [*] Best dice 2 checkpoint loading!")
            self.net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_dice_model2.pth")))
            return True
        elif which_check=="last":
            print(" [*] Last checkpoint loading!")
            self.net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "last_model.pth")))
            return True
        elif which_check=="best_npv":
            print(" [*] Best Tversky checkpoint loading!")
            self.net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_npv_model.pth")))
            return True
        elif which_check=="best_tversky":
            print(" [*] Best Tversky checkpoint loading!")
            self.net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_tversky_model.pth")))
            return True
        elif which_check=="low_val":
            print(" [*] Lowest val loss checkpoint loading!")
            self.net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "lowest_val_loss_model.pth"), map_location=torch.device(self.device)))
            return True
        else: 
            False
        
    def sample_model(self, epoch, idx, validate_loader):
        
        pin_memory = True
        self.net.eval()
        
        with torch.no_grad():
        
            element = random.choices(self.validate_list, k=3)
    
            validate_ds = CacheDataset(data=element, transform=self.train_tranforms,cache_rate=1.0)
            validate_loader = DataLoader(validate_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=pin_memory)
            
            batch_data_val =  first(validate_loader)  
            input_, gtv = batch_data_val['input'].to(self.device), batch_data_val['gtv'].to(self.device)
    
            output = self.net(input_)
            
            save_img(gtv, self.image_saver_png,'GTOriginal_{:02d}_{:04d}.png'.format(epoch, idx))
            save_img(output[2], self.image_saver_png,'GTPredicted_{:02d}_{:04d}.png'.format(epoch, idx))
        
    def test(self, config):
        
        from monai.metrics import ROCAUCMetric,ConfusionMatrixMetric, compute_confusion_matrix_metric
        
        pin_memory = True
        
        dice_metric = DiceMetric(include_background=False, get_not_nans=False)
        recall = ConfusionMatrixMetric(include_background=False, metric_name="sensitivity", compute_sample=False, get_not_nans=False)
        precision = ConfusionMatrixMetric(include_background=False, metric_name="precision", compute_sample=False, get_not_nans=False)
        
        post_pred = Compose([EnsureType(), AsDiscrete(threshold=0.5, to_onehot=2)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
        
        if config['phase']=="train":
            ds = CacheDataset(data=self.train_list, transform=self.train_tranforms,cache_rate=1.0)
        elif config['phase']=="test":
            ds = CacheDataset(data=self.validate_list, transform=self.test_tranforms,cache_rate=1.0)
        
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=pin_memory)
        
        checkpoint_dir = self.checkpoint_dir
        
        if self.load(checkpoint_dir, which_check=self.loading_checkpoint):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            
        dice_one=list()
        recall_one=list()
        precision_one=list()
        pathss=list()
        
        self.net.eval()
        with torch.no_grad():
            for batch_data_val in loader:
                input_, slice_, id_, gtv = batch_data_val['input'].to(self.device), batch_data_val['slice'].to(self.device), batch_data_val['ID'], batch_data_val['gtv'].to(self.device)
                
                output = self.net(input_)
                
                val_outputs = [post_pred(val_output) for val_output in decollate_batch(output[2])]
                labels = [post_label(label) for label in decollate_batch(gtv)]
                
                dice_calculate = dice_metric(y_pred=val_outputs, y=labels).cpu().detach().numpy()[0]
                print(dice_metric(y_pred=val_outputs, y=labels).cpu().detach().numpy())
                
                conf_metrix1=recall(y_pred=val_outputs, y=labels)
                conf_metrix2=precision(y_pred=val_outputs, y=labels)
                
                sensitivity=compute_confusion_matrix_metric("sensitivity",conf_metrix1).cpu().detach().numpy()[0]
                precision_=compute_confusion_matrix_metric("precision",conf_metrix2).cpu().detach().numpy()[0]
                
                dice_one.append(dice_calculate[0])
                recall_one.append(sensitivity[0])
                precision_one.append(precision_[0])
                
                pathss.append('{0}.nii.gz'.format(id_[0] + '_seq_' + str(slice_[0].cpu().numpy())))
                
                save_img(gtv.squeeze(), self.image_saver_nifty, '{0}_gtv.nii.gz'.format(id_[0] + '_seq_' + str(slice_[0].cpu().numpy())))
                save_img(output[2].squeeze(), self.image_saver_nifty, '{0}.nii.gz'.format(id_[0] + '_seq_' + str(slice_[0].cpu().numpy())))
        
        df = pd.DataFrame(list(zip(pathss, dice_one, recall_one, precision_one)),columns =['paths','Dice Primary Tumor','Recall Primary Tumor','Precision Primary Tumor'])
        
        df.to_excel(os.path.join(self.output_dir, 'Dice_validation_val.xlsx'))
        
        
    def test_all(self, config):
        
        images_already_there = [k.split('_')[0] for k in os.listdir(self.test_dir) if '_'+self.direction+'.nii.gz' in k] ######## COMMENT HERE
        print("Already tested:")
        print(len(images_already_there))
        
        data_folder=config['data_folder'] #'/data/pg-umcg_opc_outpred/Data/UMCG_seg/'
        
        pin_memory = True
        
        checkpoint_dir = self.checkpoint_dir
        
        if self.load(checkpoint_dir, which_check=self.loading_checkpoint):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        split_file = os.path.join(pathlib.Path(config['experiment_dir']), pathlib.Path(config['split_file']))

        split_data = read_split_file(split_file)
        
        patients_testing = split_data['validate']
        patients_testing = [i for i in patients_testing if i not in images_already_there]   ######## COMMENT HERE
        no_found = list()
        
        for patient in patients_testing:
            
            patient_list=list()
            
            if os.path.isfile(os.path.join(data_folder, patient+"_ct.nii.gz"))==True and os.path.isfile(os.path.join(data_folder, patient+"_pt.nii.gz"))==True:
                for i in range(0,self.image_size-2):
                    patient_list.append([patient, os.path.join(data_folder, patient+"_ct.nii.gz"), os.path.join(data_folder, patient+"_pt.nii.gz"), os.path.join(data_folder, patient+"_ct_GTVtot.nii.gz"), i])    
                    
                dataset_p = dataset_prep(patient_list)
        
                ds = CacheDataset(data=dataset_p, transform=self.test_tranforms,cache_rate=1.0)
                loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=pin_memory)
                
                self.net.eval()
                with torch.no_grad():
                
                    for batch_data_val in loader:
                    
                        input_, slice_, id_, gtv = batch_data_val['input'].to(self.device), batch_data_val['slice'].to(self.device), batch_data_val['ID'], batch_data_val['gtv'].to(self.device)
                        output = self.net(input_)
                        #save_img(gtv.squeeze(), self.image_saver_nifty, '{0}_gtv.nii.gz'.format(id_[0] + '_seq_' + str(slice_[0].cpu().numpy())))
                        save_img(output[2].squeeze(), self.image_saver_nifty, '{0}.nii.gz'.format(id_[0] + '_seq_' + str(slice_[0].cpu().numpy())))
                
                operation_comb=combine_slices(self.test_dir, patient, self.direction, self.image_size)
                
                path_=os.path.join(self.test_dir,'{0}.nii.gz'.format(patient + '_'+self.direction))
                
                _image=list()
        
                for i in range(0,self.image_size):
                    _image.append(operation_comb[i])
                
                out_ = sitk.GetImageFromArray(np.stack(_image))
                sitk.WriteImage(out_,path_)      
                          
            else:
                no_found.append(patient)
                print("No CT and/or PET was found for patient " + patient)
                    
        print("Patients No found:")
        print(no_found)    
        

            
        
        