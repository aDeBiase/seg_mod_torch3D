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
from segmentation.eval_plot import save_plots

pin_memory = True
        
def read_split_file(path):

    with open(path, 'r') as inf:
        dict_from_file = eval(inf.read())
        
    return dict_from_file
    

        
split_file = "/data/split_train/image_split_x1.json"
        
split_data = read_split_file(split_file)
        
train_list = dataset_prep(split_data['training'])[:2]

image_keys=('ct', 'pet', 'gtv')

train_tranforms = initialize_transform(image_keys, config['direction'], level=60, window=350, ct_norm="z_norm", pet_norm="z_norm", test=False)
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      
train_ds = CacheDataset(data=train_list, transform=train_tranforms,cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=pin_memory)





