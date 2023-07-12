import os
import tempfile
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Any, Mapping, Hashable
from PIL import Image
import random
import numpy as np
import SimpleITK as sitk
random.seed(30)
from monai.transforms import (
    MapTransform,
    ConcatItemsd,
    AddChanneld,
    Compose,
    LoadImaged,
    ThresholdIntensityd,
    NormalizeIntensityd,
    ToTensord,
    Resized,
)
from monai.data.nifti_saver import NiftiSaver


def read_split_file(path):

    with open(path, 'r') as inf:
        dict_from_file = eval(inf.read())
        
    return dict_from_file


def dataset_prep(lists):
    dataset_train=[]
    for lista in lists:
        dataset_train.append(
            {'ct':lista[1],
             'pet':lista[2],
             'gtv':lista[3],
             'slice':lista[4],
             'ID':lista[0]
            })
    return dataset_train

class Create_sequences(MapTransform):
    
    def __init__(self, keys, seq = 3, plane = "x"):
        
        super().__init__(keys)
        self.sequences = seq
        self.plane=plane
        
    def slicing(self, img, starting):
        
        indx=starting
        sequence=[]
        
        for i in range(indx, indx+self.sequences):
            
            if self.plane=="x":
                ima = np.rot90(img[:,:,i],3)
            elif self.plane=="y":
                ima = np.rot90(img[:,i,:])
            elif self.plane=="z":
                ima = np.rot90(img[i,:,:])
                
            sequence.append(ima)
        
        return np.dstack(sequence)    

    def __call__(self, dictionary):
        
        dictionary = dict(dictionary)
        starting = dictionary["slice"]
        
        for key in self.keys:
            dictionary[key] = self.slicing(dictionary[key],starting) #self.create_sequence.
        
        return dictionary


class Resized_mine(MapTransform):
    
    def __init__(self, keys, spatial_size):
        
        super().__init__(keys)
        self.spatial_size = spatial_size
        
    def resize(self, img):
        
        if np.shape(img)!=self.spatial_size:
            return img[:self.spatial_size[0],:self.spatial_size[1],:self.spatial_size[2]] #self.resizer(img) 
        else:
            return img
             

    def __call__(self, dictionary):
        
        dictionary = dict(dictionary)
        
        for key in self.keys:
            dictionary[key] = self.resize(dictionary[key]) 
        
        return dictionary


def initialize_transform(images_keys, plane_, level, window, image_size, ct_norm="z_norm", pet_norm="z_norm", test=False):
    
    normalize_ct={
        "z_norm": NormalizeIntensityd(("ct"), subtrahend=None, divisor=None, nonzero=False, channel_wise=False, allow_missing_keys=False),
        "min_max_norm": NormalizeIntensityd(("ct"), subtrahend=None, divisor=None, nonzero=False, channel_wise=False, allow_missing_keys=False)
    }
    normalize_pet={
        "z_norm": NormalizeIntensityd(("pet"), subtrahend=None, divisor=None, nonzero=False, channel_wise=False, allow_missing_keys=False),
        "min_max_norm": NormalizeIntensityd(("pet"), subtrahend=None, divisor=None, nonzero=False, channel_wise=False, allow_missing_keys=False)
    }
    
    load_seq=[
        LoadImaged(keys=images_keys),
        Resized_mine(keys=images_keys, spatial_size=(image_size,image_size,image_size)),
    ]
    
    #if test==False:
        #load_seq.append(RandFlipd(keys=images_keys, prob=0.5, spatial_axis=2))
    
    max_ = 200 #level + window/2
    min_ = -200 #level - window/2
    
    pre_processing_ct=[
        ThresholdIntensityd(("ct"), threshold=min_, above=True, cval=min_, allow_missing_keys=False),
        ThresholdIntensityd(("ct"), threshold=max_, above=False, cval=max_, allow_missing_keys=False),
    ]

    pre_processing_pet=[
        ThresholdIntensityd(("pet"), threshold=0, above=True, cval=0.0, allow_missing_keys=False)
    ]
    
    try:
        pre_processing_ct.append(normalize_ct[ct_norm])
    except KeyError:
        "CT normalization technique not available!"
        
    try:
        pre_processing_pet.append(normalize_pet[pet_norm])
    except KeyError:
        "PET normalization technique not available!"
        
    sequence_prep=[
        Create_sequences(keys=images_keys, plane=plane_),
        AddChanneld(keys=images_keys), 
        ConcatItemsd(keys=["ct","pet"], name="input"),
        ToTensord(keys=["input", "gtv"])
    ]
    
    train_tranforms = Compose(load_seq+pre_processing_ct+pre_processing_pet+sequence_prep)
    
    return train_tranforms
  
  
class ConvertToMultiChannelBasedOnClassesd(MapTransform):
       
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            
            result.append((d[key] == 0))
               # merge label 2 and label 3 to construct TC
            result.append((d[key] == 1))
               # merge labels 1, 2 and 3 to construct WT
            result.append((d[key] == 2))

            d[key] = np.stack(result, axis=0).astype(np.float32)

        return d 
  
    
def initialize_transform_lynph(images_keys, plane_, level, window, image_size, ct_norm="z_norm", pet_norm="z_norm", test=False):
    
    normalize_ct={
        "z_norm": NormalizeIntensityd(("ct"), subtrahend=None, divisor=None, nonzero=False, channel_wise=False, allow_missing_keys=False),
        "min_max_norm": NormalizeIntensityd(("ct"), subtrahend=None, divisor=None, nonzero=False, channel_wise=False, allow_missing_keys=False)
    }
    normalize_pet={
        "z_norm": NormalizeIntensityd(("pet"), subtrahend=None, divisor=None, nonzero=False, channel_wise=False, allow_missing_keys=False),
        "min_max_norm": NormalizeIntensityd(("pet"), subtrahend=None, divisor=None, nonzero=False, channel_wise=False, allow_missing_keys=False)
    }
    
    load_seq=[
        LoadImaged(keys=images_keys),
        Resized_mine(keys=images_keys, spatial_size=(image_size,image_size,image_size)),
    ]
    
    #if test==False:
        #load_seq.append(RandFlipd(keys=images_keys, prob=0.5, spatial_axis=2))
    
    max_ = 200 #level + window/2
    min_ = -200 #level - window/2
    
    pre_processing_ct=[
        ThresholdIntensityd(("ct"), threshold=min_, above=True, cval=min_, allow_missing_keys=False),
        ThresholdIntensityd(("ct"), threshold=max_, above=False, cval=max_, allow_missing_keys=False),
    ]

    pre_processing_pet=[
        ThresholdIntensityd(("pet"), threshold=0, above=True, cval=0.0, allow_missing_keys=False)
    ]
    
    try:
        pre_processing_ct.append(normalize_ct[ct_norm])
    except KeyError:
        "CT normalization technique not available!"
        
    try:
        pre_processing_pet.append(normalize_pet[pet_norm])
    except KeyError:
        "PET normalization technique not available!"
        
    sequence_prep=[
        Create_sequences(keys=images_keys, plane=plane_),
        ConvertToMultiChannelBasedOnClassesd(keys="gtv"),
        AddChanneld(keys=["ct","pet"]),
        ConcatItemsd(keys=["ct","pet"], name="input"),
        ToTensord(keys=["input", "gtv"])
    ]
    
    train_tranforms = Compose(load_seq+pre_processing_ct+pre_processing_pet+sequence_prep)
    
    return train_tranforms


def save_img(image, image_saver, name):
    
    image = torch.squeeze(image,0)
    image = torch.cat(torch.unbind(image,2),1)
    image_saver(image,{'filename_or_obj' : name})#
    

def save_img_lymph(image, image_saver, name):
    
    if int(len(image.shape))>4:
    
        for i in range(0,int(image.shape[0])):
        
            #image = torch.squeeze(image,0)
            new_im = image[i,:,:,:,:]
            new_im = torch.cat(torch.unbind(new_im ,3),1)
            
            #print(torch.max(image[0,:,:])) #negative class
            #print(torch.max(image[1,:,:])) #lymph nodes
            #print(torch.max(image[2,:,:])) #primary tumor
            
            new_name = name.split('.')[0]+'_'+str(i)+'_'+name.split('.')[-1]
            
            image_saver(new_im,{'filename_or_obj' : new_name})#
            
    else:
      
        image = torch.squeeze(image,0)
        new_im = torch.cat(torch.unbind(image ,3),1)
        
        new_name = name.split('.')[0]
        image_saver(new_im,{'filename_or_obj' : new_name})#
    
    
def combine_slices(TEST_FOLDER, patient_ID, direction, image_size):
    
    mean={}
    
    sequences=list(set([int(k.split('.')[0].split('_')[-1]) for k in os.listdir(TEST_FOLDER) if (k.split('_')[0]==patient_ID and  'seq' in k and 'gtv' not in k and direction not in k.split('.')[0])]))
    sequences.sort()
    last=sequences[-1]
    sequences=sequences+[last+1,last+2]
        
    print('Patient ' + patient_ID)
        
    dicts=list(ensamble_exam(TEST_FOLDER,patient_ID, direction, image_size))
    
    big_dict = {}
    
    for k in range(len(sequences)):
        big_dict[k] = [b for a,b in dicts if a==k]

    big_dict_mean={}
        
    for k in big_dict:
        summa=0
        for im in big_dict[k]:
            summa=summa+im
        big_dict_mean[k]=summa/len(big_dict[k])
        
    return big_dict_mean
    

def combine_slices_lymph(TEST_FOLDER, patient_ID, direction, image_size):
    
    mean={}
    
    sequences=list(set([int(k.split('.')[0].split('_')[-1]) for k in os.listdir(TEST_FOLDER) if (k.split('_')[0]==patient_ID and  'seq' in k and 'gtv' not in k and direction not in k.split('.')[0])]))
    sequences.sort()
    last=sequences[-1]
    sequences=sequences+[last+1,last+2]
        
    print('Patient ' + patient_ID)
         
    dicts=list(ensamble_exam_lymph(TEST_FOLDER,patient_ID, direction, image_size))
    
    big_dict = {}
    
    for k in range(len(sequences)):
        big_dict[k] = [b for a,b in dicts if a==k]

    big_dict_mean={}
        
    for k in big_dict:
        summa=0
        for im in big_dict[k]:
            
            summa=summa+im
        big_dict_mean[k]=summa/len(big_dict[k])
        
    return big_dict_mean

    
def ensamble_exam(test_FOLDER,patient_ID, direction, image_size): #,where
    
    sequences=list(set([int(k.split('.')[0].split('_')[-1]) for k in os.listdir(test_FOLDER) if (k.split('_')[0]==patient_ID and  'seq' in k and 'gtv' not in k and direction not in k.split('.')[0])]))
    sequences.sort()
    
    totale=[]
    
    positions=[]
    images=[]
    
    for pos in sequences:
    
        path_=test_FOLDER+'/'+patient_ID+'_seq_'+str(pos)+'.nii.gz'

        np_image_prediction = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(path_)))

        im1=np_image_prediction[:image_size,:image_size] #pos
        im2=np_image_prediction[:image_size,image_size:int(np_image_prediction.shape[1]/3*2)] #pos+1
        im3=np_image_prediction[:image_size,int(np_image_prediction.shape[1]/3*2):] #pos+2
        
        positions=positions+[pos,pos+1,pos+2]
        images=images+[im1,im2,im3]
        
        os.remove(path_)
        
    return zip(positions,images)
    
    
def ensamble_exam_lymph(test_FOLDER,patient_ID, direction, image_size): #,where
    
    sequences=list(set([int(k.split('.')[0].split('_')[-1]) for k in os.listdir(test_FOLDER) if (k.split('_')[0]==patient_ID and  'seq' in k and 'gtv' not in k and direction not in k.split('.')[0])]))
    sequences.sort()
    
    totale=[]
    
    positions=[]
    images=[]
    
    for pos in sequences:
    
        path_=test_FOLDER+'/'+patient_ID+'_seq_'+str(pos)+'.nii.gz'

        np_image_prediction = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(path_)))
        
        im1=np_image_prediction[:,:image_size,:image_size] #pos
        im2=np_image_prediction[:,:image_size,image_size:int(np_image_prediction.shape[2]/3*2)] #pos+1
        im3=np_image_prediction[:,:image_size,int(np_image_prediction.shape[2]/3*2):] #pos+2
        
        positions=positions+[pos,pos+1,pos+2]
        images=images+[im1,im2,im3]
        
        os.remove(path_)
        
    return zip(positions,images)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
