import glob
import argparse
import random
import os
import json
import sys
import numpy as np
from PIL import Image 
import SimpleITK as sitk
sys.dont_write_bytecode = True
from scipy.stats import kurtosis
from scipy.stats import skew
from matplotlib import cm
import pandas as pd


def entropy(signal):
    lensig=signal.size
    symset=list(set(signal))
    numsym=len(symset)
    propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
    ent=np.sum([p*np.log2(1.0/p) for p in propab])
    return ent

def en(im_arr):
    
    im = Image.fromarray(np.uint8(cm.plasma(im_arr)*255)) #entropy
    
    greyIm=im.convert('L')
    greyIm = np.array(greyIm)

    N=5
    S=greyIm.shape
    E=np.array(greyIm)
    for row in range(S[0]):
            for col in range(S[1]):
                    Lx=np.max([0,col-N])
                    Ux=np.min([S[1],col+N])
                    Ly=np.max([0,row-N])
                    Uy=np.min([S[0],row+N])
                    region=greyIm[Ly:Uy,Lx:Ux].flatten()
                    E[row,col]=entropy(region)
    return np.mean(E)


def check_GT(path,i,axis,threshold): 
    
    im = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)[:144,:144,:144]
    
    if axis=="x":
        ima = im[i:i+3,:,:]
    elif axis=="y":
        ima = im[:,i:i+3,:]
    elif axis=="z":
        ima = im[:,:,i:i+3]
    
    ima=ima.reshape(144*3,144)
    tumor=np.sum(ima==1)
        
    tot_size = ima.size
        
    amm_tum = np.round(tumor/tot_size,3).astype(np.float32)
        
    if amm_tum >= threshold:
        return 1
    else:
        return 0


def pet_info_extr(pathPET,pathGT,axis,ID,th): 
    
    df = pd.DataFrame(columns = ["ID", "slice", "meanSUV", "maxSUV", "minSUV", "std", "skewness", "kurtosis", "entropy","tumor"])
    
    im = sitk.GetArrayFromImage(sitk.ReadImage(pathPET)).astype(np.float32)[:144,:144,:144]
    im = im[:144,:144,:144]
    
    if axis=="x":
        num_slices = im.shape[0]
    elif axis=="y":
        num_slices = im.shape[1]
    elif axis=="z":
        num_slices = im.shape[2]
        
    for i in list(range(0, num_slices-2)):
        if axis=="x":
            pet = im[i:i+3,:,:]
        elif axis=="y":
            pet = im[:,i:i+3,:]
        elif axis=="z":
            pet = im[:,:,i:i+3]
            
        pet=pet.reshape(144*3,144)
        slice_ = i
        maxv = np.max(pet)
        mean = np.mean(pet)
        minv = np.min(pet)
        std = np.std(pet)
        sk = np.mean(skew(pet))
        kurt = np.mean(kurtosis(pet))
        entropy_mean = en(pet)
        tumor=check_GT(pathGT,i,axis,th)
        
        d={'ID':ID,'slice': slice_, 'maxSUV': maxv, 'minSUV': minv, 'meanSUV': mean, 'std': std, 'skewness': sk, 'kurtosis':kurt, 'entropy':entropy_mean, 'tumor':tumor}
        
        df=df.append(d, ignore_index=True)
        print(df)
        
    return df
    
  
def gtv_info_extract(path,axis,ID): 
    
    df = pd.DataFrame(columns = ["ID", "tumor_amm"])
    
    im = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)[:144,:144,:144]
    
    im = im[:144,:144,:144]
    
    if axis=="x":
        num_slices = im.shape[0]
    elif axis=="y":
        num_slices = im.shape[1]
    elif axis=="z":
        num_slices = im.shape[2]
    
    for i in list(range(0, num_slices-2)):
    
        if axis=="x":
            ima = im[i:i+3,:,:]
        elif axis=="y":
            ima = im[:,i:i+3,:]
        elif axis=="z":
            ima = im[:,:,i:i+3]
        
        ima=ima.reshape(144*3,144)
        tumor=np.sum(ima==1)
            
        tot_size = ima.size
            
        amm_tum = np.round(tumor/tot_size,3).astype(np.float32)
        
        d={'ID':ID,'tumor_amm':amm_tum}
        
        df=df.append(d, ignore_index=True)
        print(df)
        
    return df
   
def get_infoGTV_from_file(list_patients, file,path, output_folder, type_gtv, axis):

    """
    @param images_path: list of images paths. Each image name is in the format P[patient number]_[organ]_(...).nii.gz
    @return: dictionary of images used for training and testing for both A and B, A and B have two different split files
    """

    df = file
    train_patients=list_patients
    train_patients.sort()

    for patient in train_patients:
        
        GT_path = [s for s in path if str(patient + "_ct_"+str(type_gtv)+".nii.gz") in s][0]
        
        patient_ID = patient
        
        d = gtv_info_extract(GT_path,axis,patient_ID)
        df=df.append(d, ignore_index=True)
        
        print(df)
        print(df.groupby('ID').count())
        df.to_csv(output_folder+'/gtv_tmp'+str(axis)+'.csv', index = False, header=True)
        
    return df     

def get_infoPT_from_file(list_patients, file,path, output_folder, type_gtv, axis):

    """
    @param images_path: list of images paths. Each image name is in the format P[patient number]_[organ]_(...).nii.gz
    @return: dictionary of images used for training and testing for both A and B, A and B have two different split files
    """
    #df = pd.DataFrame(columns = ["ID", "slice", "meanSUV", "maxSUV", "minSUV", "std", "skewness", "kurtosis", "entropy","tumor"])

    df = file
    train_patients=list_patients
    print(list_patients)
    train_patients.sort()

    for patient in train_patients:
        
        GT_path = [s for s in path if str(patient + "_ct_"+str(type_gtv)+".nii.gz") in s][0]
        PET_path = [s for s in path if str(patient + "_pt.nii.gz") in s][0]
        
        patient_ID = patient
        
        d = pet_info_extr(PET_path,GT_path,axis,patient_ID,th=0.025)
        df=df.append(d, ignore_index=True)
        
        print(df)
        print(df.groupby('ID').count())
        df.to_csv(output_folder+'/table_tmp'+str(axis)+'.csv', index = False, header=True)
        
    return df

def read_split_file(path):
    
    with open(path, 'r') as inf:
        dict_from_file = eval(inf.read())
        
    return dict_from_file


def main(directory, owd, type_gtv, axis):
    
    images_path = glob.glob(str(directory + '/*.nii.gz'))
    
    folder_path = 'pet_info_test' #fol_name #+ str(splits+1)
    
    output_folder=os.path.join(owd,folder_path)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    os.chdir(output_folder)

    split=read_split_file(owd + 'patient_split.json')
    file=pd.DataFrame()
    
    file=pd.read_csv('table_tmp'+axis+'.csv')
    patients_done=list(set(file['ID']))
    patients_done.sort()
    
    print("Patients done:")
    print(patients_done)
    
    to_do=[k for k in split['training'] if k not in patients_done]
    to_do.sort()
    print("------")
    
    print("Patients to do:")
    print(to_do)
            
    print('Training Data info:')
    
    data_= get_infoPT_from_file(to_do, file,images_path, output_folder, type_gtv, axis)
    data_ = data_.append(file)
    data_.to_csv(os.path.join(owd,folder_path,'training_data_'+str(axis)+'.csv'), index = False, header=True)
    


def main2(directory, owd, type_gtv, axis):

    images_path = glob.glob(str(directory + '/*.nii.gz'))
    
    folder_path = 'pet_info_test' #fol_name #+ str(splits+1)
    
    output_folder=os.path.join(owd,folder_path)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    os.chdir(output_folder)

    split=read_split_file(owd + 'patient_split.json')
    file=pd.DataFrame()
            
    print('Training Data info:')
    
    data_= get_infoGTV_from_file(split['training'], file,images_path, output_folder, type_gtv, axis)
    data_ = data_.append(file)
    data_.to_csv(os.path.join(owd,folder_path,'training_data_gtv_'+str(axis)+'.csv'), index = False, header=True)
            
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', dest='dataset_dir',
                        default="/data/p302386/hecktor/umcg_data/UMCG_automseg/", #default="/data/p302386/hecktor/data/resampled"
                        help='path of the dataset')

    parser.add_argument('--model_dir', dest='model_dir',
                        default="/data/p302386/hecktor/monai_model/experiment_def2/", #
                        help='path of the model directory')
                        
    parser.add_argument('--type_gtv', dest='type_gtv',
                        default="GTVpt",
                        help='type GTV to use for extracting slices')
    
    
    parser.add_argument('--axis', dest='axis',
                        default="z",
                        help='axis for slice extraction')

    args = parser.parse_args([])
    
    print(args)
    
    main(directory=args.dataset_dir, owd = args.model_dir, type_gtv = args.type_gtv, axis = args.axis)

