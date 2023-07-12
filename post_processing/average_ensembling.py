
import os
import numpy as np
import SimpleITK as sitk

def read_split_file(path):

    with open(path, 'r') as inf:
        dict_from_file = eval(inf.read())
        
    return dict_from_file


path_data = '/data/p302386/hecktor/umcg_data/UMCG_automseg/' #'/data/p302386/hecktor/data/resampled/'#'/data/p302386/hecktor/umcg_data/UMCG_automseg/' #"/data/p302386/hecktor/umcg_data/UMCG_14_17_resampled/"
experiment_folder = '/data/p302386/hecktor/monai_model/experiment_def_new'
segmentation_folder = 'testing_images'

split_file = os.path.join(experiment_folder, 'patient_split.json') #'/data/p302386/hecktor/monai_model/experiment_1_new/split_hecktor.txt' #'/data/p302386/hecktor/srcmodel/split_14_17/test_split.txt' #os.path.join(experiment_folder, 'patient_split.json')
split_file_read=read_split_file(split_file)
patients_IDs=split_file_read["validate"]
patients_IDs.sort()

AX = ['x','y','z']
folds = 3


for axis in AX:
    print("Axis = " + axis)
    for ID in patients_IDs:
        print("Processing patient: " + ID)
        tot=0
        
        for num in range(1,folds+1):
        
            path_ = os.path.join(experiment_folder,axis+'_'+str(num),'test','{0}.nii.gz'.format(ID + '_'+axis))
        
            image = sitk.GetArrayFromImage(sitk.ReadImage(path_))
            
            tot = image + tot
        
        out = sitk.GetImageFromArray(tot/folds)
        
        path_dest = os.path.join(experiment_folder,segmentation_folder,'{0}.nii.gz'.format(ID + '_'+axis+'.nii.gz'))
        
        sitk.WriteImage(out, path_dest)
    
    