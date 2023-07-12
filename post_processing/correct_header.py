
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.ndimage.measurements import label
import pandas as pd
import os


data_path = '/data/p302386/hecktor/umcg_data/UMCG_automseg/' #"/data/p302386/hecktor/umcg_data/UMCG_14_17_resampled/"# '/data/p302386/hecktor/data/resampled/'

experiment_folder = '/data/p302386/hecktor/monai_model/experiment_def_new' #'/data/p302386/hecktor/monai_model/experiment_1_new'

multi_view_segmentation_folder = 'reconstructed_volumes'#'reconstructed_volumes_validation'
segmentation_folder = 'testing_images' #'testing_heckt'#'testing_images_3D' #'z_3/test'
 
pred_path = os.path.join(experiment_folder,segmentation_folder)#os.path.join(experiment_folder,segmentation_folder) #
dest_path = pred_path #os.path.join(experiment_folder,'validation_images')

patientsID = os.listdir(pred_path)

AX = ['x','y','z'] #,'y','z'
final_convert = False

patientsID=  list(set([x.split('_')[0] for x in patientsID]))
#patientsID=  list(set([x.split('_')[0].split('.')[0] for x in patientsID]))
patientsID.sort()

print (patientsID)

for pID in patientsID:

  for axis in AX:
  
    print (pID)
    
    ct = sitk.ReadImage(data_path + '/' + str(pID) + '_ct.nii.gz')
    ct_array =  sitk.GetArrayFromImage(ct)
    
    if final_convert==False:
        seg = sitk.ReadImage(pred_path + '/' + str(pID) + '_'+str(axis)+'.nii.gz')
    else:
        seg = sitk.ReadImage(pred_path + '/' + str(pID) + '.nii.gz')
        
    seg_array = sitk.GetArrayFromImage(seg)
  
    seg_array[seg_array<0.005]  = 0
    
    ct_array_zero = np.zeros_like(ct_array)
    
    if axis=='x':
      seg_array=np.flip(seg_array,2)
      ct_array_zero[: seg_array.shape[0], :  seg_array.shape[1], :  seg_array.shape[2]] = seg_array
    elif axis=='y':
      seg_array=np.flip(seg_array,1)
      seg_array=seg_array.transpose(1, 0, 2)
      ct_array_zero[: seg_array.shape[0], :  seg_array.shape[1], :  seg_array.shape[2]] = seg_array
    elif axis=='z':
      seg_array=np.flip(seg_array,1)
      seg_array=seg_array.transpose(1, 2, 0)
      ct_array_zero[: seg_array.shape[0], :  seg_array.shape[1], :  seg_array.shape[2]] = seg_array
    
    seg_threshold = sitk.GetImageFromArray(ct_array_zero)
    seg_threshold.CopyInformation(ct)
    sitk.WriteImage(seg_threshold, dest_path + '/' + str(pID) + '_'+str(axis)+'corr.nii.gz')  
    
    