import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.ndimage.measurements import label
import pandas as pd
import os


data_path = '/data/DATA_automseg_lymph/' 

experiment_folder = '/data/model/trial/' 

multi_view_segmentation_folder = 'reconstructed_volumes'
segmentation_folder = 'test' 
 
pred_path = os.path.join(experiment_folder,segmentation_folder)
dest_path = os.path.join(experiment_folder,'test_sep') 

patientsID = os.listdir(pred_path)

AX = ['x'] #,'y','z'
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
    
    lymph_nodes = seg_array[1,:,:,:]
    primary_tumor = seg_array[2,:,:,:]
    
    ct_array_one = np.zeros_like(ct_array)
    ct_array_two = np.zeros_like(ct_array)
    
    if axis=='x':
      
      lymph_nodes=np.flip(lymph_nodes,2)
      ct_array_one[: lymph_nodes.shape[0], :  lymph_nodes.shape[1], :  lymph_nodes.shape[2]] = lymph_nodes
      
      primary_tumor=np.flip(primary_tumor,2)
      ct_array_two[: primary_tumor.shape[0], :  primary_tumor.shape[1], :  primary_tumor.shape[2]] = primary_tumor
      
    elif axis=='y':
      
      lymph_nodes=np.flip(lymph_nodes,1)
      lymph_nodes=lymph_nodes.transpose(1, 0, 2)
      ct_array_one[: lymph_nodes.shape[0], :  lymph_nodes.shape[1], :  lymph_nodes.shape[2]] = lymph_nodes
      
      primary_tumor=np.flip(primary_tumor,1)
      primary_tumor=primary_tumor.transpose(1, 0, 2)
      ct_array_two[: primary_tumor.shape[0], :  primary_tumor.shape[1], :  primary_tumor.shape[2]] = primary_tumor
      
    elif axis=='z':
    
      lymph_nodes=np.flip(lymph_nodes,1)
      lymph_nodes=lymph_nodes.transpose(1, 2, 0)
      ct_array_one[: lymph_nodes.shape[0], :  lymph_nodes.shape[1], :  lymph_nodes.shape[2]] = lymph_nodes
      
      primary_tumor=np.flip(primary_tumor,1)
      primary_tumor=primary_tumor.transpose(1, 2, 0)
      ct_array_two[: primary_tumor.shape[0], :  primary_tumor.shape[1], :  primary_tumor.shape[2]] = primary_tumor
      
    
    seg_ln = sitk.GetImageFromArray(ct_array_one)
    seg_ln.CopyInformation(ct)
    sitk.WriteImage(seg_ln, dest_path + '/' + str(pID) + '_ln' + '_'+str(axis)+'.nii.gz')  
    
    seg_pt = sitk.GetImageFromArray(ct_array_two)
    seg_pt.CopyInformation(ct)
    sitk.WriteImage(seg_pt, dest_path + '/' + str(pID) + '_ptum' +'_'+str(axis)+'.nii.gz')  
    
    
    
