import os
import numpy as np
import SimpleITK as sitk

def read_split_file(path):

    with open(path, 'r') as inf:
        dict_from_file = eval(inf.read())
        
    return dict_from_file
    
    
def mix_directions(experiment_folder,segmentation_folder,patient_ID):
    
    path_x=os.path.join(experiment_folder,segmentation_folder)
    path_y=os.path.join(experiment_folder,segmentation_folder) 
    path_z=os.path.join(experiment_folder,segmentation_folder) 

    vol_seg_x=sitk.ReadImage(os.path.join(path_x,'{0}.nii.gz'.format(patient_ID + '_xcorr')))
    vol_seg_y=sitk.ReadImage(os.path.join(path_y,'{0}.nii.gz'.format(patient_ID + '_ycorr')))
    vol_seg_z=sitk.ReadImage(os.path.join(path_z,'{0}.nii.gz'.format(patient_ID + '_zcorr')))
    
    SEG_X=sitk.GetArrayFromImage(vol_seg_x)
    SEG_Y=sitk.GetArrayFromImage(vol_seg_y)
    SEG_Z=sitk.GetArrayFromImage(vol_seg_z)
    
    SEG_X[SEG_X < 0.05]=0
    SEG_Y[SEG_Y < 0.05]=0
    SEG_Z[SEG_Z < 0.05]=0
    
    to_include = [SEG_X,SEG_Y,SEG_Z]
    
    f = 0
    for im in to_include:
        f = f + im
        
    f = f/3
    f[f<0.01]  = 0
    
    image_array_zero = np.zeros_like(SEG_X)
    image_array_zero[: f.shape[0], :  f.shape[1], :  f.shape[2]] = f
    
    seg_threshold = sitk.GetImageFromArray(image_array_zero)
    seg_threshold.CopyInformation(vol_seg_x)
        
    return seg_threshold
        
    

path_data = '/data/data/DATA_automseg/' 
experiment_folder = '/data/model/experiment_def' 
segmentation_folder = 'testing_images'
multi_view_segmentation_folder = 'reconstructed_volumes'
split_file = os.path.join(experiment_folder, 'patient_split.json') 

split_file_read=read_split_file(split_file)
patient_IDs=split_file_read["validate"] #validate
patient_IDs.sort()


for ID in patient_IDs:
    
    print("Multi-view fusion for:" + str(ID))
    
    if not os.path.isdir(os.path.join(experiment_folder,multi_view_segmentation_folder)):
        os.mkdir(os.path.join(experiment_folder,multi_view_segmentation_folder))
    
    path_out = os.path.join(experiment_folder,multi_view_segmentation_folder,ID+'.nii.gz')
    
    output_image = mix_directions(experiment_folder,segmentation_folder,ID)
    
    sitk.WriteImage(output_image, path_out)  
    
    
    
