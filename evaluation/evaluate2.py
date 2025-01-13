
from surface_distance import metrics
from surface_distance.metrics import *
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy.spatial import cKDTree

def get_largest_component(image):
    """
    get the largest component from 2D or 3D binary image
    image: nd array
    """
    dim = len(image.shape)
    if(image.sum() == 0 ):
        print('the largest component is null')
        return image, image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    output_opp = np.asarray(labeled_array != max_label, np.uint8)
    return  output, output_opp 
    
    
def precision(mask_gt, mask_seg):
    
    A=mask_seg.reshape(mask_seg.shape[0]*mask_seg.shape[1]*mask_seg.shape[2], 1)
    B=mask_gt.reshape(mask_gt.shape[0]*mask_gt.shape[1]*mask_gt.shape[2], 1)
    
    TP=np.sum(np.logical_and(A == 1, B == 1))
    FP=np.sum(np.logical_and(A == 1, B == 0))
    
    if float(TP+FP)>0:
        prec = float(TP)/float(TP+FP)
    else:
        prec = 'nan'
    
    return prec
    
    
def recall(mask_gt, mask_seg):
    
    A=mask_seg.reshape(mask_seg.shape[0]*mask_seg.shape[1]*mask_seg.shape[2], 1)
    B=mask_gt.reshape(mask_gt.shape[0]*mask_gt.shape[1]*mask_gt.shape[2], 1)
    
    FN=np.sum(np.logical_and(A == 0, B == 1))
    TP=np.sum(np.logical_and(A == 1, B == 1))
    P = TP + FN
    
    if P>0:
      rec = float(TP)/float(P)
    else:
      rec = 'nan'
    
    return rec

    
    
def read_split_file(path):

    with open(path, 'r') as inf:
        dict_from_file = eval(inf.read())
        
    return dict_from_file
    

path_data = "/data/DATA_resampled/" 
experiment_folder = '/data/model/experiment_1'
save_files_folder = '/data/documents/' 
segmentation_folder = 'testing_images/testing_images_original'
multi_view_segmentation_folder = 'reconstructed_volumes/reconstructed_volumes_original'

images_ = os.listdir('/data/model/experiment_1/reconstructed_volumes/reconstructed_volumes_original/')
patient_IDs = [k.split('.')[0] for k in images_]
patient_IDs.sort()

original_resolution_file = "/data/documents/original_resolution_ct.csv"
resolution_df = pd.read_csv(original_resolution_file)
resolution_df = resolution_df.set_index('PatientID')

analysis = pd.DataFrame(columns=['PatientID',
            'threshold',
            'dicex', 
            'dicey', 
            'dicez', 
            'dice_multi',
            'dice after postpro',
            'surface_dice_x', 
            'surface_dice_y', 
            'surface_dice_z', 
            'surface_dice_multi', 
            'surface_dice_multi_postpro',
            'surface_overlap',
            'robust_hausdorff',
            'average_surface_distance',   
            'precision',
            'recall',   
            'precision_postpro',
            'recall_postpro',          
            ])


for ID in patient_IDs:

    res = [resolution_df.loc[ID, 'Resolution_x'],
                        resolution_df.loc[ID, 'Resolution_y'],
                        resolution_df.loc[ID, 'Resolution_z']]

    print("Evaluating patient:" + str(ID))
    
    vol_gt=sitk.ReadImage(path_data+'/'+ID+"_ct_GTVpt.nii.gz") #GTVp
    GTV=sitk.GetArrayFromImage(vol_gt)
    print('GTV shape: ' + str(GTV.shape))
    mask_gt = GTV.astype(dtype=bool)
    
    path_seg_3D=os.path.join(experiment_folder,multi_view_segmentation_folder) 
    vol_seg_3D=sitk.ReadImage(os.path.join(path_seg_3D,'{0}.nii.gz'.format(ID)))
    SEG_3D=sitk.GetArrayFromImage(vol_seg_3D)
    print('SEG shape: ' + str(SEG_3D.shape))
    
    path_=os.path.join(experiment_folder,segmentation_folder)
    
    vol_seg_x=sitk.ReadImage(os.path.join(path_,'{0}.nii.gz'.format(ID + '_xcorr')))
    vol_seg_y=sitk.ReadImage(os.path.join(path_,'{0}.nii.gz'.format(ID + '_ycorr')))
    vol_seg_z=sitk.ReadImage(os.path.join(path_,'{0}.nii.gz'.format(ID + '_zcorr')))
    
    SEG_X=sitk.GetArrayFromImage(vol_seg_x)
    SEG_Y=sitk.GetArrayFromImage(vol_seg_y)
    SEG_Z=sitk.GetArrayFromImage(vol_seg_z)

    X_volume = np.ma.masked_where(SEG_X < 0.05, SEG_X)
    Y_volume = np.ma.masked_where(SEG_Y < 0.05, SEG_Y)
    Z_volume = np.ma.masked_where(SEG_Z < 0.05, SEG_Z)
    
    for th in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        
        mask_seg = (SEG_3D>=th)
        mask_out=mask_seg*1
        mask_seg = np.ma.masked_where(mask_seg < 0.05, mask_seg)
        mask_seg = mask_seg.astype(dtype=bool)
        
        mask_out2, rest = get_largest_component(mask_out)
        keep_mask = mask_out2
        mask_out2 = np.ma.masked_where(mask_out2 < 0.05, mask_out2)
        
        X = (X_volume>=th)
        Y = (Y_volume>=th)
        Z = (Z_volume>=th)
        
        mask_out2 = mask_out2.astype(dtype=bool)
        
        info_image = compute_surface_distances(mask_gt=mask_gt, mask_pred=mask_seg, spacing_mm=res)  
        info_imagex = compute_surface_distances(mask_gt=mask_gt, mask_pred=X, spacing_mm=res)  
        info_imagey = compute_surface_distances(mask_gt=mask_gt, mask_pred=Y, spacing_mm=res)  
        info_imagez = compute_surface_distances(mask_gt=mask_gt, mask_pred=Z, spacing_mm=res)  
        info_image2 = compute_surface_distances(mask_gt=mask_gt, mask_pred=mask_out2, spacing_mm=res)  
        
        analysis=analysis.append({
            'PatientID':ID,
            'threshold': th,
            'dicex': compute_dice_coefficient(mask_gt, X), 
            'dicey': compute_dice_coefficient(mask_gt, Y), 
            'dicez': compute_dice_coefficient(mask_gt, Z), 
            'dice_multi': compute_dice_coefficient(mask_gt, mask_seg),
            'dice after postpro': compute_dice_coefficient(mask_gt, mask_out2),
            'surface_dice_x': compute_surface_dice_at_tolerance(surface_distances=info_imagex, tolerance_mm=3),
            'surface_dice_y': compute_surface_dice_at_tolerance(surface_distances=info_imagey, tolerance_mm=3), 
            'surface_dice_z': compute_surface_dice_at_tolerance(surface_distances=info_imagez, tolerance_mm=3), 
            'surface_dice_multi': compute_surface_dice_at_tolerance(surface_distances=info_image, tolerance_mm=3),
            'surface_dice_multi_postpro': compute_surface_dice_at_tolerance(surface_distances=info_image2, tolerance_mm=3),  
            'surface_overlap': compute_surface_overlap_at_tolerance(surface_distances=info_image, tolerance_mm=1),
            'robust_hausdorff': compute_robust_hausdorff(surface_distances=info_image, percent=95),
            'average_surface_distance': compute_average_surface_distance(surface_distances=info_image),    
            'precision': precision(GTV, mask_out),
            'recall': recall(GTV, mask_out),      
            'precision_postpro': precision(GTV, keep_mask),
            'recall_postpro': recall(GTV, keep_mask),        
        },ignore_index=True)
        
        print('Values at threshold: ' + str(th))
        print('dice = ' + str(compute_dice_coefficient(mask_gt, mask_seg)))
        print('surface dice = ' + str(compute_surface_dice_at_tolerance(surface_distances=info_image, tolerance_mm=1)))
        print('')
    
    print('---')
    
    analysis.to_excel(os.path.join(save_files_folder, experiment_folder.split('/')[-1]+'_results_test_OR.xlsx'))

