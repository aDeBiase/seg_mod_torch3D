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
    

path_data = "/data/DATA_automseg_lymph/" 
experiment_folder = '/data/model/lymph_optimization/trial2/'
save_files_folder = '/data/model/lymph_optimization/documents/' 
segmentation_folder = 'test_sep'

images_ = os.listdir(os.path.join(experiment_folder,segmentation_folder))
patient_IDs = list(set([k.split('_')[0] for k in images_]))
patient_IDs.sort()

original_resolution_file = "/data/documents/original_resolution_ct.csv"
resolution_df = pd.read_csv(original_resolution_file)
resolution_df = resolution_df.set_index('PatientID')

analysis = pd.DataFrame(columns=['PatientID',
            'threshold',
            'dice_pt', 
            'dice_ln', 
            'surface_dice_pt', 
            'surface_dice_ln', 
            'robust_hausdorff_pt',
            'robust_hausdorff_ln',
            'precision_pt',
            'precision_ln',
            'recall_pt',   
            'recall_ln',   
            ])


for ID in patient_IDs:

    #res = [resolution_df.loc[ID, 'Resolution_x'],
                        #resolution_df.loc[ID, 'Resolution_y'],
                        #resolution_df.loc[ID, 'Resolution_z']]

    print("Evaluating patient:" + str(ID))
    
    vol_gt=sitk.ReadImage(path_data+'/'+ID+"_ct_GTV.nii.gz") 
    GTV=sitk.GetArrayFromImage(vol_gt)
    print('GTV shape: ' + str(GTV.shape))
    
    GTV_pt = (GTV==2).astype(dtype=bool)
    
    GTV_ln = (GTV==1).astype(dtype=bool)
    
    path_=os.path.join(experiment_folder,segmentation_folder)
    
    vol_seg_pt=sitk.ReadImage(os.path.join(path_,'{0}.nii.gz'.format(ID + '_ptum_x')))
    vol_seg_ln=sitk.ReadImage(os.path.join(path_,'{0}.nii.gz'.format(ID + '_ln_x')))
    
    SEG_pt=sitk.GetArrayFromImage(vol_seg_pt)
    SEG_ln=sitk.GetArrayFromImage(vol_seg_ln)

    PT_volume = np.ma.masked_where(SEG_pt < 0.05, SEG_pt)
    LN_volume = np.ma.masked_where(SEG_ln < 0.05, SEG_ln)
    
    for th in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        
        PT = (PT_volume>=th)
        PT_n = np.ma.masked_where(PT < 0.05, PT)
        PT_n = PT_n.astype(dtype=bool)
        
        LN = (LN_volume>=th)
        LN_n = np.ma.masked_where(LN < 0.05, LN)
        LN_n = LN_n.astype(dtype=bool)
        
        info_image_pt = compute_surface_distances(mask_gt=GTV_pt, mask_pred=PT_n,spacing_mm=[1,1,1])  #, spacing_mm=res
        info_image_ln = compute_surface_distances(mask_gt=GTV_ln, mask_pred=LN_n,spacing_mm=[1,1,1])  #, spacing_mm=res
        
        analysis=analysis.append({
            'PatientID':ID,
            'threshold': th,
            'dice_pt': compute_dice_coefficient(GTV_pt, PT_n), 
            'dice_ln': compute_dice_coefficient(GTV_ln, LN_n), 
            'surface_dice_pt': compute_surface_dice_at_tolerance(surface_distances=info_image_pt, tolerance_mm=3), 
            'surface_dice_ln': compute_surface_dice_at_tolerance(surface_distances=info_image_ln, tolerance_mm=3), 
            'robust_hausdorff_pt': compute_robust_hausdorff(surface_distances=info_image_pt, percent=95),
            'robust_hausdorff_ln': compute_robust_hausdorff(surface_distances=info_image_ln, percent=95),
            'precision_pt': precision(GTV_pt, PT_n),
            'precision_ln': precision(GTV_ln, LN_n),
            'recall_pt' : recall(GTV_pt, PT_n),   
            'recall_ln' : recall(GTV_ln, LN_n),   
        },ignore_index=True)
        
        
        print('Values at threshold: ' + str(th))
        print('dice for pt = ' + str(compute_dice_coefficient(GTV_pt, PT_n)))
        print('dice for ln = ' + str(compute_dice_coefficient(GTV_ln, LN_n)))
        print('surface dice pt = ' + str(compute_surface_dice_at_tolerance(surface_distances=info_image_pt, tolerance_mm=3)))
        print('surface dice ln = ' + str(compute_surface_dice_at_tolerance(surface_distances=info_image_ln, tolerance_mm=3)))
        print('')
    
    print('---')
    
    analysis.to_excel(os.path.join(save_files_folder, experiment_folder.split('/')[-1]+'_results.xlsx'))

