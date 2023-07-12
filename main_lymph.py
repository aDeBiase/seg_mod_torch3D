import torch
import torch.backends.cudnn as cudnn
from segmentation.train_lynph import seq_by_seq
import argparse
import os
import sys
import pathlib
import yaml
import wandb

sys.dont_write_bytecode = True
cudnn.benchmark = True

def main(args):

    path_to_config = pathlib.Path(args.path)
    with open(path_to_config) as f:
        config = yaml.safe_load(f)
        
    config['data_folder']=pathlib.Path(args.path_data)
    
    experiment_folder = os.path.join(pathlib.Path(config['experiment_dir']), pathlib.Path(config['output_dir']))

    #checkpoint_dir_second = os.path.join(pathlib.Path(config['output_dir']), pathlib.Path(config['checkpoint_dir_second']))
    checkpoint_dir = os.path.join(pathlib.Path(experiment_folder), pathlib.Path(config['checkpoint_dir']))
    sample_dir = os.path.join(pathlib.Path(experiment_folder), pathlib.Path(config['sample_dir']))
    test_dir = os.path.join(pathlib.Path(experiment_folder), pathlib.Path(config['test_dir']))
    
    os.makedirs(experiment_folder, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)


    if config['model_type'] == 'UNET2D':
        model = UNET2D(config)   
    elif config['model_type'] == 'BILSTM_SC_UNET':
        model = seq_by_seq(config)
    else:
        assert False, 'Unknown model'

    if config['phase'] == 'train':
        #with wandb.init(project="Lymphnode-segmentation", config=config):
            #config = wandb.config
            #print(config)
        model.train(config)
    elif config['phase'] == 'test':
        model.test(config)
    elif config['phase'] == 'test all':
        model.test_all(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    parser.add_argument("-d", "--path_data", type=str, required=True, help="path to the data folder")
    args = parser.parse_args()
    
    from monai.config import print_config
    print_config()
    
    main(args)
