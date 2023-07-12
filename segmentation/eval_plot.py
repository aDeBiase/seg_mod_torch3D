import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pathlib
import yaml


def save_plots(loss_train, loss_val, dice_val, dice_pt, dice_ln, save_dir):
    
    plt.figure(figsize=(10,10))
    
    x = [i + 1 for i in range(len(loss_train))]
    y1 = loss_train
    y2 = loss_val
    
    plt.title("Epoch Average Loss - train & validation")
    plt.xlabel("epoch")
    plt.ylabel('$Loss$')
    
    plt.plot(x, y1, label='training')
    plt.plot(x, y2, label='validation')
    
    plt.grid()
    
    plt.savefig(os.path.join(save_dir, 'Loss_function.png'), bbox_inches='tight')
    
    plt.figure(figsize=(20,10))
    
    plt.subplot(1, 2, 1)
    
    plt.title("Mean Dice Score - validation")
    x = [i + 1 for i in range(len(dice_val))]
    y = dice_val
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.grid()
    
    plt.subplot(1, 2, 2)
    
    plt.title("Mean Dice Score - pt vs ln")
    x = [i + 1 for i in range(len(dice_pt))]
    y1 = dice_pt
    y2 = dice_ln
    
    plt.xlabel("epoch")
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.grid()
    
    plt.savefig(os.path.join(save_dir, 'validation_metrics.png'), bbox_inches='tight')
    plt.close()
    
    
    
def save_plots_pt(loss_train, loss_val, dice_val, dice_pt, recall, save_dir):
    
    plt.figure(figsize=(10,10))
    
    x = [i + 1 for i in range(len(loss_train))]
    y1 = loss_train
    y2 = loss_val
    
    plt.title("Epoch Average Loss - train & validation")
    plt.xlabel("epoch")
    plt.ylabel('$Loss$')
    
    plt.plot(x, y1, label='training')
    plt.plot(x, y2, label='validation')
    
    plt.grid()
    
    plt.savefig(os.path.join(save_dir, 'Loss_function.png'), bbox_inches='tight')
    
    plt.figure(figsize=(20,10))
    
    plt.subplot(1, 2, 1)
    
    plt.title("Mean Dice Score - tot vs pt")
    x = [i + 1 for i in range(len(dice_val))]
    y = dice_val
    y1 = dice_pt
    
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.plot(x, y1)
    plt.grid()
    
    plt.subplot(1, 2, 2)
    
    plt.title("Recall Validation")
    x = [i + 1 for i in range(len(dice_pt))]
    y = recall
    
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.grid()
    
    plt.savefig(os.path.join(save_dir, 'validation_metrics.png'), bbox_inches='tight')
    plt.close()
    
    