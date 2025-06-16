import glob
import os, losses
import utils
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models import VxmDense_1, VxmDense_2, VxmDense_huge
import wandb
from datetime import datetime
from feature_extract import feature_extract
from skimage.filters import frangi
import argparse

def save_visualization_slices(x, y, x_def, count, results_dir='results'): 
    os.makedirs(results_dir, exist_ok=True)
    slices = [1/3, 1/2, 2/3]

    # Creating a 3x3 grid for subfigures, with each row representing one of x, y, x_def
    fig, axs = plt.subplots(3, 3, figsize=(15, 15)) 

    for col, slice_frac in enumerate(slices):
        slice_idx = int(x.shape[2] * slice_frac)  # Adjusting slice index based on the fraction

        # Plotting slices for x
        x_slice = x[0, 0, slice_idx, :, :].detach().cpu().numpy()
        axs[0, col].imshow(x_slice, cmap='gray')
        axs[0, col].set_title(f'X Slice {int(slice_frac*100)}%')
        
        # Plotting slices for y
        y_slice = y[0, 0, slice_idx, :, :].detach().cpu().numpy()
        axs[1, col].imshow(y_slice, cmap='gray')
        axs[1, col].set_title(f'Y Slice {int(slice_frac*100)}%')

        # Plotting slices for x_def
        x_def_slice = x_def[0, 0, slice_idx, :, :].detach().cpu().numpy()
        axs[2, col].imshow(x_def_slice, cmap='gray')
        axs[2, col].set_title(f'X_def Slice {int(slice_frac*100)}%')

    # Turn off axis visibility for all subplots
    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    # Save the figure with a unique name based on 'count'
    fig.savefig(os.path.join(results_dir, f"visual_{count}.png"))
    plt.close(fig)  # Close the figure to free memory

def init_wandb(weights):
    date_time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    wandb_run = wandb.init(
        project="breast_registration_voxel_morph_inference2",
        name=date_time_str + f"two_stage_affine_mse_unsup_{weights[0]}_{weights[1]}_{weights[2]}",
        # name=date_time_str + "two_stage_affine_cc_unsup",
        config={
            "architecture": "UNet_Cbam_STN",
        }
    )
    return wandb_run

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main(GPU_iden, weights):
    batch_size = 1
    train_dir = '../LPB40/train/'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_frequency = 2
    wandb_run = init_wandb(weights)
    save_dir = 'vxm_2_mse_{}_diffusion_{}_{}_2/'.format(weights[0], weights[1], weights[2])
    experiment_dir = os.path.join(base_dir, 'experiments', save_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    lr = 0.0005
    epoch_start = 0
    max_epoch = 15000
    img_size = (128, 256, 256)
    cont_training = False

    '''
    Initialize model
    '''
    model = VxmDense_2(img_size)
    model.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 0
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-3])['state_dict']
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])
    train_set = datasets.JHUBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    print("train_set length is ", len(train_set))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = nn.MSELoss()
    # criterion = losses.NCC()
    criterions = [criterion, criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    count = 0
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            count = adjust_learning_rate(optimizer, epoch, max_epoch, lr, count)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            # second stage
            # x2, y2 = feature_extract(x.clone(), y.clone())
            x2, y2 = feature_extract(x.clone(), y.clone())

            # first stage
            x_in = torch.cat((x,y),dim=1)
            output = model(x_in, x2, y2)
            loss = 0
            loss_vals = []
            if idx % 1000 == 0:
                save_visualization_slices(x, y, output[0],idx)
                save_visualization_slices(x2, y2, output[1], idx, results_dir='results_edge')
            for n, loss_function in enumerate(criterions):
                if n == 0 or n == 2:
                    curr_loss = loss_function(output[n], y) * weights[n]
                else:
                    curr_loss = loss_function(output[n], y2) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_in
            del output
            # flip fixed and moving images
            loss = 0
            x_in = torch.cat((y, x), dim=1)
            output = model(x_in, y2, x2)
            for n, loss_function in enumerate(criterions):
                if n == 0 or n == 2:
                    curr_loss = loss_function(output[n], x) * weights[n]
                else:
                    curr_loss = loss_function(output[n], x2) * weights[n]
                loss_vals[n] += curr_loss
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Edge Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item()/2, loss_vals[1].item()/2, loss_vals[2].item()/2))
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''

        loss_all.reset()

        # Save checkpoint every `save_frequency` epochs
        if (epoch + 1) % save_frequency == 0:
            save_checkpoint({
                'epoch': epoch + 1, # Save the epoch number that just finished
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_dir=experiment_dir, filename='final.pth.tar') # Overwrite the same file
            print(f"Saved checkpoint at epoch {epoch} to {os.path.join(experiment_dir, 'final.pth.tar')}")
    wandb_run.finish()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, count, power=0.9):
    if count >= 5:
        for param_group in optimizer.param_groups:
            param_group['lr'] = round(INIT_LR * 0.9, 8)
            count = 0
            print("update learning rate to ", param_group['lr'])
    return count

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(128, 256, 256)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    parser = argparse.ArgumentParser(description='Train TransMorph with custom GPU and loss weights.')
    parser.add_argument('--GPU_iden', type=int, default=0, help='GPU index to use')
    parser.add_argument('--weights', type=float, nargs=3, default=[1.0, 1.0, 0.08],
                        help='Three loss weights (e.g., --weights 1 1 0.02)')
    args = parser.parse_args()
    print(args)
    GPU_iden = args.GPU_iden
    weights = args.weights
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main(GPU_iden, weights)