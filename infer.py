import glob
import os, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models import VxmDense_1, VxmDense_2, VxmDense_huge
import time
import matplotlib.pyplot as plt
from feature_extract import feature_extract, feature_extract_torch
from edge import *

def save_visualization_slices(y, x_def, x_seg, def_out, count, results_dir='visualization_results'):
    # Create directory for saving results
    study_dir = os.path.join(results_dir, f'study_{count}')
    os.makedirs(study_dir, exist_ok=True)
    
    # Save original def_out before dilation
    save_visualization(y, x_def, x_seg, def_out, study_dir, count, suffix='before_dilate')
    
    # Convert to numpy array
    def_out_np = def_out.detach().cpu().numpy()
    
    # Create output array for dilated result
    def_out_dilated = np.zeros_like(def_out_np)
    
    # Apply dilation slice by slice
    kernel = np.ones((3, 3), np.uint8)
    for z in range(def_out_np.shape[2]):  # Iterate through depth
        def_out_dilated[0, 0, z] = cv2.dilate(def_out_np[0, 0, z], kernel, iterations=1)
    
    # Convert back to tensor for visualization
    def_out_dilated = torch.from_numpy(def_out_dilated).to(def_out.device)
    
    # Save visualization after dilation
    save_visualization(y, x_def, x_seg, def_out_dilated, study_dir, count, suffix='after_dilate')

def save_visualization(y, x_def, x_seg, def_out, study_dir, count, suffix):
    slices = [1/3, 1/2, 2/3]
    
    # Creating a 4x3 grid for subfigures
    fig, axs = plt.subplots(4, 3, figsize=(15, 20)) 

    for col, slice_frac in enumerate(slices):
        slice_idx = int(y.shape[2] * slice_frac)

        # Plotting slices for y (target image)
        y_slice = y[0, 0, slice_idx, :, :].detach().cpu().numpy()
        axs[0, col].imshow(y_slice, cmap='gray')
        axs[0, col].set_title(f'Target (Y) Slice {int(slice_frac*100)}%')
        
        # Plotting slices for x_def (deformed image)
        x_def_slice = x_def[0, 0, slice_idx, :, :].detach().cpu().numpy()
        axs[1, col].imshow(x_def_slice, cmap='gray')
        axs[1, col].set_title(f'Deformed (X_def) Slice {int(slice_frac*100)}%')

        # Plotting slices for x_seg (original segmentation)
        x_seg_slice = x_seg[0, 0, slice_idx, :, :].detach().cpu().numpy()
        axs[2, col].imshow(x_seg_slice, cmap='gray')
        axs[2, col].set_title(f'Original Seg Slice {int(slice_frac*100)}%')

        # Plotting slices for def_out (deformed segmentation)
        def_out_slice = def_out[0, 0, slice_idx, :, :].detach().cpu().numpy()
        axs[3, col].imshow(def_out_slice, cmap='gray')
        axs[3, col].set_title(f'Deformed Seg Slice {int(slice_frac*100)}%')

    # Turn off axis visibility
    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    # Save figure with appropriate suffix
    plt.savefig(os.path.join(study_dir, f"visual_{count}_{suffix}.png"))
    plt.close(fig)

def main(should_save_viz):
    test_dir = '../LPB40/test_breast_for_breast/'
    model_idx = -1
    img_size = (128, 256, 256)
    weights = [1, 1, 0.06]
    print("weights: ", weights)
    # weights = [1, 4, 4]
    # model_folder = 'vxm_2_cc_{}_diffusion_{}_{}_2/'.format(weights[0], weights[1], weights[2])
    model_folder = 'vxm_2_mse_{}_diffusion_{}_{}_2/'.format(weights[0], weights[1], weights[2])
    model_dir = 'experiments/' + model_folder
    dict = utils.process_label()
    if os.path.exists('experiments/'+model_folder[:-1]+'.csv'):
        os.remove('experiments/'+model_folder[:-1]+'.csv')
    csv_writter(model_folder[:-1], 'experiments/' + model_folder[:-1])
    line = ''
    model = VxmDense_2(img_size)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx], map_location='cuda:0')['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, mode='nearest', use_distance_transform=True)
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.JHUBrainInferBreastDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_ssim_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    eval_time = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_breast_seg = data[2]
            y_breast_seg = data[3]
            x_seg = data[4]
            y_seg = data[5]
            x2_edge = get_edge_ssim(x.clone(), x_breast_seg.clone())
            y2_edge = get_edge_ssim(y.clone(), y_breast_seg.clone())
            # first stage
            x_in = torch.cat((x,y),dim=1)
            # second stage
            y2, x2 = feature_extract(y.clone(), x.clone())
            start_time = time.time()
            x_def, _, flow = model(x_in, x2, y2)
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            def_edge_out = reg_model([x2_edge.cuda().float(), flow.cuda()])
            end_time = time.time()
            elapsed_time = end_time - start_time
            eval_time.update(elapsed_time, x.size(0))
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            csv_writter(line, 'experiments/' + model_folder[:-1])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

            dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            ssim_trans = utils.ssim_val(def_edge_out, y2_edge)
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}, ssim: {:.4f}'.format(dsc_trans.item(),dsc_raw.item(), ssim_trans.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            eval_ssim_def.update(ssim_trans.item(), x.size(0))
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Deformed SSIM: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}, time: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_ssim_def.avg,
                                                                                    eval_ssim_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std,
                                                                                    eval_time.avg,
                                                                                    eval_time.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    should_save_viz = False
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main(should_save_viz)