import glob
import os
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from models import VxmDense_2
import time
import json

def load_infer_config(config_path='infer_config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_model(config, device):
    img_size = tuple(config['img_size'])
    weights = config['weights']
    model_idx = config['model_idx']
    model_folder = config['model_folder_template'].format(*weights)
    model_dir = os.path.join('experiments', model_folder)
    model = VxmDense_2(img_size)
    best_model = torch.load(model_dir + sorted(os.listdir(model_dir))[model_idx], map_location=device)['state_dict']
    print('Best model: {}'.format(sorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.to(device)
    return model

def prepare_dataloader(config):
    test_dir = config['test_dir']
    test_composed = transforms.Compose([
        trans.Seg_norm(),
        trans.NumpyType((np.float32, np.int16)),
    ])
    test_set = datasets.JHUBrainInferBreastDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    return test_loader

def run_inference(model, test_loader, config, device):
    import utils
    from feature_extract import feature_extract
    img_size = tuple(config['img_size'])
    weights = config['weights']
    model_folder = config['model_folder_template'].format(*weights)
    reg_model = utils.register_model(img_size, mode='nearest', use_distance_transform=True)
    reg_model.to(device)
    eval_time = utils.AverageMeter()
    # For statistics of Dice for each label
    label_dsc_list = []
    label_count = None
    with torch.no_grad():
        for data in test_loader:
            model.eval()
            x = data[0].to(device)
            y = data[1].to(device)
            label_pairs = data[2]  # list of (x_seg, y_seg)
            if label_count is None:
                label_count = len(label_pairs)
                label_dsc_list = [utils.AverageMeter() for _ in range(label_count)]
            x_in = torch.cat((x, y), dim=1)
            y2, x2 = feature_extract(y.clone(), x.clone())
            start_time = time.time()
            x_def, _, flow = model(x_in, x2, y2)
            elapsed_time = time.time() - start_time
            eval_time.update(elapsed_time, x.size(0))
            for idx, (x_seg, y_seg) in enumerate(label_pairs):
                x_seg = x_seg.to(device)
                y_seg = y_seg.to(device)
                def_out = reg_model([x_seg.float(), flow.to(device)])
                dsc_val = utils.dice_val(def_out.long(), y_seg.long(), 46)
                label_dsc_list[idx].update(dsc_val.item(), x.size(0))
                print(f'Label {idx}: Dice: {dsc_val.item():.4f}')
    for idx in range(label_count):
        print(f'Label {idx}: Mean Dice: {label_dsc_list[idx].avg:.4f} ± {label_dsc_list[idx].std:.4f}')
    print('Avg time per case: {:.3f} ± {:.3f} sec'.format(eval_time.avg, eval_time.std))

def main():
    config = load_infer_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Number of GPU: ' + str(torch.cuda.device_count()))
    for GPU_idx in range(torch.cuda.device_count()):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(0)
    print('Currently using: ' + torch.cuda.get_device_name(0))
    print('If the GPU is available? ' + str(torch.cuda.is_available()))
    model = load_model(config, device)
    test_loader = prepare_dataloader(config)
    run_inference(model, test_loader, config, device)

if __name__ == '__main__':
    main()