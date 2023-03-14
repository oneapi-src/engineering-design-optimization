# pylint: disable =C0305,C0114,E0401,C0413,C0411,W0611,W0621,C0303,E1101,C0103,C0301
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image
from utils import anomaly, unet


class NormalDataset(Dataset):
    def __init__(self, path, resize=224, cropsize=224, grayscale=True, normalize=True, n=None):
        # assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.path = path
        self.resize = resize
        self.cropsize = cropsize

        # load dataset
        self.x = self.load_dataset_folder(n)

        # set transforms
        transform_x = [T.Resize(resize, Image.LANCZOS), # Image.ANTIALIAS
                       T.CenterCrop(cropsize)]

        if grayscale:
            transform_x.append(T.Grayscale(num_output_channels=3))

        transform_x.append(T.ToTensor())

        if normalize:
            transform_x.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225]))
        self.transform_x = T.Compose(transform_x)

    def __getitem__(self, idx):
        x = self.x[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)
        return x

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self, n):
        img_dir = self.path
        x = []
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                 for f in os.listdir(img_dir)
                                 if (f.endswith('.png') or f.endswith('.jpg'))])
        x.extend(img_fpath_list)

        return list(x[:n]) if n != None else list(x)


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data_path', type=str, required=True,
                    default='./datasets/biked',
                    help='Path to dataset, the folder should directly contain the images')
parser.add_argument('-i', '--intel', type=int,
                    default=0, help='To apply Intel IPEX optimization')
parser.add_argument('-b', '--batch_size', type=int,
                    default=1, help='Batch size to be used by the model for inference')
parser.add_argument('-uqm', '--use_quantized_models',
                    action='store_true', help="Use INC Quantized model in pipeline")
parser.add_argument('-m', '--seg_model_path', type=str, required=True,
                    default='./models/segmentation_model.pt',
                    help="Path to segmentation model")
parser.add_argument('-qm', '--quant_seg_model_path', type=str,
                    default='./models/quantized_segmentation_model/best_model.pt',
                    required=False, help="Path to quantized segmentation model")
parser.add_argument('-cls', '--seg_total_class', type=int,
                    default=7, help="Total class/channel in segmentation model")

args = parser.parse_args()

intel = args.intel
seg_class = args.seg_total_class
seg_model_path = args.seg_model_path
batch_size = args.batch_size
quant_model_path = args.quant_seg_model_path

name='bike'

data_path = args.data_path
train_set_size = 1943
use_quantized_models = args.use_quantized_models

train_dataset = NormalDataset(data_path, grayscale=False, normalize=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True) # batch_size = batch_size
image_files = train_dataset.x

step = train_set_size

images = []
for x in range(0, train_set_size):

    images.append(Image.open(image_files[x]).resize((256,256)))

device = "cpu"
seg_model = unet.ResNetUNet(seg_class).cpu()

seg_model.load_state_dict(torch.load(seg_model_path,map_location=torch.device('cpu')))

if use_quantized_models:
    from neural_compressor.utils.pytorch import load
    seg_model = load(quant_model_path,seg_model)
    print("Quantized Int8 Segmentation model loaded")

seg_model.eval()

if intel:
    import intel_extension_for_pytorch as ipex
    #self.model = self.model.to(memory_format=torch.channels_last)
    seg_model = ipex.optimize(seg_model)
    print("Intel Optimization segmentation model loaded")

print('unet loaded')

tot_time = 0
wrm_up = None

for x in train_dataloader:
    if not wrm_up:
        print("warm up run started...")
        for i in range(10):
            _ = seg_model(x.cpu()).sigmoid().detach().cpu()
        print("completed warm up. Proceeding to benchmarking...")
        wrm_up = 'Done'

    start = time.time()
    _ = seg_model(x)
    seg_time = time.time() - start

    tot_time += seg_time

print("Total Segmentation time: ", tot_time/len(train_dataloader))