# pylint: disable=C0305,C0114,E0401,W0404,C0115,C0411,C0412,W0611,C0303,C0301,E1101
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
import os
import torch
#################### code changes ####################
import intel_extension_for_pytorch as ipex
from torchvision.models import wide_resnet50_2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torchvision import transforms as T
from neural_compressor.experimental import Quantization, common 
from creativegan.utils import unet
import numpy as np
######################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained FP32 segmentation model')
parser.add_argument('--out_path', type=str, required=True, help='Directory path to save the quantized INT8 model')

args = parser.parse_args()

model_path = args.model_path
out_path = args.out_path

class NormalDataset(Dataset):
    def __init__(self, data, resize=224):
        self.resize = resize
        self.x = data
    def __getitem__(self, idx):
        x = self.x[idx]
        return x
    def __len__(self):
        return len(self.x)
seg_class = 7
seg_model = unet.ResNetUNet(seg_class).cpu()

seg_model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
seg_model.eval()

#Not passing Ipex optimized fo quatization as it is crashing while loading
#seg_model = ipex.optimize(seg_model)
#print("Intel Optimization applied in segmentation model")

print('unet loaded')
data = torch.rand(300,3,224,224)
#data = np.random.rand(300,3,224,224)

gen_dataset = NormalDataset(data)
calib_dataloader = torch.utils.data.DataLoader(gen_dataset, batch_size=1)

quantizer = Quantization('src/resunet_inc_conf.yaml')
quantizer.model = common.Model(seg_model)
quantizer.calib_dataloader = common.DataLoader(gen_dataset)
q_model = quantizer.fit()
q_model.save(out_path)


