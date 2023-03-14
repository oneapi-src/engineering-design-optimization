# pylint: disable=C0305,C0114,E0401,W0404,C0115,C0411,C0412,W0611,C0303,C0301,C0103,W0621,W0613,E1101,C0103,R0913
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
######################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--out_path', type=str, required=True, help='Directory path to save the quantized INT8 model')

args = parser.parse_args()
out_path = args.out_path

class NormalDataset(Dataset):
    def __init__(self, data, resize=224, cropsize=224, grayscale=True, normalize=True, n=None):
        self.x = data

    def __getitem__(self, idx):
        x = self.x[idx]
        return x

    def __len__(self):
        return len(self.x)

model = wide_resnet50_2(pretrained=True, progress=True)
model.to('cpu')
model.eval()
#model = ipex.optimize(model, conv_bn_folding=False, linear_bn_folding=False)
data = torch.rand(10,3,224,224)
gen_dataset = NormalDataset(data, n=10, grayscale=False, normalize=False, resize=224, cropsize=224)
calib_dataloader = torch.utils.data.DataLoader(gen_dataset, batch_size=1)

quantizer = Quantization('src/wideresnet_inc_conf.yaml')
quantizer.model = common.Model(model)
quantizer.calib_dataloader = common.DataLoader(gen_dataset)
q_model = quantizer.fit()
q_model.save(out_path)

