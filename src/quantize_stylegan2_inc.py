# pylint: disable=C0305,C0114,E0401,C0115,W0404,C0415,W0611,C0411,C0301,C0303,W0621,E1101,W0621,C0116,C0103,W0613
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#################### code changes ####################
import os
import argparse
import time
import numpy
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from neural_compressor.experimental import Quantization, common
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
from creativegan.utils.stylegan2.models import SeqStyleGAN2
from creativegan.utils import zdataset
import warnings
warnings.filterwarnings("ignore")
######################################################


def load_fp32_model(model_path, truncation=0.5, size=256, **kwargs):
    """ fp32"""
    state_dict = torch.load(model_path)
    g = SeqStyleGAN2(size, style_dim=512, n_mlp=8, truncation=truncation, **kwargs)
    g.load_state_dict(state_dict['g_ema'],
                      latent_avg=state_dict['latent_avg'])
    g.cpu()
    g.eval()
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='./models/stylegan2_bike.pt',
                        required=False, help='Path to pretrained FP32 segmentation model')
    parser.add_argument('--out_path', type=str, required=False, default='./models/quantized_StyleGan2',
                        help='Directory path to save the quantized INT8 model')

    args = parser.parse_args()

    model_path = args.model_path
    out_path = args.out_path

    # creating file path if not exists
    os.makedirs(out_path, exist_ok=True)

    model = load_fp32_model(model_path=model_path, truncation=0.5, size=512)

    sample_z = torch.randn(10, 512, device='cpu')

    ######################ipex quant #################

    qconfig = ipex.quantization.default_static_qconfig
    prepared_model = prepare(model, qconfig, example_inputs=sample_z, inplace=False)

    converted_model = convert(prepared_model)
    # with torch.no_grad():
    #     traced_model = torch.jit.trace(converted_model, sample_z)
    #     traced_model = torch.jit.freeze(traced_model)
    #
    # torch.save(traced_model.state_dict() , out_path + "/best_model.pt")
    torch.save(converted_model.state_dict(), out_path + "/best_model.pt")
    # torch.save(model.state_dict(), out_path + "/model.pt")
    print("Successfully Quantized StyleGan2 Model using IPEX Quantization.")

if __name__ == '__main__':
    main()