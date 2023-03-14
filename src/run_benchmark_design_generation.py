# pylint: disable=C0304,C0114,E0401,C0410,C0413,C0411,W0611,C0301,C0303,C0209,C0103
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import os
import argparse
import torch
from utils import zdataset, renormalize
from rewrite import ganrewrite, rewriteapp
import utils.stylegan2, utils.proggan
from utils.stylegan2.models import SeqStyleGAN2


def load_seq_stylegan(category, path=False, size=256, truncation=1.0,
                      intel=False, quant_model=False, **kwargs):  # mconv='seq'):
    """ loads nn sequential version of stylegan2 and puts on gpu"""
    if path:
        state_dict = torch.load(category)
    else:
        size = sizes[category]
        state_dict = load_state_dict(category)
    # size = sizes[category]
    g = SeqStyleGAN2(size, style_dim=512, n_mlp=8, truncation=truncation, **kwargs)

    if quant_model:
        prefix = "_fqn_to_auto_quant_state_map."
        f_clip = len(prefix)
        suffix = "._extra_state"
        b_clip = len(suffix)
        adapted_dict = {}
        for k, v in state_dict.items():
            k = k.replace(':', '.')
            if k.startswith(prefix):
                k = k[f_clip:]
            if k.endswith(suffix):
                k = k[:-b_clip]
            if k != ' ':
                adapted_dict[k] = v
        g.load_state_dict(adapted_dict)
    else:
        g.load_state_dict(state_dict['g_ema'],
        latent_avg=state_dict['latent_avg'])

    g.cpu().eval()

    if intel:
        import intel_extension_for_pytorch as ipex
        g = ipex.optimize(g)
        print("Intel Optimization applied in StyleGAN2 Model")
    return g



parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model_path', type=str, required=False,
                    default='./models/stylegan2_bike.pt',
                    help='Path to Stylegan model')
parser.add_argument('--model_size', type=int, default=512, help='GAN model output size')
parser.add_argument('--truncation', type=float, default=0.5,
                    help="Value for truncation trick in Stylegan")
parser.add_argument('--intel', type=int, default=0, help='To apply Intel IPEX optimization')
parser.add_argument('--gan_infer_batch_size', type=int, default=1,
                    help="Batch size to generate design samples using StyleGAN2 model")
parser.add_argument('--num_samples_gan', type=int, default=10,
                    help="Num samples to generate from StyleGAN2")
parser.add_argument('--lr', type=float, default=0.05, help='learning rate in rewriting')

parser.add_argument('--use_quantized_models', type=int,
                    default=0, help="Use IPEX Quantized model in pipeline")
parser.add_argument('--Stylegan_quantized_model_path', type=str,
                    help="Stylegan quantized model path")

args = parser.parse_args()

model_path = args.model_path
model_size = args.model_size
truncation = args.truncation
use_quantized_models = args.use_quantized_models
Stylegan_quantized_model_path = args.Stylegan_quantized_model_path
intel = args.intel
lr = args.lr
print(intel)
name='bike'
layer=6
rank=30

# Choices: ganname = 'stylegan' or ganname = 'proggan'
ganname = 'stylegan'
modelname = name
layernum = layer

# Number of images to sample when gathering statistics.
size = 10000

# Make a directory for caching some data.
layerscheme = 'default'
expdir = 'results/pgw/%s/%s/%s/layer%d' % (ganname, modelname, layerscheme, layernum)
os.makedirs(expdir, exist_ok=True)

# Load (and download) a pretrained GAN
model, Rewriter = None, None
if ganname == 'stylegan':
    model = load_seq_stylegan(model_path, path=True, size=model_size, mconv='seq',
                              truncation=truncation, intel=False, quant_model=False)
    print('StyleGAN2 model loaded')
    Rewriter = ganrewrite.SeqStyleGanRewriter
elif ganname == 'proggan':
    model = utils.proggan.load_pretrained(modelname)
    Rewriter = ganrewrite.ProgressiveGanRewriter
    
# Create a Rewriter object - this implements our method.#import pdb;pdb.set_trace()
zds = zdataset.z_dataset_for_model(model, size=size)
# zds = torch.randn(10, 512, device='cpu')

gw = Rewriter(
    model, zds, layernum, cachedir=expdir,
    low_rank_insert=True, low_rank_gradient=False,
    use_linear_insert=False,  # If you set this to True, increase learning rate.e
    key_method='zca', intel=intel, lr = lr)
print('GAN Rewriter object created')

# Display a user interface to allow model rewriting.
savedir = f'masks/{ganname}/{modelname}'
interface = rewriteapp.GanRewriteApp(gw, size=256, mask_dir=savedir, num_canvases=32)
print('GAN Rewriter App object created')

import matplotlib.pyplot as plt

step = args.num_samples_gan
total_img_gen_time = 0
print('Generating design samples from StyleGAN2', args.num_samples_gan)

count = 0
for i in range(0, args.num_samples_gan, step):
    images, gen_time = gw.render_image_batch(list(range(i,i+step)), batch_size=args.gan_infer_batch_size)
    total_img_gen_time += gen_time
    count += 1
total_img_gen_time = total_img_gen_time if count == 1 else total_img_gen_time / count
print('Total time taken in image generation : ',total_img_gen_time)
print("Number of itereations..", count)