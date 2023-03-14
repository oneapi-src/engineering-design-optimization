# pylint: disable =C0305,C0114,E0401,C0413,C0411,W0611,W0621,C0303,E1101,C0103,C0301
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils import anomaly


parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, required=True, help='Path to dataset, the folder should directly contain the images')
parser.add_argument('--k', type=int, default=50, help='topk value for anomaly detection')
parser.add_argument('--anomaly_threshold', type=int, default=3.5, help='Threshold for novelty segmentation')
parser.add_argument('--intel', type=int, default=0, help='To apply Intel IPEX optimization')
parser.add_argument('--novelty_infer_batch_size', type=int, default=32, help="Batch size to extract the features from design images using WideResNet model")
parser.add_argument('--use_quantized_models', type=int, default=0, help="Use INC Quantized model in pipeline")
parser.add_argument('--wideresnet_quantized_model_path', type=str, help="Wideresnet quantized model path")
parser.add_argument('--train_set_size', type=int, default=500, help='Training set size ')


args = parser.parse_args()

intel = args.intel
print(intel)
name='bike'

#data_path = './datasets/biked'
data_path = args.data_path
k=args.k
train_set_size = args.train_set_size

anomaly_threshold = args.anomaly_threshold
use_quantized_models = args.use_quantized_models
wideresnet_quantized_model_path = args.wideresnet_quantized_model_path

# Create detector instance given a directory of the normal images
ad = anomaly.AnomalyDetector(data_path, name=name, topk=k , batch_size = args.novelty_infer_batch_size, intel=intel, 
                            use_quantized_models=use_quantized_models,
                             quantized_model_path=wideresnet_quantized_model_path, train_set_size=train_set_size)
print('AnomalyDetector object created')

# Extract and cache embeddings of the normal images
#print('Start Loading Training Features')
#ad.load_train_features()
#print('Loaded Training Features')


train_dataset = anomaly.NormalDataset(data_path, grayscale=False, normalize=False)
image_files = train_dataset.x

anomaly_scores = []
step = train_set_size

images = []
for x in range(0, train_set_size):
    images.append(Image.open(image_files[x]).resize((256,256)))
print('Predicting Novelty Scores in Training Images')

scores,total_pred_time = ad.predict_anomaly_scores(images, topk=k)
anomaly_scores.append(scores)

anomaly_scores = np.concatenate(anomaly_scores)
top_idx = anomaly_scores.argsort()[::-1]
print("Total Feature Extraction time: ", total_pred_time)

