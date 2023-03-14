# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#!/bin/bash
#python top_novel_bikes.py
echo 'Top novel bikes extracted'

for frame in 907 #728 348 960
do
echo 'Running for single frame'
	python src/creativegan/creativegan.py --name "bike" \
                       --model_path "./models/stylegan2_bike.pt" \
                       --seg_model_path './models/segmentation_bike.pt' \
                       --seg_channels 0,3 \
                       --data_path './datasets/test_data' \
                       --copy_id $frame \
                       --paste_id 7 \
                       --context_ids 7-12 \
                       --layernum 6 \
                       --use_quantized_models 1 \
                       --resunet_quantized_model_path './models/quantized_segmentation_model' \
                       --gan_infer_batch_size 10 \
                       --novelty_infer_batch_size 10 \
                       --segment_infer_batch_size 10 \
                       --novelty_score \
                       --intel 1
done

# for handle in 580 811 576
# do
# 	python src/creativegan.py --name "bike" \
#                        --model_path "./models/stylegan2_bike.pt" \
#                        --seg_model_path './models/segmentation_bike.pt' \
#                        --seg_channels 3 \
#                        --data_path './datasets/biked' \
#                        --copy_id $handle \
#                        --paste_id 7 \
#                        --context_ids 7-12 \
#                        --layernum 8 \
#                        --ssim \
#                        --novelty_score
# done
