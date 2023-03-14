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
                       --gan_infer_batch_size 10 \
                       --novelty_infer_batch_size 10 \
                       --segment_infer_batch_size 10 \
                       --novelty_score \
                       --intel 0
done

