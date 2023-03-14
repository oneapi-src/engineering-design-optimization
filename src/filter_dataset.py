# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
import os
import shutil
import glob

DATASET_DIR = "."
SUBSET_COUNT = 1956
os.makedirs(os.path.join(DATASET_DIR,"biked_subset"),exist_ok=True)
files = glob.glob(DATASET_DIR+"/biked/*.jpg")
print(len(files))
sub_files = files[0:SUBSET_COUNT]
print(len(sub_files))
for file in sub_files:
    file = file.split("/")[-1]
    #print(file)
    shutil.copyfile(os.path.join(DATASET_DIR,"biked",file),os.path.join(DATASET_DIR,"biked_subset",file))
print("Subset created")
