#!/bin/bash
source ~/anaconda3/bin/activate
conda activate sd

# source ./config.sh

export HF_HOME="~/huggingface"

python ../sd-scripts/tools/detect_face_rotate.py \
	--src_dir ./raw_images \
	--dst_dir ./test \
	--rotate \
	--resize_fit \
