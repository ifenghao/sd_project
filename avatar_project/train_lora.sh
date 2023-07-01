source ~/anaconda3/bin/activate
conda activate sd

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export HF_HOME="~/huggingface"

source ./config.sh

date_str=`date +'%Y%m%d%H%M%S'`

accelerate launch --num_cpu_threads_per_process=4 train_network_online.py \
    --pretrained_model_name_or_path=${checkpoint_file} \
    --train_data_dir=${train_image_path} \
    --output_dir=${train_output_model_path} \
    --output_name=${lora_name}_${date_str} \
    --logging_dir=${train_log_path} \
    --sample_prompts=${train_sample_path}/prompt.txt \
    --save_model_as=safetensors \
    --network_module=networks.lora \
    --resolution=512,512 \
    --network_dim=32 \
    --network_alpha=32 \
    --network_args "conv_dim=32" \
    --text_encoder_lr=1e-5 \
    --network_train_unet_only \
    --unet_lr=1e-4 \
    --lr_warmup_steps="200" \
    --learning_rate="1e-5" \
    --lr_scheduler="constant_with_warmup" \
    --train_batch_size="1" \
    --max_train_steps="1000" \
    --max_train_epochs="10" \
    --save_every_n_epochs="5" \
    --mixed_precision="bf16" \
    --save_precision="bf16" \
    --cache_latents \
    --optimizer_type="AdamW8bit" \
    --max_data_loader_n_workers="16" \
    --bucket_reso_steps=64 \
    --bucket_no_upscale  \
    --sample_sampler=euler_a \
    --sample_every_n_steps="100" \
    --sample_every_n_epochs=1 \
    --max_train_epochs=2 \
    --xformers
    