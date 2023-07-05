# Train
torchrun --nproc_per_node=2 main.py --gpu_ids 8,9 --train 1 --img_size 224 --model unet_base --batchsize 1024 --save_name unet_base


# Inference
torchrun --nproc_per_node=2 main.py --gpu_ids 8,9 --infer 1 --model unet_base --batchsize 1024

