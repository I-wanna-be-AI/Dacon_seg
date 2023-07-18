# Train
torchrun --nproc_per_node=2 main.py --wandb_id dk58319 --gpu_ids 8,9 --train 1 --img_size 256 --model unet_base --batchsize 128 --save_name 22 --one_cycle_max_lr 1e-2 --epochs 10


# Inference
torchrun --nproc_per_node=2 main.py --wandb_id dk58319 --gpu_ids 8,9 --infer 1 --model unet_mlt_b4 --batchsize 128

