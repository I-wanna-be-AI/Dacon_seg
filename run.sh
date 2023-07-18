# Train
torchrun --nproc_per_node=2 main.py --wandb_id dk58319 --gpu_ids 8,9 --train 1 --img_size 224 --model DeepLabV3_resnet34 --batchsize 128 --save_name 1 --epochs 30


# Inference
torchrun --nproc_per_node=1 main.py  --gpu_ids 8 --infer 1 --model DeepLabV3_resnet34 --batchsize 128

