# Train
torchrun --nproc_per_node=1 main.py --wandb_id dk58319 --gpu_ids 9 --train 1 --img_size 224 --model DeepLabV3_resnet34 --batchsize 128 --save_name _new_loss --epochs 30


torchrun --nproc_per_node=2 main.py --gpu_ids 8,9 --train 1 --img_size 224 --model unetplus_resnext101 --batchsize 64 --epochs 40 --lr 0.001
torchrun --nproc_per_node=2 main.py --gpu_ids 8,9 --train 1 --img_size 256 --model effnet3 --batchsize 64 --epochs 40 --lr 0.01 --one_cycle_max_lr 0.01
torchrun --nproc_per_node=2 main.py --gpu_ids 8,9 --train 1 --img_size 256 --model unetplus_inception --batchsize 64 --epochs 40 --lr 0.01 --one_cycle_max_lr 0.01
torchrun --nproc_per_node=2 main.py --gpu_ids 8,9 --train 1 --img_size 200 --model unetplus_inception --batchsize 64 --epochs 40 --lr 0.01 --one_cycle_max_lr 0.01
torchrun --nproc_per_node=2 main.py --gpu_ids 8,9 --train 1 --img_size 256 --model unetplus_xception --batchsize 64 --epochs 40 --lr 0.01 --one_cycle_max_lr 0.01
# Inference
torchrun --nproc_per_node=1 main.py  --gpu_ids 8 --infer 1 --model unetplus_inception --batchsize 128 --img_size 256
torchrun --nproc_per_node=1 main.py  --gpu_ids 8 --infer 1 --model DeepLabV3_resnet34 --batchsize 128 --img_size 416

