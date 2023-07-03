# Train
torchrun --nproc_per_node=2 main.py --gpu_ids 6,7 --train 1 --img_size 224 --model efficientnet_b1 --batchsize 128 --save_name eff_b1_img224
torchrun --nproc_per_node=2 main.py --gpu_ids 8,9 --train 1 --img_size 768 --model efficientnet_b0 --batchsize 200 --save_name eff_b0_img768
torchrun --nproc_per_node=2 --master_port=12347 main.py --gpu_ids 8,9 --train 1 --img_size 224 --model convenxt --batchsize 200 --save_name convnext224

# Inference
torchrun --nproc_per_node=2 main.py --gpu_ids 1,2 --infer 1 --model efficientnet_b0 --batchsize 1024
torchrun --nproc_per_node=2 main.py --gpu_ids 6,7 --infer 1 --img_size 224 --model efficientnet_b1 --batchsize 128
torchrun --nproc_per_node=2 --master_port=12347 main.py --gpu_ids 8,9 --infer 1 --img_size 224 --model convenxt --batchsize 200 --save_name convnext224
# torchrun --nproc_per_node=2 --master_port=12346 main.py --gpu_ids 1,2 --infer 1 --model efficientnet_b0 --batchsize 1024