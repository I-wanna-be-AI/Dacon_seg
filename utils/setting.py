import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

def init_cuda_distributed(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    torch.distributed.init_process_group( backend='nccl', init_method='env://')
    args.local_rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size() 

    args.is_master = args.local_rank == 0    
    args.device = torch.device(f'cuda:{args.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(args.local_rank) # 표기
    seed_everything(args.seed + args.local_rank)

# Set Seed
def seed_everything(seed ):              
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

def save_model(args, model):
    torch.save(model.state_dict(), os.path.join(args.chkpt_path, args.model + args.save_name + "_model_768.pt"))
    torch.save(model.module.state_dict(), os.path.join(args.chkpt_path, args.model + "_model_768(X).pt"))

def get_metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    return acc
