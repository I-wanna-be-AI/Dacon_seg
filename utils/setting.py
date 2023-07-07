import os
import random
from typing import List

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score


from dataset import rle_decode


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

def dice_score(prediction: np.array, ground_truth: np.array, smooth=1e-7) -> float:
    '''
    Calculate Dice Score between two binary masks.
    '''
    intersection = np.sum(prediction * ground_truth)
    return (2.0 * intersection + smooth) / (np.sum(prediction) + np.sum(ground_truth) + smooth)


def calculate_dice_scores(ground_truth_df, prediction_df, img_shape=(224, 224)) -> List[float]:
    '''
    Calculate Dice scores for a dataset.
    '''


    # Keep only the rows in the prediction dataframe that have matching img_ids in the ground truth dataframe
    prediction_df = prediction_df[prediction_df.iloc[:, 0].isin(ground_truth_df.iloc[:, 0])]
    prediction_df.index = range(prediction_df.shape[0])


    # Extract the mask_rle columns
    pred_mask_rle = prediction_df.iloc[:, 1]
    gt_mask_rle = ground_truth_df.iloc[:, 1]


    def calculate_dice(pred_rle, gt_rle):
        pred_mask = rle_decode(pred_rle, img_shape)
        gt_mask = rle_decode(gt_rle, img_shape)


        if np.sum(gt_mask) > 0 or np.sum(pred_mask) > 0:
            return dice_score(pred_mask, gt_mask)
        else:
            return None  # No valid masks found, return None


    dice_scores = Parallel(n_jobs=-1)(
        delayed(calculate_dice)(pred_rle, gt_rle) for pred_rle, gt_rle in zip(pred_mask_rle, gt_mask_rle)
    )


    dice_scores = [score for score in dice_scores if score is not None]  # Exclude None values


    return np.mean(dice_scores)