import os

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.rle import rle_encode


def inference(args, model, test_dataloader):
    model.load_state_dict(torch.load("./chkpt/unet_mlt_b4_model.pt", map_location=args.device))

    if args.is_master:
        print("model Evaluate")

    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(args.device)

            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8)  # Threshold = 0.35
            # #resize mask to original size
            # masks = np.array([cv2.resize(mask, (224, 224)) for mask in masks])

            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '':  # 예측된 건물 픽셀이 아예 없는 경우 -1
                    result.append(-1)
                    #print("-1")
                else:
                    result.append(mask_rle)
                    #print(mask_rle)

    submit = pd.read_csv('./data/sample_submission.csv')
    submit['mask_rle'] = result
    submit.to_csv('./submit/submit_unet_mlt_th0.4.csv', index=False)
    print("success")