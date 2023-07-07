import os
import torch
import numpy as np
import pandas as pd

def inference(args, model, test_dl, submit_df):
    model.load_state_dict(torch.load("chkpt/unet_baseunet_base_model_768.pt", map_location = args.device))
    
    if args.is_master:
        print("model Evaluate")
    
    preds = []
    filenames = []
    threshold = 0.5
    model.eval()
    for img, fname in test_dl:        
        img = img.to(args.device)
        
        with torch.no_grad():
            pred_v2 = model(img)
            model_pred = pred_v2.squeeze(1).to('cpu')
            preds += model_pred.tolist()
        filenames.extend(fname)
            
    preds = np.where(np.array(preds) > threshold, 1, 0)

    if args.is_master:
        # Save predictions according to the sample submission format
        pred_df = pd.DataFrame({'ImageId':filenames, 'answer': preds})
        result = submit_df.merge(pred_df, on='ImageId', how='left')
        result.drop('answer_x', axis=1, inplace=True)
        result.rename(columns={'answer_y':'answer'}, inplace=True)`
        result.to_csv(os.path.join(args.submit_path,'unet_baseunet_base_model_768.csv'), index=False)
        # submit_df.to_csv(os.path.join(args.submit_path,'effb0_224.csv'), index=False)
        print("success")
    
    