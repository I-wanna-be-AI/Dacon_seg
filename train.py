import torch
import time
from utils import *
import wandb

def do_train(args, model, optimizer, criterion, train_dl, valid_dl, train_sampler, scheduler):

    print()
    if args.is_master:
        wandb.init(name = f"{args.model}", project = "AI or Not Competition", reinit = True, entity = "psboys", config = args)
        print("Stat Train and Valid")

    best_loss = 1e10
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        if args.is_master:
            print(f"Epoch :  {epoch + 1}")

        model.train()
        for img, label in train_dl:
            img, label = img.to(args.device, dtype=torch.float), label.to(args.device, dtype=torch.float)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled = True):    
                pred = model(img)
                loss = criterion(pred.squeeze(-1), label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
        # scheduler.step()

        true_label = []
        preds = []
        test_loss, threshold = 0, 0.5

        model.eval()
        for img, label in valid_dl:         # # valid code
            img, label = img.to(args.device, dtype=torch.float), label.to(args.device, dtype=torch.float)
            with torch.no_grad():
                pred_v2 = model(img)
                model_pred = pred_v2.squeeze(1).to('cpu')
                preds += model_pred.tolist()
                true_label += label.tolist()
            
            loss = criterion(pred_v2.squeeze(-1), label)
            test_loss += loss
        
        preds = np.where(np.array(preds) > threshold, 1, 0)
        acc, f1_score = get_metric(np.array(true_label), preds)

        if args.is_master:
            print(f"Accuracy : {acc}   f1 Score : {f1_score}  Loss : {test_loss}")
            if args.save_model and best_loss > test_loss:
                best_loss = test_loss
                print("save", test_loss)
                save_model(args, model)

            wandb.log({
                    "Valid_Acc" : acc,
                    "Valid_F1" : f1_score,
                    "Loss" : test_loss, 
                })
            