import torch
import time

from tqdm import tqdm

from utils import *
import wandb

def do_train(args, model, optimizer, criterion, train_dl, valid_dl, scheduler):

    print()
    if args.is_master:
        wandb.init(name = f"{args.model}", project = "Dacon_Segmentation", reinit = True, entity = "dk58319", config = args)
        print("Stat Train and Valid")

    best_loss = 1
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    for epoch in range(args.epochs):
        train_dl.sampler.set_epoch(epoch)

        if args.is_master:
            print(f"Epoch :  {epoch + 1}")

        model.train()
        for img, mask in tqdm(train_dl):
            img, mask = img.to(args.device, dtype=torch.float), mask.to(args.device, dtype=torch.float)
            optimizer.zero_grad()
            epoch_loss =0
            with torch.cuda.amp.autocast(enabled = True):    
                outputs= model(img)
                loss = criterion(outputs, mask.unsqueeze(1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            scheduler.step()

        true_label = []
        preds = []
        test_loss, threshold = 0, 0.35
        print(f"Validation step, epoch: {epoch + 1}")
        model.eval()
        dice_scores=0
        for img, mask in valid_dl:         # # valid code
            img, mask = img.to(args.device, dtype=torch.float), mask.to(args.device, dtype=torch.float)
            with torch.no_grad():

                outputs = model(img)
                masks = torch.sigmoid(outputs).cpu().numpy()
                masks = np.squeeze(masks, axis=1)
                masks = (masks>0.35).astype(np.uint8)


            loss = criterion(outputs, mask.unsqueeze(1))
            ds = dice_score(masks, mask.cpu().numpy())
            dice_scores+= ds
            test_loss += loss
        test_loss = test_loss/len(valid_dl)
        test_dice= dice_scores/len(valid_dl)

        if args.is_master:
            print(f" Loss : {test_loss}, Dice : {test_dice}")
            if args.save_model and best_loss > test_loss:
                best_loss = test_loss
                print("save", test_loss)
                save_model(args, model)

            wandb.log({
                    "Loss" : test_loss,
                    "Dice" : test_dice
                })
            