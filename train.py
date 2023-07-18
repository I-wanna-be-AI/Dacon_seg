import torch
import time

from tqdm import tqdm

from utils import *
import wandb
import torch.distributed as dist

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt

def do_train(args, model, optimizer, criterion, train_dl, valid_dl, scheduler):

    print()
    if args.is_master:
        wandb.init(name = f"{args.model}", project = "Dacon_Segmentation", reinit = True, entity = args.wandb_id, config = args)
        print("Stat Train and Valid")

    best_loss = 1
    train_loss = 0
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

                train_loss += reduce_tensor(loss, args.world_size) if args.distributed else loss


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            scheduler.step()
        train_loss /= len(train_dl)
        test_loss, threshold = 0, 0.35
        print(f"Validation step, epoch: {epoch + 1}")
        model.eval()
        dice_scores=0
        for img, mask in valid_dl:         # # valid code
            img, mask = img.to(args.device, dtype=torch.float), mask.to(args.device, dtype=torch.float)
            with torch.no_grad():

                outputs = model(img)
                masks = torch.sigmoid(outputs).squeeze(1)
                masks = (masks>0.35).float()

                # masks = torch.sigmoid(outputs).cpu().numpy()
                # masks = np.squeeze(masks, axis=1)
                # masks = (masks>0.35).astype(np.uint8)

            loss = criterion(outputs, mask.unsqueeze(1))
            ds = dice_score_torch(masks, mask)
            if args.distributed:
                ds = reduce_tensor(ds, args.world_size)
                loss = reduce_tensor(loss, args.world_size)

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
                    "train Loss" : train_loss,
                    "test Loss" : test_loss,
                    "test Dice" : test_dice
                })
            