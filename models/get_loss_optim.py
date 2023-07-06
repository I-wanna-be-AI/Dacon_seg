import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def get_optimizer(args, model):

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr= args.lr)

    return optimizer, criterion

def get_scheduler(args, optimizer, train_dl):
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = args.one_cycle_max_lr, epochs = args.epochs,
                                                                steps_per_epoch=len(train_dl), 
                                                                pct_start=args.one_cycle_pct_start)

    return scheduler

