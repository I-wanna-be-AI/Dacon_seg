import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def get_optimizer(args, model):

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr= args.lr)

    return optimizer, criterion

def get_scheduler(args, optimizer, train_dl):
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = args.lr, epochs = args.epochs,
                                                                steps_per_epoch=len(train_dl), 
                                                                )

    return scheduler

