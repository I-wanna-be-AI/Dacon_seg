from segmentation_models_pytorch import losses
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
def get_optimizer(args, model):

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    return optimizer, criterion

def get_scheduler(args, optimizer, train_dl):
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = args.one_cycle_max_lr, epochs = args.epochs,
    #                                                             steps_per_epoch=len(train_dl),
    #                                                             pct_start=args.one_cycle_pct_start)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.one_cycle_max_lr, epochs=args.epochs,
                                              steps_per_epoch=len(train_dl))

    return scheduler

