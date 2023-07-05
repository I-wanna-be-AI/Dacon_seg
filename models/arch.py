import torch
import timm
import segmentation_models_pytorch as smp

from torch.nn.parallel import DistributedDataParallel

from models.Unet_baseline import UNet


def get_model(args):
    if args.model == "efficientnet_b0":
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes = 1)

    elif args.model == "unet_base":
        model=UNet()

    elif args.model == "unet_resnext101":
        model = smp.Unet(encoder_name="resnext101_32x8d", encoder_weights="imagenet", in_channels=3,   classes=1)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.local_rank)
    model = DistributedDataParallel(model, static_graph=False, device_ids=[args.local_rank])
    
    return model