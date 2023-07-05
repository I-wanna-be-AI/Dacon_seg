import torch
import timm
import Unet_baseline

from torch.nn.parallel import DistributedDataParallel



def get_model(args):
    if args.model == "efficientnet_b0":
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes = 1)

    elif args.model == "unet_base":
        model=Unet_baseline.UNet()
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.local_rank)
    model = DistributedDataParallel(model, static_graph=False, device_ids=[args.local_rank])
    
    return model