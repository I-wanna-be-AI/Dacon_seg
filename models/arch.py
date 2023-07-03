import torch
import timm

from torchvision import models
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

def get_model(args):
    if args.model == "efficientnet_b0":
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes = 1)
    elif args.model == "efficientnet_b1":
        model = timm.create_model('efficientnet_b1', pretrained=True, num_classes = 1)
    elif args.model == "efficientnet_b2":
        model = timm.create_model('efficientnet_b2', pretrained=True, num_classes = 1)
    elif args.model == "efficientnet_b3":
        model = timm.create_model('efficientnet_b3', pretrained=True, num_classes = 1)
    elif args.model == "efficientnet_b4":
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes = 1)
    elif args.model == "efficientnet_b5":
        model = timm.create_model('efficientnet_b5', pretrained=True, num_classes = 1)
    elif args.model == "efficientnet_b6":
        model = timm.create_model('efficientnet_b6', pretrained=True, num_classes = 1)
    elif args.model == "convenxt":
        model = timm.create_model('convnext_base', pretrained=True, num_classes = 1)
    elif args.model == "efficientnetv2":
        model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True, num_classes = 1)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.local_rank)
    model = DistributedDataParallel(model, static_graph=False, device_ids=[args.local_rank])
    
    return model