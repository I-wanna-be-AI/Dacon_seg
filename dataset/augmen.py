from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms
import albumentations as A

def get_aug(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_aug = A.Compose([
        A.Resize(width=args.img_size, height=args.img_size),
        ToTensorV2()
    ])
    val_aug = A.Compose([
        A.Resize(width=args.img_size, height=args.img_size),
        ToTensorV2()
    ])




    return train_aug, val_aug
