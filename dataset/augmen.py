from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms
import albumentations as A

def get_aug(args):

    train_aug = A.Compose([
        A.Resize(width=args.img_size, height=args.img_size),
        ToTensorV2()
    ])
    val_aug = A.Compose([
        A.Resize(width=args.img_size, height=args.img_size),
        ToTensorV2()
    ])




    return train_aug, val_aug
