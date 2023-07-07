from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms
import albumentations as A

def get_aug(args):

    train_aug = A.Compose([
        A.Resize(width=args.img_size, height=args.img_size),
        A.Normalize(),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.5),
        # A.ShiftScaleRotate(p=0.5),
        # A.RandomBrightnessContrast(p=0.5),
        ToTensorV2()
    ])
    val_aug = A.Compose([
        A.Resize(width=args.img_size, height=args.img_size),
        A.Normalize(),
        ToTensorV2()
    ])




    return train_aug, val_aug
