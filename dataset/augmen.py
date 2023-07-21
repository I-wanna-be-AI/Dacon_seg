import cv2
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms
import albumentations as A

def get_aug(args):

    train_aug = A.Compose([
        A.RandomCrop(200,200),
        #A.RandomCrop(args.img_size, args.img_size),
        A.Resize(width=args.img_size, height=args.img_size),
        A.PadIfNeeded(min_height=224, min_width=224, p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        ], p=0.8),
        A.Normalize(),
        ToTensorV2()
    ])
    val_aug = A.Compose([
        A.RandomCrop(224,224),
        #A.RandomCrop(args.img_size, args.img_size),
        A.Resize(width=args.img_size, height=args.img_size),
        A.Normalize(),
        ToTensorV2()
    ])
    test_aug = A.Compose([
        A.Resize(width=args.img_size, height=args.img_size),
        A.Normalize(),
        ToTensorV2()
    ])




    return train_aug, val_aug, test_aug
