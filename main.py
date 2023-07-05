import warnings

from torch import device
from tqdm import tqdm

from utils import *
from models import *
from dataset import *
from train import *


warnings.filterwarnings("ignore")



if __name__ == "__main__":
    args = get_argparser()
    init_cuda_distributed(args)
    seed_everything(args.seed)
    
    modeled = get_model(args)
    optimizer, criterion = get_optimizer(args, modeled)

    if args.train:

        train_transform , valid_transform = get_aug(args)
        dataset = SatelliteDataset(csv_file='./data/train.csv', transform=train_transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
        #train_dl, valid_dl, train_sampler = get_dataloader(args)
        scheduler = get_scheduler(args, optimizer, train_loader)
        do_train(args, modeled, optimizer, criterion, train_loader, val_loader, scheduler)


    if args.infer:
        #test_dl, test_df = get_testloader(args)
        _,transform = get_aug(args)
        test_dataset = SatelliteDataset(csv_file='./data/test.csv', transform=transform, infer=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
        model = modeled
        model.load_state_dict(torch.load("chkpt/unet_baseunet_base_model_768.pt", map_location=args.device))
        with torch.no_grad():
            model.eval()
            result = []
            for images in tqdm(test_loader):
                images = images.float().to(args.device)

                outputs = model(images)
                masks = torch.sigmoid(outputs).cpu().numpy()
                masks = np.squeeze(masks, axis=1)
                masks = (masks > 0.35).astype(np.uint8)  # Threshold = 0.35

                for i in range(len(images)):
                    mask_rle = rle_encode(masks[i])
                    if mask_rle == '':  # 예측된 건물 픽셀이 아예 없는 경우 -1
                        result.append(-1)
                    else:
                        result.append(mask_rle)

        submit = pd.read_csv('./data/sample_submission.csv')
        submit['mask_rle'] = result
        submit.to_csv('./submit/submit.csv', index=False)

