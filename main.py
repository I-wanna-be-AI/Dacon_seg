import warnings

from torch import device
from tqdm import tqdm

from utils import *
from models import *
from dataset import *
from train import *
from infer import *

warnings.filterwarnings("ignore")



if __name__ == "__main__":
    args = get_argparser()
    init_cuda_distributed(args)
    seed_everything(args.seed)
    
    modeled = get_model(args)
    optimizer, criterion = get_optimizer(args, modeled)

    if args.train:

        train_transform, valid_transform = get_aug(args)
        dataset = SatelliteDataset(csv_file='./data/train.csv')
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)
        #train_dl, valid_dl, train_sampler = get_dataloader(args)
        scheduler = get_scheduler(args, optimizer, train_loader)
        do_train(args, modeled, optimizer, criterion, train_loader, val_loader, scheduler)


    if args.infer:
        #test_dl, test_df = get_testloader(args)
        _,transform = get_aug(args)
        test_dataset = SatelliteDataset(csv_file='./data/test.csv', transform=transform, infer=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)
        inference(args, modeled, test_loader)

