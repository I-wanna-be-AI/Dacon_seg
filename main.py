import warnings
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
        train_dl, valid_dl, train_sampler = get_dataloader(args)
        scheduler = get_scheduler(args, optimizer, train_dl)
        do_train(args, modeled, optimizer, criterion, train_dl, valid_dl, train_sampler, scheduler)
    
    if args.infer:
        test_dl, test_df = get_testloader(args)
        inference(args, modeled, test_dl, test_df)