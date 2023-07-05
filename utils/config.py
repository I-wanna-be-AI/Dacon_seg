import argparse

def get_argparser():                    
    parser = argparse.ArgumentParser("Competition Version", add_help=False)
    parser.add_argument("--seed", type = int, required = False, default = 0)
    parser.add_argument("--gpu_ids", type = str)
    parser.add_argument("--img_size", type = int, default = 224, required = False)
    
    # train or infer
    parser.add_argument("--train", type = int, required = False, default = 0)
    parser.add_argument("--split", type = float, required = False, default = 0.15)
    parser.add_argument("--infer", type = int, required = False, default = 0)

    # model
    parser.add_argument("--model", type = str, required = False)
    parser.add_argument("--batchsize", type = int, default= 256, required = False)
    parser.add_argument("--epochs", type = int, default= 60, required = False)
    parser.add_argument("--lr", type = float, default= 1e-5, required = False)
    parser.add_argument("--one_cycle_max_lr", type = float, default= 1e-5, required = False)
    parser.add_argument("--one_cycle_pct_start", type = float, default= 0.1, required = False)
    
    # path
    parser.add_argument("--save_model", type = int, default = 1, required = False)
    parser.add_argument("--submit_path", type = str, default = "submit", required = False)
    parser.add_argument("--chkpt_path", type = str, default = "chkpt", required = False)
    parser.add_argument("--save_name", type = str, default = "", required = False)

    # # Wandb
    # parser.add_argument("--is_wandb", type = int, required = False, default = 1)
    # parser.add_argument("--wandb_name", type = str, required = False, default = 1)

    return parser.parse_args()