import argparse
import torch
import datetime
import json
import yaml
import os

from src.main_model import DiffPO
from src.utils import train, evaluate
from dataset_acic import get_dataloader

from PropensityNet import load_data

import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="DiffPO",
    notes="DiffPO"
)

torch.manual_seed(0)

parser = argparse.ArgumentParser(description="DiffPO")
parser.add_argument("--config", type=str, default="acic2018.yaml")
parser.add_argument("--current_id", type=str, default="")


parser.add_argument("--device", default="cuda", help="Device")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.2)
parser.add_argument("--nfold", type=int, default=1, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--train", type=int, default=1)

args = parser.parse_args()
print(args)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print('Dataset is:')
print(config["dataset"]["data_name"])

print(json.dumps(config, indent=4))

# Create folder
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/acic_fold" + str(args.nfold) + "_" + current_time + "/"
# print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


current_id = args.current_id
print('Start exe_acic on current_id', current_id)

# Every loader contains "observed_data", "observed_mask", "gt_mask", "timepoints"
train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
    dataset_name = config["dataset"]["data_name"],
    current_id = current_id
)

#=======================First train and fix propnet======================
# Train a propensitynet on this dataset

propnet = load_data(dataset_name = config["dataset"]["data_name"], current_id=current_id)
# frozen the trained_propnet
print('Finish training propnet and fix the parameters')
propnet.eval()
# ========================================================================

propnet = propnet.to(args.device)
model = DiffPO(config, args.device).to(args.device)

if train:
    # save training setting
    wandb.config = {"epochs": config["train"]["epochs"], "num_steps": config["diffusion"]["num_steps"],"lr": config["train"]["lr"]}

    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        valid_epoch_interval=config["train"]["valid_epoch_interval"],
        foldername=foldername,
        propnet = propnet
    )
    print('----------------Finish training------------')
    print('----------------Check trainresults---------')

    print("---------------Start testing---------------")

    evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
    # save test model
    directory = "./save_model/" + args.current_id
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), "./save_model/" + args.current_id + "/model_weights.pth")
    wandb.finish()

else:
    model.load_state_dict(torch.load("./save_model/ "+ args.current_id + "/model_weights.pth"))
