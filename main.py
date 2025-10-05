import os
import argparse
import time
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from configs import get_cfg_defaults
from utils import mkdir, graph_collate_func
from dataloader import DTIDataset
from model import ASHL
from trainer import  Trainer
import logging
from loggerConfig import LoggerConfig
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Drug-Target Interaction Prediction')
parser.add_argument('--cfg', required=True, type=str) 
parser.add_argument('--outname', required=True, type=str) 
parser.add_argument('--data', required=True, type=str) 

parser.add_argument('--num_worker', required=True, type=int)  # 0
args = parser.parse_args()

sub_dir = args.data
logger = LoggerConfig.setup_logger(
    name="ASHL",
    level=logging.INFO,
    log_format="%(asctime)s - %(levelname)s - %(message)s",
    log_prefix="training",
    sub_dir= sub_dir
)

def main():
    torch.cuda.empty_cache()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    datafolder = f'{cfg.Data.Path}/{args.data}'
    logger.info(f"datafolder: {datafolder}")
    outfolder = f'./output/{args.outname}' if device == torch.device(
        'cpu') else f'{cfg.Result.Output_Dir}/{args.outname}'
    logger.info(f"save folder: {outfolder}")
    mkdir(outfolder)


    # load data
    train_data = pd.read_csv(os.path.join(datafolder, 'train.csv'))
    val_data = pd.read_csv(os.path.join(datafolder, 'val.csv'))
    test_data = pd.read_csv(os.path.join(datafolder, 'test.csv'))
    train_dataset = DTIDataset(train_data.index.values, train_data, max_drug_nodes=cfg.Drug.Nodes,
                               max_protein_length=cfg.Protein.Length)
    val_dataset = DTIDataset(val_data.index.values, val_data, max_drug_nodes=cfg.Drug.Nodes,
                             max_protein_length=cfg.Protein.Length)
    test_dataset = DTIDataset(test_data.index.values, test_data, max_drug_nodes=cfg.Drug.Nodes,
                              max_protein_length=cfg.Protein.Length)
    train_size = len(train_dataset)
    logger.info('Begin training')

    params = {'batch_size': cfg.Global.Batch_Size, 'shuffle': True, 'num_workers': args.num_worker,
              'drop_last': True, 'collate_fn': graph_collate_func}

    train_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)

    test_generator = DataLoader(test_dataset, **params)


    model = ASHL(cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.Global.LR, weight_decay=5e-5)

    loss = nn.CrossEntropyLoss()  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=10)

    with open(os.path.join(outfolder, "model_configs.txt"), "w") as f:
        f.write(str(cfg))
    with open(os.path.join(outfolder, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    start = time.time()
    trainer = Trainer(model, opt, loss, device, train_generator, val_generator, test_generator, cfg, outfolder, scheduler, logger)
    trainer.set_tensorboard(path=outfolder)
    trainer.train()
    end = time.time()

    logger.info(f"End! Total running time: {round((end - start) / 60, 2)} min")


if __name__ == '__main__':
    main()
