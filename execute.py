import os
import torch
import random
import pathlib
import matplotlib
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

# created modules
from args import get_arguments
from model import TransformerClassifier
from data_setup import make_dummy_data, preprocess_data, get_labels, LLMHallucinationDataset
from utils import train_step, test_step, get_metrics, plot_confmat, export_prediction, \
                  save_model, log_arguments, log_progress, test_best_model

def run():
    # get user's input arguments
    args = get_arguments()

    font = {'size': args.FONT_SIZE}
    matplotlib.rc('font', **font)

    random.seed(args.RANDOM_SEED)
    np.random.seed(args.RANDOM_SEED)
    torch.manual_seed(args.RANDOM_SEED)
    torch.cuda.manual_seed_all(args.RANDOM_SEED)

    if args.SAVE_MODEL:
        model_path = pathlib.Path(args.SAVE_PATH)
        model_path.mkdir(parents=True, exist_ok=True)
    if args.EXPORT_PREDICTION:
        pred_path = pathlib.Path(args.PREDICTION_PATH)
        pred_path.mkdir(parents=True, exist_ok=True)
    info_path = pathlib.Path(args.INFO_PATH)
    info_path.mkdir(parents=True, exist_ok=True)

    data_path = pathlib.Path(args.DATA_PATH)
    if args.USE_DUMMY:
        dummy_path = pathlib.Path(args.DUMMY_PATH)
        make_dummy_data(data_path, args.DUMMY_DATASET, dummy_path, args.DUMMY_SAMPLES, dev_size=args.DUMMY_DEV_SIZE, shuffle=True)
        data_path = pathlib.Path(args.DUMMY_PATH)

    tokenizer = AutoTokenizer.from_pretrained(args.PLM)
    train_df = preprocess_data(args, data_path, 'train', tokenizer)
    dev_df = preprocess_data(args, data_path, 'dev', tokenizer)

    labels_to_ids, ids_to_labels = get_labels(train_df)
    train_df['labels'] = train_df['labels'].map(labels_to_ids).fillna(train_df['labels']).astype(int)
    dev_df['labels'] = dev_df['labels'].map(labels_to_ids).fillna(dev_df['labels']).astype(int)

    train_data = LLMHallucinationDataset(train_df)
    dev_data = LLMHallucinationDataset(dev_df)

    no_workers = os.cpu_count()
    train_dataloader = DataLoader(train_data, batch_size=args.TRAIN_BATCH, shuffle=True,
                                  pin_memory=True, num_workers=no_workers)
    dev_dataloader = DataLoader(dev_data, batch_size=args.TEST_BATCH, shuffle=False,
                                pin_memory=True, num_workers=no_workers)
    
    
    prompts_contexts_plm = AutoModel.from_pretrained(args.PLM).to(args.DEVICE)
    responses_plm = AutoModel.from_pretrained(args.PLM).to(args.DEVICE)
    cls = TransformerClassifier(prompts_contexts_plm.config, labels_to_ids).to(args.DEVICE)

    train_params = ({'params': prompts_contexts_plm.parameters(), 'lr': args.PLM_LR},
                    {'params': responses_plm.parameters(), 'lr': args.PLM_LR},
                    {'params': cls.parameters(), 'lr': args.CLS_LR})
    
    optimizer_map = {'ASGD': optim.ASGD, 'Adadelta': optim.Adadelta, 'Adagrad': optim.Adagrad, 'Adam': optim.Adam,
                     'AdamW': optim.AdamW, 'Adamax': optim.Adamax, 'LBFGS': optim.LBFGS, 'NAdam': optim.NAdam, 'RAdam': optim.RAdam,
                     'RMSprop': optim.RMSprop, 'Rprop': optim.Rprop,'SGD': optim.SGD, 'SparseAdam': optim.SparseAdam}

    optimizer = optimizer_map[args.OPTIMIZER](train_params)
    loss_function = nn.CrossEntropyLoss()

    print(f'Number of samples in train set: {len(train_data)}')
    print(f'Number of samples in dev set: {len(dev_data)}')
    print(f'Number of train batches: {len(train_dataloader)}')
    print(f'Number of dev batches: {len(dev_dataloader)}')
    print(f'Labels to ids: {labels_to_ids}')
    print(f'Ids to labels: {ids_to_labels}\n')

    print(f'Arguments:')
    print('------------------------')
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('------------------------')


    best_macro_f1, best_accuracy, best_epoch = 0, 0, 0
    log_arguments(args, info_path)