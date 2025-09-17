import os
import sys
import tqdm
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
        make_dummy_data(data_path, args.DUMMY_DATASET, dummy_path, 
                        args.DUMMY_SAMPLES, dev_size=args.DUMMY_DEV_SIZE)
        data_path = pathlib.Path(args.DUMMY_PATH)

    tokenizer = AutoTokenizer.from_pretrained(args.PLM)
    train_df = preprocess_data(args, data_path, 'train', tokenizer)
    dev_df = preprocess_data(args, data_path, 'dev', tokenizer)

    labels_to_ids, ids_to_labels = get_labels(train_df)
    train_df['labels'] = train_df['labels'].map(labels_to_ids).fillna(0).astype(int)
    dev_df['labels'] = dev_df['labels'].map(labels_to_ids).fillna(0).astype(int)

    train_data = LLMHallucinationDataset(train_df)
    dev_data = LLMHallucinationDataset(dev_df)

    no_workers = os.cpu_count()
    train_dataloader = DataLoader(train_data, batch_size=args.TRAIN_BATCH, shuffle=True,
                                  pin_memory=True, num_workers=no_workers)
    dev_dataloader = DataLoader(dev_data, batch_size=args.DEV_BATCH, shuffle=False,
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

    print('Loading model from checkpoint...\n')
    if args.CONTINUE_FROM_CHECKPOINT:
        checkpoint_path = pathlib.Path(args.CHECKPOINT_PATH)
        state = torch.load(checkpoint_path, weights_only=False)
        prompts_contexts_plm.load_state_dict(state['prompts_contexts_plm'])
        responses_plm.load_state_dict(state['responses_plm'])
        cls.load_state_dict(state['cls'])
        optimizer.load_state_dict(state['optimizer'])

    print(f'Number of samples in train set: {len(train_data)}')
    print(f'Number of samples in dev set: {len(dev_data)}')
    print(f'Number of train batches: {len(train_dataloader)}')
    print(f'Number of dev batches: {len(dev_dataloader)}')
    print(f'Labels to ids: {labels_to_ids}')
    print(f'Ids to labels: {ids_to_labels}\n')

    if args.DEVICE == 'cuda':
        if torch.cuda.device_count() > 1:
            print('--Using multiple GPUs to train--\n')
            prompts_contexts_plm = torch.nn.DataParallel(prompts_contexts_plm)
            responses_plm = torch.nn.DataParallel(responses_plm)
            cls = torch.nn.DataParallel(cls)
        else: print('--Using single GPU to train--\n')
    else: print('--No GPU detected, using CPU to train--\n')

    print(f'Arguments:')
    print('------------------------')
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('------------------------')

    best_macro_f1 = 0
    log_arguments(args, info_path)
    for epoch in tqdm.trange(args.EPOCHS, file=sys.stdout):
        print(f'\n\nEpoch {epoch}:')
        # train
        print('-----------')
        loss_total, loss_average = train_step(args, prompts_contexts_plm, responses_plm,
                                            cls, loss_function, optimizer, train_dataloader)
        print(f'Total loss: {loss_total:.5f} | Average loss: {loss_average:.5f}')
        print('-----------')
        # test
        labels_dev_true, labels_dev_pred = test_step(args, prompts_contexts_plm,
                                                     responses_plm, cls, dev_dataloader)
        
        if args.GET_METRICS:
            cls_report, macro_f1 = get_metrics(labels_dev_true, labels_dev_pred, labels_to_ids)
            print('[+] METRICS:')
            print(f'Classification report:\n{cls_report}')
            log_progress(args, epoch, loss_total, loss_average, info_path, cls_report)
            if args.PLOT_CONFMAT:
                plot_confmat(args, labels_dev_true, labels_dev_pred, labels_to_ids)
            if args.SAVE_MODEL:
                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    save_model(prompts_contexts_plm, responses_plm, cls, 
                               optimizer, model_path, f"{round(best_macro_f1, 4)}.pt")
                    saved_models = sorted(float(model[:-3]) for model in os.listdir(model_path) if model.split('.')[-1] == 'pt')
                    if len(saved_models) > args.MODELS_LIMIT:
                        os.remove(model_path / f'{saved_models[0]}.pt')

        else:
            log_progress(args, epoch, loss_total, loss_average, info_path)
            if args.SAVE_MODEL:
                save_model(prompts_contexts_plm, responses_plm, cls,
                           optimizer, model_path, f'epoch_{epoch}.pt')
                saved_models = sorted(int(model[:-3].split('_')[1]) for model in os.listdir(model_path) if model.split('.')[-1] == 'pt')
                if len(saved_models) > args.MODELS_LIMIT:
                    os.remove(model_path / f'epoch_{saved_models[0]}.pt')

        if args.EXPORT_PREDICTION:
            if args.PREDICTION_PER_EPOCH:
                export_prediction(dev_df, labels_dev_pred, ids_to_labels, pred_path, 
                                  csv_name=f'epoch_{epoch}.csv', zip_name='prediction')
            else:
                export_prediction(dev_df, labels_dev_pred, ids_to_labels, pred_path,
                                  csv_name=f'prediction.csv', zip_name='prediction')
                
    if args.TEST_BEST_MODEL and args.GET_METRICS and args.SAVE_MODEL:
        print('\n\n\n******TESTING THE BEST MODEL******')
        labels_dev_true, labels_dev_pred = test_best_model(args, labels_to_ids, dev_dataloader, model_path,
                                                           prompts_contexts_plm, responses_plm, cls)
        cls_report, _ = get_metrics(labels_dev_true, labels_dev_pred, labels_to_ids)
        print('[+] METRICS:')
        print(f'Classification report:\n{cls_report}')
        if args.PLOT_CONFMAT:
            plot_confmat(args, labels_dev_true, labels_dev_pred, labels_to_ids)
        if args.EXPORT_PREDICTION:
            export_prediction(dev_df, labels_dev_pred, ids_to_labels, pred_path, 
                              csv_name=f'best_prediction.csv', zip_name='best_prediction')
        print('**************FINISH**************')