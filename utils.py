import os
import torch
import shutil
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

def train_step(args, prompts_contexts_plm, responses_plm,
               cls, loss_function, optimizer, dataloader):
    prompts_contexts_plm.train()
    responses_plm.train()
    cls.train()
    loss_total = 0
    for batch_index, data in enumerate(dataloader):
        prompts_contexts_input_ids = data['prompts_contexts_input_ids'].to(args.DEVICE)
        prompts_contexts_attention_mask = data['prompts_contexts_attention_mask'].to(args.DEVICE)
        responses_input_ids = data['responses_input_ids'].to(args.DEVICE)
        responses_attention_mask = data['responses_attention_mask'].to(args.DEVICE)
        labels = data['labels'].to(args.DEVICE)
        prompts_contexts_logit = prompts_contexts_plm(input_ids=prompts_contexts_input_ids, attention_mask=prompts_contexts_attention_mask).last_hidden_state
        responses_logit = responses_plm(input_ids=responses_input_ids, attention_mask=responses_attention_mask).last_hidden_state
        logit = cls(prompts_contexts_logit, responses_logit)[:, 0, :]
        loss = loss_function(logit, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss
        if (batch_index+1)%args.PRINT_BATCH == 0:
            print(f'Loss after {batch_index+1} batches: {loss:.5f}')
    loss_average = loss_total / len(dataloader)
    return loss_total, loss_average

def test_step(args, prompts_contexts_plm,
              responses_plm, cls, dataloader):
    prompts_contexts_plm.eval()
    responses_plm.eval()
    cls.eval()
    labels_true, labels_pred = [], []
    with torch.inference_mode():
        for batch_index, data in enumerate(dataloader):
            prompts_contexts_input_ids = data['prompts_contexts_input_ids'].to(args.DEVICE)
            prompts_contexts_attention_mask = data['prompts_contexts_attention_mask'].to(args.DEVICE)
            responses_input_ids = data['responses_input_ids'].to(args.DEVICE)
            responses_attention_mask = data['responses_attention_mask'].to(args.DEVICE)
            labels = data['labels'].to(args.DEVICE)
            prompts_contexts_logit = prompts_contexts_plm(input_ids=prompts_contexts_input_ids, attention_mask=prompts_contexts_attention_mask).last_hidden_state
            responses_logit = responses_plm(input_ids=responses_input_ids, attention_mask=responses_attention_mask).last_hidden_state
            logit = cls(prompts_contexts_logit, responses_logit)[:, 0, :]
            labels_true.extend(labels.tolist())
            labels_pred.extend(logit.argmax(dim=-1).tolist())
    return labels_true, labels_pred

def get_metrics(labels_true, labels_pred, labels_to_ids):
    cls_report = metrics.classification_report(labels_true, labels_pred, target_names=labels_to_ids,
                                               labels=tuple(labels_to_ids.values()), zero_division=0.0, digits=5)
    macro_f1 = metrics.f1_score(labels_true, labels_pred, 
                                labels=tuple(labels_to_ids.values()), average='macro', zero_division=0.0)
    return cls_report, macro_f1

def plot_confmat(args, labels_true, labels_pred, labels_to_ids):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred,
                                                           labels=tuple(labels_to_ids.values()),
                                                           display_labels=labels_to_ids,
                                                           xticks_rotation='vertical')
    fig = disp.ax_.get_figure()
    fig.set_figwidth(args.FIG_SIZE)
    fig.set_figheight(args.FIG_SIZE)
    plt.show()

def save_model(prompts_contexts_plm, responses_plm, 
               cls, optimizer, path, model_name):
    save_path = path / model_name
    print(f'** Saving model to: {save_path} **')
    state = {"prompts_contexts_plm": prompts_contexts_plm.state_dict(),
             "responses_plm": responses_plm.state_dict(),
             "cls": cls.state_dict(),
             "optimizer": optimizer.state_dict()}
    torch.save(state, save_path)

def log_arguments(args, path):
    with open(path / args.INFO_FILE, "w") as f:
        f.write(f'Arguments:\n')
        f.write('------------------------\n')
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
        f.write('------------------------\n')

def log_progress(args, epoch, loss_total, loss_average, path, cls_report=None):
    with open(path / args.INFO_FILE, "a") as f:
        f.write(f'\nepoch {epoch}:\n')
        f.write(f'Total loss: {loss_total:.5f} | Average loss: {loss_average:.5f}\n')
        if cls_report:
            f.write(f'Classification report:\n{cls_report}')

def export_prediction(df, labels_pred, ids_to_labels, path, csv_name, zip_name):
    pred_dict = {'id': df['ids'].tolist(),
                 'predict_label': map(ids_to_labels.get, labels_pred)}
    pd.DataFrame(pred_dict).to_csv(path / csv_name, index=False)
    shutil.make_archive(path / zip_name, 'zip', path, csv_name)

def test_best_model(args, labels_to_ids, dataloader, model_path, 
                    prompts_contexts_plm, responses_plm, cls):
    saved_models = sorted(float(model[:-3]) for model in os.listdir(model_path) if model.split('.')[-1] == 'pt')
    state = torch.load(model_path / f'{saved_models[-1]}.pt', weights_only=False)
    prompts_contexts_plm.load_state_dict(state['plm'])
    responses_plm.load_state_dict(state['pvm'])
    cls.load_state_dict(state['encoder'])
    labels_true, labels_pred = test_step(args, prompts_contexts_plm,
                                         responses_plm, cls, dataloader)
    return labels_true, labels_pred