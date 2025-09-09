import torch
import shutil
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

def get_metrics(labels_true, labels_pred, labels_to_ids):
    cls_report = metrics.classification_report(labels_true, labels_pred, target_names=labels_to_ids,
                                               labels=tuple(labels_to_ids.values()), zero_division=0.0, digits=5)
    macro_f1 = metrics.f1_score(labels_true, labels_pred, 
                                labels=tuple(labels_to_ids.values()), average='macro', zero_division=0.0)
    accuracy = metrics.accuracy_score(labels_true, labels_pred)
    return cls_report, macro_f1, accuracy

def plot_confmat(args, labels_true, labels_pred, labels_to_ids):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred,
                                                           labels=tuple(labels_to_ids.values()),
                                                           display_labels=labels_to_ids,
                                                           xticks_rotation='vertical')
    fig = disp.ax_.get_figure()
    fig.set_figwidth(args.FIG_SIZE)
    fig.set_figheight(args.FIG_SIZE)
    plt.show()

def save_model(pvm, plm, encoder, optimizer, path, model_name):
    save_path = path / model_name
    print(f'** Saving model to: {save_path} **')
    state = {"pvm": pvm.state_dict(),
             "plm": plm.state_dict(),
             "encoder": encoder.state_dict(),
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

def export_prediction(df, labels_pred, ids_to_labels, path, 
                      csv_name='submit.csv', zip_name='submit'):
    pred_dict = {'id': df['id'].tolist(),
                 'predict_label': map(ids_to_labels.get, labels_pred)}
    pd.DataFrame(pred_dict).to_csv(path / csv_name, index=False)
    shutil.make_archive(path / zip_name, 'zip', path, csv_name)