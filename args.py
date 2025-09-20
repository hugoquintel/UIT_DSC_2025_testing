import torch
import argparse
from distutils.util import strtobool

# all the arguments the user can input
def get_arguments():
    parser = argparse.ArgumentParser('Hyperparameters for training and some additional arguments')
    parser.add_argument('--EPOCHS', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--PLM', type=str, default='Fsoft-AIC/videberta-base',
                        help='HuggingFace pre-trained language model (https://huggingface.co/models)')
    parser.add_argument('--PROMPT_CONTEXT_MAX_TOKEN', type=int, default=350,
                        help='Max number of tokens when tokenized by the pre-trained tokenizer for the prompt and context pair')
    parser.add_argument('--RESPONSE_MAX_TOKEN', type=int, default=100,
                        help='Max number of tokens when tokenized by the pre-trained tokenizer for the response')
    parser.add_argument('--WORD_SEG', type=strtobool, default=True,
                        help='Apply word segmentation to the inputs; this only works with Vietnamese \
                              (should be set to True only when working with language models requiring word segmentation, e.g., PhoBert).')
    parser.add_argument('--PLM_LR', type=float, default=1e-5,
                        help='Learning rate for pre-trained language model')
    parser.add_argument('--CLS_LR', type=float, default=1e-4,
                        help='Learning rate for the classifier block')
    parser.add_argument('--OPTIMIZER', type=str, default='AdamW',
                        help='Pytorch optimizer (check torch.optim for the full list)')
    parser.add_argument('--TRAIN_BATCH', type=int, default=8,
                        help='Number of instances in a batch during training')
    parser.add_argument('--DEV_BATCH', type=int, default=4,
                        help='Number of instances in a batch during inference')
    parser.add_argument('--PRINT_BATCH', type=int, default=200,
                        help='Print loss after a number of batches')
    parser.add_argument('--RANDOM_SEED', type=int, default=2025,
                        help='Random seed')
    parser.add_argument('--DATA_PATH', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--USE_DUMMY', type=strtobool, default=False,
                        help='Whether to create and use a dummy dataset by splitting the input train dataset into smaller \
                              datasets with numbers of labels proportional to the original one')
    parser.add_argument('--DUMMY_PATH', type=str, default='dummy_data',
                        help='Create path to dummy data directory')
    parser.add_argument('--DUMMY_DATASET', type=str, default='train',
                        help='The original dataset that will be used to make dummy data')
    parser.add_argument('--DUMMY_SAMPLES', type=int, default=7000,
                        help='Number of samples in the dummy dataset (must be equal or smaller than the original dataset)')
    parser.add_argument('--DUMMY_DEV_SIZE', type=float, default=0.1,
                        help='The proportion of the dummy dataset to include in the dev split (between 0.0 and 1.0)')
    
    parser.add_argument('--CONTINUE_FROM_CHECKPOINT', type=strtobool, default=False,
                        help='Whether to continue training from a previously saved model')
    parser.add_argument('--   ', type=str, default='',
                        help='Path to the saved model if continuing training from one')
    
    parser.add_argument('--GET_METRICS', type=strtobool, default=False,
                        help='Perform evaluation on the devset and show the metrics - sklearn classification report \
                             (Set to True only when the devset has labels or else the result will be wrong)')
    parser.add_argument('--PLOT_CONFMAT', type=strtobool, default=False,
                        help='Show confusion matrix at each epoch, recommend to set to False when executing in terminal. \
                              This option only works when GET_METRICS is True')
    parser.add_argument('--FIG_SIZE', type=float, default=2.5,
                        help='The height and width of the confusion matrix plot (matplotlib)')
    parser.add_argument('--FONT_SIZE', type=int, default=8,
                        help='The font size for matplotlib confusion matrix')
    parser.add_argument('--SAVE_MODEL', type=strtobool, default=True,
                        help='Save the model at each epoch while training')
    parser.add_argument('--SAVE_PATH', type=str, default='saved_models',
                        help='Create a folder (if not exist) to store all the saved models')
    parser.add_argument('--EXPORT_PREDICTION', type=strtobool, default=False,
                        help='Whether to export the predicted result (.csv and zip folder)')
    parser.add_argument('--PREDICTION_PATH', type=str, default='prediction',
                        help='Create a folder (if not exist) to store the prediction file')
    parser.add_argument('--PREDICTION_PER_EPOCH', type=strtobool, default=False,
                        help='Whether to export the prediction result at each epoch (zip the latest epoch), if no then just at the latest epoch')
    parser.add_argument('--TEST_BEST_MODEL', type=strtobool, default=True,
                        help='Evaluate the performance of the best model on the valset (SAVE_MODEL and GET_METRICS must be True)')
    parser.add_argument('--INFO_PATH', type=str, default='info',
                        help='Path to the info file')
    parser.add_argument('--INFO_FILE', type=str, default='info.txt',
                        help='Name of the information file')
    parser.add_argument('--MODELS_LIMIT', type=int, default=3,
                        help='The max number of saved models')
    args = parser.parse_args()
    # set up GPU for training (if available) agnostically
    args.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args
