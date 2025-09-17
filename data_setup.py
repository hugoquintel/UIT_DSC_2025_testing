import torch
import pandas as pd
from pyvi import ViTokenizer
from torch.utils.data import Dataset

def make_dummy_data(data_path, dataset, dummy_path, no_samples, dev_size):
    df = pd.read_csv(data_path / f'{dataset}.csv')
    dummy_path.mkdir(parents=True, exist_ok=True)
    labels_count = df['label'].value_counts()
    unique_labels = labels_count.index
    no_labels = labels_count.sum()
    dummy_labels_count = {label: round(value/no_labels*no_samples) for label, value in zip(unique_labels, labels_count)}
    train_labels_count, dev_labels_count = {}, {}
    for label in dummy_labels_count:
        dev_labels_count[label] = round(dummy_labels_count[label]*dev_size)
        train_labels_count[label] = dummy_labels_count[label]-dev_labels_count[label]
    train_df, dev_df = pd.DataFrame(), pd.DataFrame()
    for label in unique_labels:
        train_df = pd.concat([train_df, df[df['label']==label].iloc[:train_labels_count[label]]], axis=0)
        dev_df = pd.concat([dev_df, df[df['label']==label].iloc[train_labels_count[label]:train_labels_count[label]+dev_labels_count[label]]], axis=0)
    train_df.sample(frac=1, ignore_index=True).to_csv(dummy_path / 'train.csv', index=False)
    dev_df.sample(frac=1, ignore_index=True).to_csv(dummy_path / 'dev.csv', index=False)

def preprocess_data(args, path, dataset, tokenizer):
    df = pd.read_csv(path / f'{dataset}.csv')
    prompts = df['prompt'].tolist()
    contexts = df['context'].tolist()
    responses = df['response'].tolist()
    if args.WORD_SEG:
        prompts = [ViTokenizer.tokenize(prompt) for prompt in prompts]
        contexts = [ViTokenizer.tokenize(context) for context in contexts]
        responses = [ViTokenizer.tokenize(response) for response in responses]
    prompts_contexts_output = tokenizer(prompts, contexts, padding='max_length',
                                        truncation=True, max_length=args.PROMPT_CONTEXT_MAX_TOKEN)
    responses_output = tokenizer(responses, padding='max_length',
                                 truncation=True, max_length=args.RESPONSE_MAX_TOKEN)
    data_dict = {'ids': df['id'].tolist(),
                 'prompts_contexts_input_ids': prompts_contexts_output.input_ids,
                 'prompts_contexts_attention_mask': prompts_contexts_output.attention_mask,
                 'responses_input_ids': responses_output.input_ids,
                 'responses_attention_mask': responses_output.attention_mask,
                 'labels': df[[column for column in df.columns if 'label' in column][0]].tolist()}
    return pd.DataFrame(data_dict)

def get_labels(df):
    labels = df['labels'].unique()
    labels_to_ids = {label:index for index, label in enumerate(labels)}
    ids_to_labels = {index:label for label, index in labels_to_ids.items()}
    return labels_to_ids, ids_to_labels

class LLMHallucinationDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        data = {'prompts_contexts_input_ids': torch.tensor(self.df['prompts_contexts_input_ids'].iloc[index]),
                'prompts_contexts_attention_mask': torch.tensor(self.df['prompts_contexts_attention_mask'].iloc[index]),
                'responses_input_ids': torch.tensor(self.df['responses_input_ids'].iloc[index]),
                'responses_attention_mask': torch.tensor(self.df['responses_attention_mask'].iloc[index]),
                'labels': self.df['labels'].iloc[index]}
        return data