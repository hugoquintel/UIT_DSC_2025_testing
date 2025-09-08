import torch
import pandas as pd
from pyvi import ViTokenizer
from torch.utils.data import Dataset

def get_labels(df):
    labels = df['label'].unique()
    labels_to_ids = {label:index for index, label in enumerate(labels)}
    ids_to_labels = {index:label for label, index in labels_to_ids.items()}
    return labels_to_ids, ids_to_labels


def preprocess_data(args, df, tokenizer, labels_to_ids):
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
    labels = df['label'].map(labels_to_ids)
    data_dict = {'prompts_contexts_input_ids': prompts_contexts_output.input_ids,
                 'prompts_contexts_attention_mask': prompts_contexts_output.attention_mask,
                 'responses_input_ids': responses_output.input_ids,
                 'responses_attention_mask': responses_output.attention_mask,
                 'labels': labels}
    return pd.DataFrame(data_dict)

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