import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoConfig
from functools import partial
import pandas as pd
import numpy as np
import argparse
import os
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, tokenizer, max_length=512):
        texts = data['agg']
        input_ids_all = []
        attention_masks = []
        for text in texts.values:
            ids = tokenizer(text)
            input_ids, attention_mask = ids['input_ids'], ids['attention_mask']

            input_ids_all.append(input_ids[:max_length])
            attention_masks.append(attention_mask[:max_length])

        self.input_ids_all = input_ids_all
        self.attention_masks = attention_masks
        if len(targets) == 0:
            self.labels = [-100] * len(texts)
        else:
            self.labels = targets

    def __len__(self):
        return len(self.input_ids_all)

    def __getitem__(self, idx):
        input_ids = self.input_ids_all[idx]
        attention_mask = self.attention_masks[idx]
        label = self.labels[idx]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }



def custom_collate_fn(batch, pad_idx):

    max_len = max(len(text['input_ids']) for text in batch)

    input_ids = [text['input_ids'] + [pad_idx] * (max_len - len(text['input_ids'])) for text in batch]
    attention_mask = [text['attention_mask'] + [0] * (max_len - (len(text['attention_mask']))) for text in batch]
    labels = [text['label'] for text in batch]

    return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels), 'attention_mask': torch.tensor(attention_mask)}


def get_data(model_name, batch_size=4):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    global pad_idx
    pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    config = AutoConfig.from_pretrained(model_name)

    max_length = 512

    dataset_test = CustomDataset(test_reviews, [], tokenizer, max_length=max_length)

    _custom_collate_fn = partial(custom_collate_fn, pad_idx=pad_idx)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=_custom_collate_fn, shuffle=False)

    return dataloader_test



class TextClassifier(torch.nn.Module):
    def __init__(self, hidden_size, do_activation=False, dropout=0.1):
        super(TextClassifier, self).__init__()
        self.output_dim = 1
        self.bert_layer = BertModel.from_pretrained(model_name)
        self.dropout_rate = dropout
        self.do_activation = do_activation
        if dropout != 0:
            self.dropout=nn.Dropout(dropout)
        if do_activation:
            self.activation = nn.ReLU()
        self.linear_relevance = nn.Linear(in_features=768, out_features=1)
        self.linear_sentiment = nn.Linear(in_features=768, out_features=1)
        self.linear_object = nn.Linear(in_features=768, out_features=3)

    def forward(self, input_ids, attention_mask):
        x = input_ids

        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample

        x = self.bert_layer(x, attention_mask=attention_mask)[1]

        if self.dropout_rate != 0:
            x = self.dropout(x)
        if self.do_activation:
            x = self.activation(x)
        rel = self.linear_relevance(x)
        sent = self.linear_sentiment(x)
        obj = self.linear_object(x)
        return rel, sent, obj

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import pandas as pd
from test_script import get_preds
# from data_and_models import check_data, get_pred, convert_df, plot_shap_explanation
import numpy as np
import matplotlib.pyplot as plt
# from aux_functions import *
import os
import re
import argparse


def process_punctuation_txt(review):
  # Обрабатываем знаки препинания
    review = re.sub(rf'(\W)', rf' \1 ', review)
    # print('review', review)

    review = re.sub(rf'\s+', rf' ', review)
    # print('review', review)
    for punct in '.,':
        review = re.sub(rf'([\d\{punct}]) (\{punct}) ([\d\{punct}])', rf'\1\2\3', review)
    for punct in '!?':
        review = re.sub(rf'(\{punct}) (\{punct}) (\{punct})', rf'\1\2\3', review)
    # print('review', review)
    if review[-1] == ' ':
        review = review[:-1]
    if review[-1] not in '.!?':
        review += ' .'
    review += '####[]'
    return review


def add_spaces(review):
    pattern=r'(\w\w[а-яё])([А-Я]\w\w)'
    review=re.sub(pattern, r'\1 \2', review)
    return review
    


def process_reviews(reviews):
    reviews = reviews.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    # reviews = reviews.apply(lambda x: x + '####[]')
    reviews = reviews.apply(add_spaces)
    reviews = reviews.apply(process_punctuation_txt)
        
    return reviews.values.tolist()



def file_to_triplets(reviews):

    # if file.name.rsplit('.', 1)[1].lower() == 'txt':
    # with open(file) as f:
    #     lines = f.read().split('####\n') 

        # lines = [line.rsplit('####')[0]+'####[]' for line in lines]
    # else:
    lines = process_reviews(reviews)



    with open('data/uploaded_file.txt', 'w') as f:
        for line in lines:
            f.write(line+'\n')
            # st.write(line[-1])
   
    # file.seek(0)
    print('Started inference')
    preds = get_preds(lines, model_path='model_58.pt', test_path='data/uploaded_file.txt')

    return preds





def get_aste_preds(reviews):
    

       
        preds = file_to_triplets(reviews)


        triplets = preds['pred_text'].values.tolist()
        all_triplets = []
        for label in triplets:
            # all_triplets.extend(eval(label))
            all_triplets.extend(label)

        data = []
        for triplet in all_triplets:
            at, ot, sp = triplet
            at = at.lower()
            ot = ot.lower()

            data.append([at, ot, sp, (at, ot, sp), (at, ot)])

        df = pd.DataFrame(data)
        df.columns = ['aspect', 'opinion', 'sentiment', 'triplet', 'aspect_opinion']

        # df.to_csv(args.save_to, index=False)
        # preds.to_csv(args.save_to[:-4]+'_preds.csv', index=False)   
        
        return preds, df
    
    
    
    
    
    
    
    
def get_emotions(questions):
    model_emotions = pipeline(model="seara/rubert-tiny2-ru-go-emotions")
    
    labels = []
    for a in questions:
        labels.append(model_emotions(a)[0]['label'])
    return labels
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/sample_reviews.txt", type=str)
    parser.add_argument("--save_to", default="data/preds_main.csv", type=str)
    
    args = parser.parse_args()
    
    
    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('ai-forever/ruBert-base')
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(-1)
    
    # with open(args.file) as f:
    #     lines = f.read().split('####\n')
    df = pd.read_csv(args.file)
    df['agg'] = df['question_2'] + '[SEP]' + df['question_3'] + '[SEP]' \
        + df['question_4'] + '[SEP]' + df['question_5']
    
    
    
    
    test_reviews = df[['agg']]
    
    
    model_name = 'ai-forever/ruBert-base'
    # model_name = 'bert-base-cased'
    # model_name = 'bert-large-uncased'

    dataloader_test = get_data(model_name, batch_size=4)


    keys_of_dataset = ['input_ids', 'attention_mask']



    params = {'dropout': 0.1,
             'do_activation' : True,
             'lr' : 5e-5,
             'num_epochs': 9}

    model = TextClassifier(hidden_size=128, dropout=params['dropout'], do_activation=params['do_activation'])


    checkpoint = 'checkpoint/bert_model.pt'
    model = torch.load(checkpoint, map_location=device)
    
    
    preds_rel = []
    preds_rel_prob = []
    labels_rel = []
    preds_sent = []
    preds_sent_prob = []
    labels_sent = []
    preds_obj = []
    preds_obj_prob = []
    labels_obj = []


    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader_test:
            batch = {k: batch[k].to(device) for k in keys_of_dataset}

            rel, sent, obj = model(**batch)

            rel = sigmoid(rel.detach().cpu()).numpy().flatten()
            preds_rel.extend(rel.round())
            preds_rel_prob.extend(rel)
            
            sent = sigmoid(sent.detach().cpu()).numpy().flatten()
            preds_sent.extend(sent.round())
            preds_sent_prob.extend(sent)
            
            preds_obj_prob.extend(softmax(obj).detach().cpu().numpy().tolist())
            preds_obj.extend(torch.argmax(obj, -1).detach().cpu().numpy().flatten())
            
            
    preds_aste, _ = get_aste_preds(test_reviews['agg'])
    
    # print(preds_rel)
    # print(df)
    # print(preds_aste)
    
    # print(preds_rel_prob[0])
    # print(preds_sent_prob[0])
    # print(preds_obj_prob[0])
    
    emotions = get_emotions(df['question_2'])
    
    
    preds = df[['agg', 'question_1', 'question_2', 'question_3', 'question_4', 'question_5', 'timestamp']]
    preds['rel'] = preds_rel
    preds['sent'] = preds_sent
    preds['obj'] = preds_obj
    
    preds['rel_prob'] = preds_rel_prob
    preds['sent_prob'] = preds_sent_prob
    preds['obj_prob'] = preds_obj_prob
    
    preds['triplets'] = preds_aste['pred_text']
    
    preds['emotion'] = emotions
    
    # preds = pd.DataFrame({'text': preds_aste['text'], 
    #                       'rel': preds_rel, 
    #                       'sent': preds_sent, 'obj': preds_obj,
    #                      'triplets': preds_aste['pred_text'],
    #                      'timestamp': df['timestamp']},
    #                     'rel_prob': preds_rel_prob,
    #                     'sent_prob': preds_sent_prob,
    #                     'obj_prob': preds_obj_prob,)
    preds[['rel', 'sent']] = preds[['rel', 'sent']].astype(int)
    preds.to_csv(args.save_to, index=False)
    
            
