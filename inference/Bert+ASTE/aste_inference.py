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

    review = re.sub(rf'\s+', rf' ', review)
    for punct in '.,':
        review = re.sub(rf'([\d\{punct}]) (\{punct}) ([\d\{punct}])', rf'\1\2\3', review)
    for punct in '!?':
        review = re.sub(rf'(\{punct}) (\{punct}) (\{punct})', rf'\1\2\3', review)
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



def file_to_triplets(file):

    # if file.name.rsplit('.', 1)[1].lower() == 'txt':
    with open(file) as f:
        lines = f.read().split('####\n') 

        # lines = [line.rsplit('####')[0]+'####[]' for line in lines]
    # else:
    lines = process_reviews(pd.Series(lines))



    with open('data/uploaded_file.txt', 'w') as f:
        for line in lines:
            f.write(line+'\n')
            # st.write(line[-1])
   
    # file.seek(0)
    print('Started inference')
    preds = get_preds(lines, model_path='model_58.pt', test_path='data/uploaded_file.txt')

    return preds



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="/Users/nik/Library/MobileDocuments/com~apple~TextEdit/Documents/sample_reviews.txt", type=str)
    parser.add_argument("--save_to", default="data/triplets.csv", type=str)
    
    args = parser.parse_args()
    

       
    preds = file_to_triplets(args.file)


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

    df.to_csv(args.save_to, index=False)
    preds.to_csv(args.save_to[:-4]+'_preds.csv', index=False)

# fig, ax = plt.subplots(figsize=(10, 10))
# plt.title('Распределение тональности триплетов', fontsize=25, fontweight='bold')
