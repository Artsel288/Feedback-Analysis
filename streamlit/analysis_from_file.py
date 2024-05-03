from llm import get_recommendation, model
import streamlit as st
import pandas as pd

# from test_script import get_preds
# from data_and_models import check_data, get_pred, convert_df, plot_shap_explanation
import numpy as np
import matplotlib.pyplot as plt

# from aux_functions import *
import os
import re
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
import seaborn as sns
import datetime

from collections import Counter
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import DBSCAN, KMeans, Birch, AgglomerativeClustering
from tqdm import tqdm


import pymorphy3
morph = pymorphy3.MorphAnalyzer()

def lemmatize(text):
    words = text.split() # разбиваем текст на слова
    res = list()
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.normal_form)

    return ' '.join(res)



def preprocess_triplets(res, triplets_columns="triplets", date_column="timestamp"):
    triplets = res[triplets_columns].values.tolist()
    dates = res[date_column].values.tolist()
    all_triplets = []
    for i, label in enumerate(triplets):
        all_triplets.extend([list(item) + [dates[i]] for item in eval(label)])

    data = []
    for triplet in all_triplets:
        at, ot, sp, ts = triplet
        at = lemmatize(at.lower())
        ot = lemmatize(ot.lower())

        data.append([ts, at, ot, sp, (at, ot, sp), (at, ot)])
    data = pd.DataFrame(data)
    data.columns = [
        "date",
        "aspect",
        "opinion",
        "sentiment",
        "triplet",
        "aspect_opinion",
    ]

    data["timestamp"] = data["date"].apply(
        lambda x: datetime.datetime.fromisoformat(x).timestamp()
    )

    return data


def page_analysis_from_file():
    preds = pd.read_csv("preds_main.csv")
    data = preds.copy()
    data["timestamp"] = data["timestamp"].replace(
        {"27.04.2024:19:00": "2024-04-27 19:00:00"}
    )
    # data
    data = preprocess_triplets(data)
    # st.write(data)

    # st.write(data)
    st.subheader("Cамые частые триплеты")

    def get_most_common_triplets(data, n):
        return [
            [as_, op, st, count]
            for (as_, op, st), count in Counter(data.triplet).most_common(n)
        ]

    most_common_triplets = get_most_common_triplets(data, 5)
    most_common_triplets = pd.DataFrame(most_common_triplets)
    most_common_triplets.columns = ["Аспект", "Мнение", "Тональность", "Частота"]
    st.write(most_common_triplets)

    def sentiment_hist(data, column):
        colors = {"POS": "green", "NEG": "red"}
        f, ax = plt.subplots(figsize=(12, 12))
        sns.countplot(x="sentiment", data=data, hue="sentiment", palette=sns.color_palette(n_colors=2), ax=ax)
        plt.title("Количество положительных и негативных триплетов")
        plt.xticks([0, 1], labels=["Негативные", "Положительные"])
        plt.xlabel("")
        plt.ylabel("Количество")
        plt.show()
        st.pyplot(f)

    st.subheader("Гистограмма тональности")
    sentiment_hist(data, "sentiment")

    def common_sentiment_bytime(data):
        data_sorted = data.sort_values("timestamp", ascending=True)

        def count_pos_neg(group):
            sent_dict = {"count_pos": 0, "count_neg": 0}
            for item in group:
                if item == "POS":
                    sent_dict["count_pos"] += 1
                else:
                    sent_dict["count_neg"] += 1
            return sent_dict

        counted_sentiment = (
            data_sorted.groupby("date")[["sentiment"]]
            .agg(count_pos_neg)
            .reset_index(names=["date"])
        )

        count_pos = 1
        count_neg = 1
        ratio = []
        for i, item in counted_sentiment.iterrows():
            count_pos += item["sentiment"]["count_pos"]
            count_neg += item["sentiment"]["count_neg"]
            # ratio.append(count_pos / count_neg )
            ratio.append(100 * count_pos / (count_neg + count_pos))

        smoothing_k = 2
        data_ratio = counted_sentiment[["date"]].iloc[::smoothing_k]
        data_ratio["Доля положительных"] = ratio[::smoothing_k]

        fig = px.line(data_ratio, x="date", y="Доля положительных")
        fig.update_yaxes(range=[min(ratio) - 1, max(ratio) + 1], autorange=False)
        #   fig.get_axes().set_ylim(min(ratio)-1, max(ratio)+1)
        # fig.update_layout(scattermode="group")
        #   fig.yaxis_title('Доля положительных триплетов')
        #   fig.xaxis_title('Дата')
        fig.update_layout(
            title=f"Доля положительных триплетов по времени",
            xaxis_title="Дата",
            yaxis_title="Доля положительных триплетов",
        )
        # fig.show()
        st.plotly_chart(fig)

        
    st.subheader("Отношение тональностей с течением времени")
    common_sentiment_bytime(data)
    
    
    

    # сюда вставить аспект

    def aspect_sentiment_bytime(data, aspect):
        aspect_data = data.groupby("aspect").get_group(aspect).sort_values("timestamp")

        def count_pos_neg(group):
            sent_dict = {"count_pos": 0, "count_neg": 0}
            for item in group:
                if item == "POS":
                    sent_dict["count_pos"] += 1
                else:
                    sent_dict["count_neg"] += 1
            return sent_dict

        counted_sentiment = (
            aspect_data.groupby("date")[["sentiment"]]
            .agg(count_pos_neg)
            .reset_index(names=["date"])
        )

        count_pos = 1
        count_neg = 1
        ratio = []
        for i, item in counted_sentiment.iterrows():
            count_pos += item["sentiment"]["count_pos"]
            count_neg += item["sentiment"]["count_neg"]
            ratio.append(100 * count_pos / (count_neg + count_pos))

        smoothing_k = 1
        data_ratio = counted_sentiment[["date"]].iloc[::smoothing_k]
        data_ratio["Доля положительных"] = ratio[::smoothing_k]

        fig = px.line(data_ratio, x="date", y="Доля положительных")
        fig.update_yaxes(range=[min(ratio) - 1, max(ratio) + 1], autorange=False)
        fig.update_layout(
            title=f'Зависимость доли положительных триплетов, Аспект - "{aspect}"',
            xaxis_title="Дата",
            yaxis_title="Доля положительных триплетов",
        )
        # fig.update_layout(scattermode="group")
        # fig.show()
        st.plotly_chart(fig)
    
    st.subheader("Тональность по аспектам с течением времени")
    possible_aspect = data['aspect'].value_counts()
    possible_aspect = possible_aspect[possible_aspect>=3]
    aspect = st.selectbox('Выберите аспект', np.unique(possible_aspect.index), index=0)

    # aspect_sentiment_bytime(data, "изучение")
    aspect_sentiment_bytime(data, aspect)
    
    # 1/0

    def get_most_neg_aspects(data, n):
        fig, ax = plt.subplots(layout="constrained")
        result = (
            data[data.sentiment == "NEG"]
            .groupby("aspect")
            .size()
            .sort_values(ascending=False)
            .iloc[:n]
        )
        sns.barplot(x=result.index, y=result.values)
        ax.set_xlabel("Аспект")
        ax.set_ylabel("Кол-во положительных триплетов")
        ax.set_title("Cамые негативные аспекты")
        plt.xticks(fontsize=12, rotation=30)
        plt.yticks(list(range(max(result.values) + 1)), fontsize=12)
        st.pyplot(fig)

    st.subheader("Гистограмма самых негативных аспектов")
    get_most_neg_aspects(data, 10)

    def get_most_pos_aspects(data, n):
        fig, ax = plt.subplots(layout="constrained")
        result = (
            data[data.sentiment == "POS"]
            .groupby("aspect")
            .size()
            .sort_values(ascending=False)
            .iloc[:10]
        )
        sns.barplot(x=result.index, y=result.values)
        plt.xticks(fontsize=10, rotation=30)
        ax.set_xlabel("Аспект")
        ax.set_ylabel("Кол-во положительных триплетов")
        ax.set_title("Самые позитивные аспекты")
        st.pyplot(fig)

    st.subheader("Гистограмма самых положительных аспектов")
    get_most_pos_aspects(data, 10)

    def get_most_popular_aspect_opinion(data, n="all"):
        fig, ax = plt.subplots()
        data["aspect-opinion"] = data["aspect_opinion"].apply(
            lambda x: x[0] + "-" + x[1]
        )
        most_common_pairs = pd.DataFrame(
            Counter(data["aspect-opinion"]).most_common(10), columns=["pair", "count"]
        )
        most_common_pairs_series = pd.Series(
            list(most_common_pairs["count"]), index=list(most_common_pairs["pair"])
        )[:10]
        fig, ax = plt.subplots(layout="constrained")
        #   ax.bar(x=most_common_pairs_series.index, height=most_common_pairs_series.values)
        sns.barplot(
            x=most_common_pairs_series.index, y=most_common_pairs_series.values, ax=ax
        )
        plt.xticks(rotation=75)
        ax.set_xlabel("Аспект-мнение")
        ax.set_ylabel("Встречаемость")
        ax.set_title("Cамые частые пары аспект-мнение")
        st.pyplot(fig)

    st.subheader("Гистограмма популярности пар аспект-мнение")
    get_most_popular_aspect_opinion(data, 12)

    def simple_wordcloud_aspect(data):
        fig, ax = plt.subplots()
        cloud_aspect = WordCloud(
            background_color="#ffffff",
            contour_width=20,
            contour_color="#2e3043",
            colormap="Set2",
            max_words=20,
        ).generate(text=" ".join(data["aspect"]))
        plt.imshow(cloud_aspect)
        plt.axis("off")
        plt.show()
        st.pyplot(fig)

    st.subheader("Облака слов для аспектов и мнений")
    simple_wordcloud_aspect(data)

    def simple_wordcloud_opinion(data):
        fig, ax = plt.subplots()
        cloud_opinion = WordCloud(background_color="#ffffff", max_words=20).generate(
            text=" ".join(data["opinion"])
        )
        plt.imshow(cloud_opinion)
        plt.axis("off")
        plt.show()
        st.pyplot(fig)

    simple_wordcloud_opinion(data)

    from transformers import pipeline

    def plot_emotions(data):
        eng_to_rus = {
            "admiration": "восхищение",
            "amusement": "веселье",
            "anger": "злость",
            "annoyance": "раздражение",
            "approval": "одобрение",
            "caring": "забота",
            "confusion": "непонимание",
            "curiosity": "любопытство",
            "desire": "желание",
            "disappointment": "разочарование",
            "disapproval": "неодобрение",
            "disgust": "отвращение",
            "embarrassment": "смущение",
            "excitement": "возбуждение",
            "fear": "страх",
            "gratitude": "признательность",
            "grief": "горе",
            "joy": "радость",
            "love": "любовь",
            "nervousness": "нервозность",
            "optimism": "оптимизм",
            "pride": "гордость",
            "realization": "осознание",
            "relief": "облегчение",
            "remorse": "раскаяние",
            "sadness": "грусть",
            "surprise": "удивление",
            "neutral": "нейтральность",
        }

        fig, ax = plt.subplots(figsize=(9, 9))
        plt.title("Эмоции в отзывах студентов")
        plt.ylabel("Встречаемость эмоции")
        plt.xlabel("Эмоция")
        sns.countplot(x=data["emotion"])
        plt.xticks(
            list(range(len(data["emotion"].value_counts().index))),
            labels=[eng_to_rus[i] for i in data["emotion"].value_counts().index],
            fontsize=12,
            rotation=30,
        )
        st.pyplot(fig)

    st.subheader("Эмоции студентов")
    plot_emotions(pd.read_csv("preds_main.csv"))

    import umap

    def cluster_aspects(data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            "cointegrated/rubert-tiny-sentiment-balanced"
        )
        model = AutoModel.from_pretrained("cointegrated/rubert-tiny-sentiment-balanced")

        model.to(device)

        aspects = set(data.aspect.tolist())
        embeddings = []

        for aspect in tqdm(aspects):

            encoded_input = tokenizer(aspect, padding=False, return_tensors="pt").to(
                device
            )

            with torch.no_grad():
                model_output = model(**encoded_input)

            embeddings.append(model_output.last_hidden_state.mean(dim=1))

        embeddings = torch.stack(embeddings).cpu().numpy().squeeze(1)

        reducer8 = umap.UMAP(n_components=4)
        reducer2 = umap.UMAP(n_components=2)

        embeds_reduced8 = reducer8.fit_transform(embeddings)
        embeds_reduced2 = reducer2.fit_transform(embeddings)

        clusters = DBSCAN(n_jobs=-1, min_samples=1).fit(embeds_reduced8)
        df = pd.DataFrame(embeds_reduced2[:, 0], columns=["x"])
        df["y"] = embeds_reduced2[:, 1]
        df["color"] = clusters.labels_
        df["aspect"] = list(aspects)
        # df["size"] = df["aspect"].apply(lambda x: len(data[data.aspect == x]) / 10)

        return px.scatter(
            df, x="x", y="y", color=df["color"], hover_data={"text": df["aspect"]},
            color_continuous_scale=[
        '#A3ADF8',
        '#7D88FA',
        
    ],
        )

    fig = cluster_aspects(data)
    st.subheader("Кластеризация аспектов")
    st.plotly_chart(fig)

    import umap

    def cluster_opinions(data):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            "cointegrated/rubert-tiny-sentiment-balanced"
        )
        model = AutoModel.from_pretrained("cointegrated/rubert-tiny-sentiment-balanced")

        model.to(device)

        opinions = set(data.opinion.tolist())
        embeddings = []

        for opinion in tqdm(opinions):

            encoded_input = tokenizer(opinion, padding=False, return_tensors="pt").to(
                device
            )

            with torch.no_grad():
                model_output = model(**encoded_input)

            embeddings.append(model_output.last_hidden_state.mean(dim=1))

        embeddings = torch.stack(embeddings).cpu().numpy().squeeze(1)

        reducer8 = umap.UMAP(n_components=8)
        reducer2 = umap.UMAP(n_components=2)

        embeds_reduced8 = reducer8.fit_transform(embeddings)
        embeds_reduced2 = reducer2.fit_transform(embeddings)

        clusters = DBSCAN(n_jobs=-1, min_samples=4).fit(embeds_reduced8)
        df = pd.DataFrame(embeds_reduced2[:, 0], columns=["x"])
        df["y"] = embeds_reduced2[:, 1]
        df["color"] = clusters.labels_
        df["opinion"] = list(opinions)
        # df["size"] = df["aspect"].apply(lambda x: len(data[data.aspect == x]) / 10)

        return px.scatter(
            df, x="x", y="y", color=df["color"], hover_data={"text": df["opinion"]},
            color_continuous_scale=[
        '#A3ADF8',
        '#7D88FA',
        
    ],
        )
    fig = cluster_opinions(data)
    st.plotly_chart(fig)

    
    
    
    
    st.subheader("Основные объекты отзыва")
    # colors = {1: 'green', 0: 'red'}
    fig, ax = plt.subplots()
    sns.countplot(x="obj", data=preds, ax=ax)
    plt.title('Основной объект отзыва')
    # plt.xlabel
    plt.xticks([0, 1, 2], labels=['Вебинар', 'Программа', 'Преподаватель'])
    plt.xlabel('')
    plt.ylabel('Количество')
    st.pyplot(fig)
    # plt.show()
    

    st.subheader("Релевантные и нерелевантные отзывы")
    colors = {1: 'green', 0: 'red'}
    fig, ax = plt.subplots()
    sns.countplot(x="rel", data=preds, palette=sns.color_palette(n_colors=2), ax=ax)
    plt.title('Количество релевантных и нерелевантных отзывов')
    # plt.xlabel
    plt.xticks([0, 1], labels=['Нерелевантные', 'Релевантные'])
    plt.xlabel('')
    plt.ylabel('Количество')
    st.pyplot(fig)
    # plt.show()

    st.subheader("Положительные и негативные отзывы")
    colors = {1: 'green', 0: 'red'}
    fig, ax = plt.subplots()
    sns.countplot(x="sent", data=preds, palette=sns.color_palette(n_colors=2), ax=ax)
    plt.title('Количество положительных и негативных отзывов')
    # plt.xlabel
    plt.xticks([0, 1], labels=['Негативные', 'Положительные'])
    # plt.xlabel('Тональность')
    plt.xlabel('')
    plt.ylabel('Количество')
    st.pyplot(fig)
    # plt.show()

    st.subheader("Скачать статистику в формате XLSX")
    
    st.download_button(
        label="Download",
        data=open("feedback.xlsx", "rb").read(),
        file_name="feedback.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
    with st.sidebar:
        messages = st.container(height=300)
        if prompt := st.chat_input("Say something"):
            messages.chat_message("user").write(prompt)
            messages.chat_message("assistant").write(f"Echo: {get_recommendation(prompt, 0, model)}")

    
    # st.write("The end")
    


#     ALLOWED_EXTENSIONS = set(['txt']) # TODO: add csv

#     file = st.file_uploader("Загрузите файл с данными, которые хотите проанализировать",
#                             type=ALLOWED_EXTENSIONS)

#     preds_hist = pd.read_csv('data/preds_hist.csv')

#     if '.' in file.name and file.name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
#             st.success(f"Файл {file.name} успешно загружен!")

#             preds = file_to_triplets(file)
#     print(preds)

#     triplets = preds['pred_text'].values.tolist()
#     all_triplets = []
#     for label in triplets:
#         # all_triplets.extend(eval(label))
#         all_triplets.extend(label)

#     data = []
#     for triplet in all_triplets:
#         at, ot, sp = triplet
#         at = at.lower()
#         ot = ot.lower()

#         data.append([at, ot, sp, (at, ot, sp), (at, ot)])
#     df = pd.DataFrame(data)
#     df.columns = ['aspect', 'opinion', 'sentiment', 'triplet', 'aspect_opinion']

#     fig, ax = plt.subplots(figsize=(10, 10))
#     plt.title('Распределение тональности триплетов', fontsize=25, fontweight='bold')
#     df['sentiment'].hist()
#     st.pyplot(fig)


# corr_y1 = corr1['y'].dropna()
# corr_y1 = corr_y1.sort_values(key=lambda x: abs(x))[:-1]
# plt.barh(corr_y1.index, abs(corr_y1))
# plt.title('Корреляция с целевой переменной', fontsize=25, fontweight='bold')
# ax.bar_label(ax.containers[0], labels=corr_y1.apply(lambda x: round(x, 3)), fmt='%.3f')
# st.pyplot(fig)
