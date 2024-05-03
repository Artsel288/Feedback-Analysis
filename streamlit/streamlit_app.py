# from data_and_models import *
from analysis_from_file import page_analysis_from_file
# from preds_by_hand import page_prediction
# from cats_info import get_similar_categories
# from business import page_business_info
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# from matplotlib.ticker import FuncFormatter
# import matplotlib.ticker as ticker
# from data_and_models import check_data

def main():
    page_analysis_from_file()
    
#     st.sidebar.title("Навигация")
#     pages = ['Предсказание и анализ для данных из файла', 'INFO', 'Предсказание по данным, введённым вручную', 'EDA',
#          'Информация об ID категориальных признаков']
#     selected_page = st.sidebar.selectbox(
#         'Доступные страницы',
#         (pages))

#     if selected_page == pages[0]:
#         page_analysis_from_file()
#     elif selected_page == "Предсказание по данным из файла":
#         print(1)
#     elif selected_page == "EDA":
#         print(1) 
#     elif selected_page == "Информация об ID категориальных признаков":
#         print(1) 
#     elif selected_page == "INFO":
#         print(1) 
    
if __name__ == '__main__':
    main()
