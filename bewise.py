import pandas as pd
import numpy as np
import pymorphy2
from nltk.util import ngrams
import torch
from transformers import AutoTokenizer, AutoModel


# Модель для получения эмбэддингов
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

# Функция косинусного расстояния

def get_similarity(x1, x2):
  return torch.nn.functional.cosine_similarity(x1,x2, axis=0)


# Функция для получения базовых эмбэддингов для вычисления косинусного сходства

def get_base_emb(list_name, model):
    for name in list_name:
        encoded_input = tokenizer(list_name, padding=True, truncation=True, max_length=64, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
            embeddings = model_output.pooler_output
    return embeddings


# Проверка на принадлежность к определенной части речи

def check_tags(items, check_list):
    for item in items:

        morph = pymorphy2.MorphAnalyzer()

        if any([check_param in morph.parse(item)[0].tag for check_param in check_list]):
            return item


# Вычисление косинусного сходства с базовыми эмбэддингами и проверка максимального значения

def check_max_score(items, model, tokenizer, base_embeddings, alpha=0.99):
    max_score = 0

    encoded_input = tokenizer(items, padding=True, truncation=True, max_length=64, return_tensors='pt')

    with torch.no_grad():

        model_output = model(**encoded_input)
        embeddings = model_output.pooler_output

        for embedding in embeddings:

            for base_embedding in base_embeddings:

                score = get_similarity(embedding, base_embedding)

                if score > max_score:
                    max_score = score

                if max_score > alpha:
                    return True


# Класс анализатора диалогов

class DialogAnalizer():

    def __init__(self, data_path):

        self.data = pd.read_csv(data_path)
        self.data_prep = self.prep_data()
        self.introduction, self.names = self.get_name()
        self.greetings = self.get_greeting()
        self.byes = self.get_bye()
        self.companies = self.get_company()
        self.polateness = self.get_polite()

    # Функция подготовки данных
    # Данные группируются по диалогам и оставляются только реплики менеджеров

    def prep_data(self):

        data_group = self.data.groupby(['dlg_id', 'role']).text.apply(lambda x: [i for i in x]).reset_index()
        data_prep = data_group[data_group.role == 'manager'].drop(columns='role').reset_index(drop=True)

        return data_prep

    # Функция определения реплик с приветствиями

    def get_greeting(self):

        list_dialog = {}

        for i, dialog in enumerate(self.data_prep.text.values):

            for reply in dialog[:4]:

                items = reply.split()

                list_name = ['здравствуйте', 'добрый', 'день', 'вечер', 'утро']
                base_embeddings = get_base_emb(list_name, model)

                if check_max_score(items, model, tokenizer, base_embeddings, alpha=0.99):
                    list_dialog[i] = reply

                    break

        return list_dialog

    # Функция определения реплик с прощаниями

    def get_bye(self):

        list_dialog = {}

        for i, dialog in enumerate(self.data_prep.text.values):

            for reply in dialog[-2:]:

                items = reply.split()

                list_name = ['свидания', 'доброго', 'хорошего']
                base_embeddings = get_base_emb(list_name, model)

                if check_max_score(items, model, tokenizer, base_embeddings, alpha=0.99):
                    list_dialog[i] = reply

                    break

        return list_dialog

    # Функция определения реплик, где менеджер представился, и определения имен

    def get_name(self):

        list_dialog = {}
        list_names = {}

        for i, dialog in enumerate(self.data_prep.text.values):

            flag = 0

            for reply in dialog[:3]:

                if flag == 2:
                    break

                items = reply.split()

                for item in items:

                    list_name = ['имя', 'зовут', 'я', 'это']
                    base_embeddings = get_base_emb(list_name, model)

                    if check_max_score([item], model, tokenizer, base_embeddings, alpha=0.99):
                        flag = 1

                    if flag == 1:
                        check_list = ['Name']
                        name = check_tags([item], check_list)

                        if name:
                            list_dialog[i] = reply
                            list_names[i] = name
                            flag = 2
                            break

        return list_dialog, list_names

    # Функция определения названия компании

    def get_company(self):

        list_dialog = {}

        for i, dialog in enumerate(self.data_prep.text.values):

            flag = 0

            company_name = ''

            for reply in dialog:

                items = reply.split()

                if flag == 2:
                    break

                for item in items:

                    list_name = ['компания', 'фирма', 'организация']
                    base_embeddings = get_base_emb(list_name, model)

                    if check_max_score([item], model, tokenizer, base_embeddings, alpha=0.99):

                        flag = 1

                    if flag == 1:

                        check_list = ['NOUN', 'ADJF', 'CONJ', 'ADJS', 'PREP', 'NUMR']

                        name = check_tags([item], check_list)

                        if name:
                            company_name += f' {name}'
                            continue

                        elif company_name:

                            list_dialog[i] = company_name
                            flag = 2
                            break

                        else:
                            flag = 0
                            continue

        return list_dialog

    # Функция вежливости

    def get_polite(self):

        list_dialog = {}

        for i in self.data_prep.index:

            if (i in self.greetings.keys()) & (i in self.byes.keys()):
                list_dialog[i] = True

            else:
                list_dialog[i] = False

        return list_dialog



if __name__ == "__main__":

    dan = DialogAnalizer('test_data.csv')
