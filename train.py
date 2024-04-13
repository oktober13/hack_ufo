import pandas as pd
import numpy as np
# -------------------------------------------
import os
from striprtf.striprtf import rtf_to_text
import pdfplumber
from docx import Document
import docx2txt
import textract
from docx2python import docx2python
# -------------------------------------------
import re
import nltk
nltk.download("stopwords")
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
# -------------------------------------------
import spacy
import spacy as sp
from spacy.tokens.doc import Doc
!python -m spacy download ru_core_news_lg -q
# -------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
# -------------------------------------------


def read_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text.strip()


def read_docx(file_path):
    doc = Document(file_path)
    paragraphs = []
    for para in doc.paragraphs:
        paragraphs.append(para.text)
    return '\n'.join(paragraphs)


def read_rtf(path):
    with open(path, 'rb') as f:
        content = f.read().decode('utf-8', errors='ignore')
        text = rtf_to_text(content)
        return [os.path.dirname(path).split("/")[-1], text, os.path.basename(path).split(".")[-1:]]


def is_word_file(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
        if content.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):  # Магическое число для файлов DOC
            return True
        elif content.startswith(b'PK\x03\x04'):  # Магическое число для файлов DOCX
            return True
        else:
            return False


def read_file(file_path, label_folder, file_ext):
    try:
        if file_ext == 'rtf':
            return read_rtf(file_path)
        elif file_ext == 'pdf':
            return [label_folder, read_pdf(file_path), file_ext]
        elif file_ext in ['docx', 'doc']:
            if is_word_file(file_path):
                return [label_folder, read_docx(file_path), file_ext]
            else:
                print(f'Файл {file_path} не является поддерживаемым форматом Word')
                bugs.append(file_path)
        else:
            print(f'Файл {file_path} не является поддерживаемым форматом')
            bugs.append(file_path)
    except Exception as e:
        print(f'Файл {file} не был прочитан. Путь: {file_path}')
        print(e)
        bugs.append(file_path)


def del_NER(text):

    doc = nlp(text)
    for ent in doc.ents:
        text = text.replace(ent.text, '')
    return text


def clean(doc):

    doc = doc.lower().strip()
    doc = doc.replace('\\[a-z]', ' ')
    # # html tags
    doc = re.sub(r'<.*?>', '', doc)
    # Замена адресов электронной почты на "почта"
    doc = re.sub(r'[\w\.-]+@[\w\.-]+', 'почта', doc)
    # Замена упоминаний на "тэг"
    doc = re.sub(r'@[\w]+', 'тэг', doc)
    # Замена URL-адреса любого вида на "сайт"
    doc = re.sub(r'(https?://)?(www\.)?[\w\.-]+\.[a-zа-я]+', 'сайт', doc)
    # # Замена дат на "дата"
    doc = re.sub(r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})|(\d{2,4}[./-]\d{1,2}[./-]\d{1,2})", "дата", doc)
    # Удаление знаков препинания и цифр
    doc = re.sub('[^а-яА-Я0-9nN] | ([nN]\b)', ' ', doc)
    doc = re.sub(r'\s+', ' ', doc)

    stop_words = set(stopwords.words('russian'))
    doc = word_tokenize(doc)
    # Фильтрация стоп-слов
    doc = [word for word in doc if word not in stop_words]

    # Стемминг
    stemmer = SnowballStemmer("russian")
    doc = ' '.join([stemmer.stem(token) for token in doc])

    return doc

    # # Лемматизация
    # morph = pymorphy2.MorphAnalyzer()
    # lemmatized = [morph.parse(word)[0].normal_form for word in filtered_tokens]

def preproc(text):
    return clean(del_NER(text))


if __name__ == "main":
    
    PATH = './data'
    folders = os.listdir(PATH)
    data = []
    bugs = []

    for label_folder in folders:
        if os.path.isdir(os.path.join(PATH, label_folder)):
            files = os.listdir(os.path.join(PATH, label_folder))
            for file in files:
                file_path = os.path.join(PATH, label_folder, file)
                file_ext = file.split(".")[-1]
                data.append(read_file(file_path, label_folder, file_ext))

    data = pd.DataFrame(data, columns=['class', 'text', 'filename'])
    
    if data.isnull().sum() > 0:
        data.dropna(inplace=True)

    df = pd.read_csv('/content/drive/MyDrive/Хакатон/train_dataset_dataset/data/sample.csv')
    data_n = pd.concat([df, data.drop(columns = 'filename')], axis=0)
    
    nlp = sp.load('ru_core_news_lg')

    data['text'] = data['text'].apply(lambda x: clean(del_NER(x)))

    tfidf = TfidfVectorizer()
    data_vector=tfidf.fit_transform(data['text']).toarray()

    X_train, X_test, y_train, y_test = train_test_split(data_vector, data['class'], test_size=0.2, random_state=42)
    

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred_test = gnb.predict(X_test)
    gnb_f1 = f1_score(y_test, y_pred_test, average='micro')


    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    y_pred_test = lr.predict(X_test)
    lr_f1 = f1_score(y_test, y_pred_test, average='micro')


    svc = LinearSVC(class_weight='balanced')
    svc.fit(X_train, y_train)

    y_pred_test = svc.predict(X_test)
    svc_f1 = f1_score(y_test, y_pred_test, average='micro')


    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    y_pred_test = dt.predict(X_test)
    dt_f1 = f1_score(y_test, y_pred_test, average='micro')


    classifiers = [('Decision Tree', dt),
                ('Logistic Regression', lr),
                    ('Naive Bayes', gnb)
                ]
    vc = VotingClassifier(estimators=classifiers)
    vc.fit(X_train, y_train)
    y_pred_test = vc.predict(X_test)
    vc_f1 = f1_score(y_test, y_pred_test, average='micro')


    lgbm = lgb.LGBMClassifier(random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)

    y_pred_test = lgbm.predict(X_test)
    lgbm_f1 = f1_score(y_test, y_pred_test, average='micro'))


    pass