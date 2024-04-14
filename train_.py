import pandas as pd
# -------------------------------------------
from striprtf.striprtf import rtf_to_text
import os
import pdfplumber
from docx import Document
import re
import joblib
# -------------------------------------------
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import spacy as sp
# -------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# -------------------------------------------
from sklearn.metrics import f1_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb


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


def read_file(file_path, label_folder=None, file_ext=None):
    try:
        if file_ext == 'rtf':
            return read_rtf(file_path)
        elif file_ext == 'pdf':
            return [label_folder, read_pdf(file_path), file_ext]
        elif file_ext in ['docx', 'doc']:
            if is_word_file(file_path):
                return [label_folder, read_docx(file_path), file_ext]
            else:
                print('Неподдерживаемый тип файла')
                pass
        else:
            print('Неподдерживаемый тип файла')
            pass
    except Exception as e:
        print(e)

def del_NER(text):
    nlp = sp.load('ru_core_news_lg')
    doc = nlp(text)
    for ent in doc.ents:
        text = text.replace(ent.text, '')

    return text

def clean(doc):
    doc = doc.lower().strip()
    doc = doc.replace('\n', ' ')
    doc = doc.replace('\r', ' ')
    doc = doc.replace('\t', ' ')
    # html tags
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

if __name__ == '__main__':

    PATH = 'data'
    folders = os.listdir(PATH)
    data = []


    for label_folder in folders:
        if os.path.isdir(os.path.join(PATH, label_folder)):
            files = os.listdir(os.path.join(PATH, label_folder))
            for file in files:
                file_path = os.path.join(PATH, label_folder, file)
                file_ext = file.split(".")[-1]
                data.append(read_file(file_path, label_folder, file_ext))

    data = pd.DataFrame(data, columns=['class', 'text', 'filename'])

    if data.isnull().sum().sum() > 0:
        data.dropna(inplace=True)

    df = pd.read_csv('data\sample.csv')

    data = pd.concat([df, data.drop(columns = 'filename')], axis=0)

    nlp = sp.load('ru_core_news_lg')

    data['text'] = data['text'].apply(lambda x: clean(del_NER(x)))

    tfidf = TfidfVectorizer()
    data_vector=tfidf.fit_transform(data['text']).toarray()
    joblib.dump(tfidf, 'tfidf.pkl')

    X_train, X_test, y_train, y_test = train_test_split(data_vector, data['class'], test_size=0.2, random_state=42)


    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_test = gnb.predict(X_test)
    gnb_f1 = f1_score(y_test, y_pred_test, average='micro')
    joblib.dump(gnb, f'models\gnb_{gnb_f1:.2}.pkl')
    print('GaussianNB done')


    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred_test = mnb.predict(X_test)
    mnb_f1 = f1_score(y_test, y_pred_test, average='micro')
    joblib.dump(mnb, f'models\mnb_{mnb_f1:.2}.pkl')
    print('MultinomialNB done')


    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    y_pred_test = lr.predict(X_test)
    lr_f1 = f1_score(y_test, y_pred_test, average='micro')
    joblib.dump(lr, f'models\lr_{lr_f1:.2}.pkl')
    print('LogisticRegression done')


    svc = LinearSVC(class_weight='balanced')
    svc.fit(X_train, y_train)
    y_pred_test = svc.predict(X_test)
    svc_f1 = f1_score(y_test, y_pred_test, average='micro')
    joblib.dump(svc, f'models\svc_{svc_f1:.2}.pkl')
    print('LinearSVC done')


    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_test = dt.predict(X_test)
    dt_f1 = f1_score(y_test, y_pred_test, average='micro')
    joblib.dump(dt, f'models\dt_{dt_f1:.2}.pkl')
    print('DecisionTree done')


    mlp = MLPClassifier(random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_test = mlp.predict(X_test)
    mlp_f1 = f1_score(y_test, y_pred_test, average='micro')
    joblib.dump(mlp, f'models\mlp_{mlp_f1:.2}.pkl')
    print('MLPClassifier done')


    classifiers = [('Decision Tree', svc),
                ('Logistic Regression', mlp),
                    ('Naive Bayes', mnb)
                ]
    vc = VotingClassifier(estimators=classifiers)
    vc.fit(X_train, y_train)
    y_pred_test = vc.predict(X_test)
    vc_f1 = f1_score(y_test, y_pred_test, average='micro')
    joblib.dump(vc, rf'models\vc_{vc_f1:.2}.pkl')
    print('VotingClassifier done')


    lgbm = lgb.LGBMClassifier(random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)
    y_pred_test = lgbm.predict(X_test)
    lgbm_f1 = f1_score(y_test, y_pred_test, average='micro')
    joblib.dump(lgbm, f'models\lgbm_{lgbm_f1:.2}.pkl')
    print('LGBMClassifier done')
