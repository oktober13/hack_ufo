import joblib
import pandas as pd
import pdfplumber

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer

from train_ import read_pdf, read_docx, read_rtf, is_word_file, read_file, del_NER, clean
from striprtf.striprtf import rtf_to_text


def read_rtf_main(file_path):
    with open(file_path, 'rb') as f:
        content = f.read().decode('utf-8', errors='ignore')
        text = rtf_to_text(content)
        return [text]


def read_file_draft(file_path):
    file_name = file_path.split('\\')[-1]
    file_ext = file_path.split('.')[-1]
    try:
        if file_ext == 'rtf':
            return read_rtf_main(file_path)[0]  # + [file_name]
        elif file_ext == 'pdf':
            print('pdf')
            return [read_pdf(file_path)][0]  #, file_name]
        elif file_ext in ['docx', 'doc']:
            if is_word_file(file_path):
                return [read_docx(file_path)][0]  # , file_name]
            else:
                print('Неподдерживаемый тип файла')
                pass
        else:
            print('Неподдерживаемый тип файла')
            pass
    except Exception as e:
        print(e)


def vectorize(text):
    tfidf = joblib.load('tfidf.pkl')
    return [tfidf.transform(text).toarray()]


def main():
    svc_model = joblib.load('models\svc_0.98.pkl')
    mlp_model = joblib.load('models\mlp_0.97.pkl')
    mnb_model = joblib.load('models\mnb_0.8.pkl')

    # tfidf = joblib.load('tfidf.pkl')

    # reading_doc = Pipeline(steps=[
    #     ('read', read_file_draft),
    # ])

    pipe_cl = Pipeline(steps=[
        # ('reading', FunctionTransformer(read_file_draft)),
        ('del_NER', FunctionTransformer(del_NER)),
        ('cleaning', FunctionTransformer(clean)),
        ('vectoring', FunctionTransformer(vectorize)),
        ('classifier', svc_model)
    ])

    pipe_no_cl = Pipeline(steps=[
        # ('reading', reading_doc),
        ('del_NER', FunctionTransformer(del_NER)),
        ('vectoring', FunctionTransformer(vectorize)),
        ('classifier', svc_model)
    ])

    # ---------------------------------------------------------------------
    data = pd.read_csv('data\data_clean.csv')

    x = data['text']
    y = data['class']
    pipe_cl.fit(x, y)
    # ---------------------------------------------------------------------
    joblib.dump(pipe_cl, 'pipelines\pipe_cl.pkl')
    joblib.dump(pipe_no_cl, 'pipelines\pipe_no_cl.pkl')
    # ---------------------------------------------------------------------

    # data = read_file_draft(path)
    # print(pipe_cl.predict(data))

    # pipe_cl_vote = Pipeline(steps=[
    #     ('reading', reading_doc),
    #     ('del_NER', del_NER),
    #     ('cleaning', clean),
    #     ('vectoring', tfidf),
    #     ('classifier', svc_model)
    # ])

    # pipe_no_cl_vote = Pipeline(steps=[
    #     ('reading', reading_doc),
    #     ('del_NER', del_NER),
    #     ('vectoring', tfidf),
    #     ('classifier', svc_model)
    # ])


if __name__ == '__main__':
    main()
