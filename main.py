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

from train_ import read_pdf, read_docx, read_rtf, is_word_file, read_file, del_NER, clean
from striprtf.striprtf import rtf_to_text


def read_rtf_main(file_path):
    with open(file_path, 'rb') as f:
        content = f.read().decode('utf-8', errors='ignore')
        text = rtf_to_text(content)
        return [text]


def read_file_draft(file_path):
    file_path = str(file_path)
    file_name = file_path.split('\\')[-1]
    file_ext = file_path.split('.')[-1]
    try:
        if file_ext == 'rtf':
            return [read_rtf_main(file_path) + [file_name]]
        elif file_ext == 'pdf':
            return [read_pdf(file_path), file_ext]
        elif file_ext in ['docx', 'doc']:
            if is_word_file(file_path):
                return [read_docx(file_path), file_ext]
            else:
                print('Неподдерживаемый тип файла')
                pass
        else:
            print('Неподдерживаемый тип файла')
            pass
    except Exception as e:
        print(e)

def main():
    svc_model = joblib.load('models\svc_0.98.pkl')
    mlp_model = joblib.load('models\mlp_0.97.pkl')
    mnb_model = joblib.load('models\mnb_0.8.pkl')


    tfidf = joblib.load('tfidf.pkl')
    path = 'D:\VSCode\Hack\data\contract offer\oferta.rtf'
    t = read_file_draft(path)
    print(t)
    # reading_doc = Pipeline(steps=[
    #     ('read', read_file()),
    # ])

#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('encoder', OneHotEncoder(handle_unknown='ignore'))
#     ])

#     preprocessor = ColumnTransformer(transformers=[
#         ('numerical', numerical_transformer, numerical_features),
#         ('categorical', categorical_transformer, categorical_features)
#     ])

#     models = (
#         LogisticRegression(solver='liblinear'),
#         RandomForestClassifier(),
#         MLPClassifier(activation='logistic', hidden_layer_sizes=(256, 128, 64))
#     )

#     best_score = .0
#     best_pipe = None
#     for model in models:
#         pipe = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('classifier', model)
#         ])
#         score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
#         print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

#         if score.mean() > best_score:
#             best_score = score.mean()
#             best_pipe = pipe

#     print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
#     joblib.dump(best_pipe, 'loan_pipe.pkl')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


