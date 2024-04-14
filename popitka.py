import joblib
from main import read_file_draft

model = joblib.load('pipelines\pipe_cl.pkl')

path = r'data\determination\1.docx'
data = read_file_draft(path)
print(model.predict(data))

# path = r'data\determination\1.docx'
# print(pipe_cl.fit_predict(x, y))
