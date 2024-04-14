import joblib
from main import read_file_draft
# попытка загрузить модель 
model = joblib.load('pipelines\pipe_cl.pkl')

# и сделать предсказания на одном из наших доков
path = r'data\determination\1.docx'
data = read_file_draft(path)
# print(data)
print(model.predict(data))


# выдает оишбку

# Traceback (most recent call last):
#   File "d:\VSCode\Hack\popitka.py", line 10, in <module>
#     print(model.predict(data))
#   File "D:\VSCode\Hack\hack_\lib\site-packages\sklearn\pipeline.py", line 602, in predict
#     Xt = transform.transform(Xt)
# AttributeError: 'function' object has no attribute 'transform'
