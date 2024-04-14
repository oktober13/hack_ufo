import joblib
from main import read_file_draft, vectorize, del_NER, clean
import os

# попытка загрузить модель 
model = joblib.load('pipelines\pipe_cl.pkl')

# и сделать предсказания на одном из наших доков
# path = r'data\determination\1.docx'
# data = read_file_draft(path)
# print(model.predict(data))
# тут работает все идеально
# ['determination']


for doc in os.listdir('data\statute'):
    print(doc)
    # главное передавать в функцию драфта полный путь
    doc = read_file_draft('data\statute\\' + doc)
    print(model.predict(doc))

# максимальное время работы на  док - 6 секунд
# всё считал идеально

# 1.docx
# ['contract offer']
# 10.docx
# ['contract offer']
# 2.docx
# ['contract offer']
# 3.docx
# ['contract offer']
# 4.docx
# ['contract offer']
# 5.docx
# ['contract offer']
# 6.docx
# ['contract offer']
# 7.docx
# ['contract offer']
# 8.docx
# ['contract offer']
# 9.docx
# ['contract offer']
# Dogovor_oferta-po-p.2-_1_-_1_.docx
# ['contract offer']
# dogovor_stroitelnogo_podryada.docx
# ['contract offer']
# oferta.rtf
# ['data\\contract offer']
# public_offer.docx
# ['contract offer']


# 1.docx
# ['statute']
# 2.docx
# ['statute']
# 3.docx
# ['statute']
# 3904.docx
# ['statute']
# 4.docx
# ['statute']
# modelnyyustavdorabms07082017-1.docx
# ['statute']
# OBRAZEC_ODIN.docx
# ['statute']
# statue_simple.docx
# ['statute']
# tipovoi-ustav-1.docx
# ['statute']
# tipovoi-ustav-2.docx
# ['statute']
# tipovoi-ustav-3.docx
# ['statute']
# Ustav 16.docx
# ['statute']
# ustav-ooo-dva-uchreditelya-obrazec-2017.docx
# ['statute']
# ustav-ooo1.docx
# ['statute']
# Ustav.docx
# ['statute']
# УСТАВ-ОБРАЗЕЦ-Ассоциации.docx
# ['statute']
# УСТАВ-ОБРАЗЕЦ-Фонд-собрание-учредителей.docx
# ['statute']