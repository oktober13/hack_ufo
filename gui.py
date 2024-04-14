import asyncio
import os.path
import random
import time
import csv

from main import read_file_draft, vectorize, del_NER, clean
# либы неявно вызываются в пайплайне, поэтому нужны тут
# иначе ошибка
import os

import uvicorn
import aiofiles
import joblib

from fastapi import FastAPI, Request, UploadFile, File, Form

from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from typing import List, Dict

from uuid import uuid4

ru_labels = {
    "proxy" : 'доверенность',
    'contract' : 'договор',
    'act' : 'акт',
    'application' : 'заявление',
    'order' : 'приказ',
    'invoice' : 'счет',
    'bill' : 'приложение',
    'arrangement' : 'соглашение',
    'contract offer' : 'договор оферты',
    'statute' : 'устав',
    'determination' : 'решение'
}

model = joblib.load('pipelines\pipe_cl.pkl')

app = FastAPI()
TMP_UPLOADS_DIRECTORY = 'uploads'

# app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# model = joblib.load('pipe_cl.pkl')

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "id": id})


@app.post("/upload", response_class=HTMLResponse)
async def upload_docs(
    request: Request,
    name: str = Form(...),
    surname: str = Form(...),
    docs: List[UploadFile] = File(...)
):
    # Формируем имя папки пользователя
    user_directory = os.path.join('uploads', f"{name}_{surname}")
    
    # Создаем папку пользователя, если она не существует
    os.makedirs(user_directory, exist_ok=True)
    os.makedirs(user_directory + '/buffer', exist_ok=True)
    
    docs_list = []
    predictions = []

    for doc in docs:
        file_ext = doc.filename.split('.')[-1]
        # Генерируем уникальное имя файла
        filename = ''
        if file_ext == 'rtf':
            filename = str(uuid4()) + '.rtf'
        elif file_ext.lower() == 'pdf':
            filename = str(uuid4()) + '.pdf'
        elif file_ext in ['docx', 'doc']:
            filename = str(uuid4()) + '.docx'
        # Генерируем уникальное имя файла
        
        # Путь к файлу в папке пользователя
        file_path = os.path.join(user_directory + r'\buffer', filename)
        
        docs_list.append(file_path)
        
        # Сохраняем файл в папку пользователя
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await doc.read()
            await out_file.write(content)
            
            
        doc = read_file_draft(file_path)
        file_name = os.path.basename(file_path)
        print(file_name)
        prediction = ru_labels[model.predict(doc)[0]]
        next_file_path = user_directory + f'/{prediction}'
        os.makedirs(next_file_path, exist_ok=True)
        os.rename(file_path, next_file_path + f'/{file_name}')
        predictions.append(prediction)

    print(predictions)
    # Формируем сообщение об успешной загрузке
    success_message = f"Файлы успешно загружены: {', '.join(docs_list)}"
    pred_message = f"Типы загруженных файлов: {', '.join(predictions)}"
    
    # Перенаправляем пользователя на начальный экран с сообщением об успешной загрузке
    return templates.TemplateResponse("index.html", {"request": request, "message": {
        'success' : success_message,
        "prediction" : pred_message
    }})


if __name__ == '__main__':
    uvicorn.run(app, port=3000)