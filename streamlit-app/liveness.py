import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import cv2
import tensorflow as tf
import io
import os
from os import path

"""
# Computer Vision
## Liveness Detection

O detector de Liveness (Vivacidade) tem por objetivo estabelecer um índice que atesta o quão 
confiável é a imagem obtida pela câmera.
Imagens estáticas, provindas de fotos manipuladas, são os principais focos de fraude neste tipo de validação.
Um modelo de classificação deve ser capaz de ler uma imagem da webcam, classificá-la como (live ou não) e 
exibir sua probabilidade da classe de predição.

"""

IMAGE_SIZE = [192, 192]
if (not str(path.exists('streamlit-app//model.h5'))):
    st.error('Modelo não encontrado! Tente novamente mais tarde.')
    st.stop()

model = tf.keras.models.load_model('streamlit-app//model.h5')

uploaded_file = st.file_uploader('Tente uma outra imagem', type=["png", "jpg"])
if uploaded_file is not None:
    img_stream = io.BytesIO(uploaded_file.getvalue())
    imagem = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
    st.image(imagem, channels="BGR")


camera = st.camera_input(
    "Tire sua foto", help="Lembre-se de permitir ao seu navegador o acesso a sua câmera.")

if camera is not None:
    bytes_data = camera.getvalue()
    imagem = cv2.imdecode(np.frombuffer(
        bytes_data, np.uint8), cv2.IMREAD_COLOR)


if camera or uploaded_file:

    with st.spinner('Classificando imagem...'):
        imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

        classificador_face = cv2.CascadeClassifier(
            'streamlit-app//haarcascade_frontalface_default.xml')
        faces = classificador_face.detectMultiScale(imagem_gray, 1.3, 3)

        imagem_anot = imagem.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(imagem_anot, (x, y), (x+w, y+h), (0, 0, 255), 2)

        st.image(imagem_anot, channels="BGR")

        if len(faces) == 0:
            st.error('Não foi possível detectar uma face!')
        else:
            for face in faces:
                (x, y, w, h) = face
                face = imagem_gray[y:y+h, x:x+w]
                face = cv2.resize(face, IMAGE_SIZE,
                                  interpolation=cv2.INTER_LANCZOS4)
                face = face.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
                prob = model.predict(face)[0][0]
                if prob > 0.5:
                    st.success('Imagem com Vivacidade, probabilidade de {}%!'.format(
                        round(prob * 100, 2)))
                else:
                    st.error('Imagem sem Vivacidade, probabilidade de {}%!'.format(
                        round(prob * 100, 2)))
