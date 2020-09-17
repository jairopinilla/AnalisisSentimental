import numpy as np
import tensorflow as tf
from tensorflow import keras
#import pandas as pd
#import seaborn as sns
#from pylab import rcParams
from tqdm import tqdm
#import matplotlib.pyplot as plt
#from matplotlib import rc
#from pandas.plotting import register_matplotlib_converters
#from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import tensorflow_text 
#from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
from pathlib import Path
import json   
from googletrans import Translator


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



# Define a flask app
app = Flask(__name__)


path = Path("models/secuencial_model.h5")
print(path)

modelo = keras.models.load_model(path)
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

'''
train_reviews_pred = ['tonto saco wea, eres un wn muy gil']
for r in tqdm(train_reviews_pred):
  emb = use(r)
  review_emb = tf.reshape(emb, [-1]).numpy()

review_emb=review_emb.reshape(1,512)
y_pred = modelo.predict(review_emb)
print(y_pred)
"Bad" if np.argmax(y_pred) == 0 else "Good"
'''
#########################################
def model_predict(frase):
    train_reviews_pred = [frase]
    
    for r in tqdm(train_reviews_pred):
        emb = use(r)
        review_emb = tf.reshape(emb, [-1]).numpy()

    review_emb=review_emb.reshape(1,512)
    y_pred = modelo.predict(review_emb)

    if (np.argmax(y_pred) == 0):
        return "Negativo"
    else:
        return "Positivo"

#########################################
@app.route('/')
def index():
    # Main page
    return "se levanta el servidor"

#########################################
@app.route('/predict', methods=['GET', 'POST'])
def evaluacion():
    
    translator = Translator()
    content = request.get_json()
    frase = content['frase']    
    print('el input es: ',frase)

    retorno = model_predict(frase)
    ############################################

    translation=translator.translate(frase, dest='en',src='es')
    traduccion=translation.text

    ##########################################
    
    analyser = SentimentIntensityAnalyzer()
    scorevader = analyser.polarity_scores(traduccion)
    print(str(scorevader))

    vaderNeg = scorevader['neg']
    vaderNeu = scorevader['neu']
    vaderPos = scorevader['pos']
    vaderCom = scorevader['compound']


    dicRespeusta = {   
    
    "Sentimiento Modelo": retorno,
    "traduccion": traduccion,
    "Vader Negativo": vaderNeg,
    "Vader Neutro": vaderNeu,
    "Vader Positivo": vaderPos,
    "vader total": vaderCom
    
    }   
   
    json_object = json.dumps(dicRespeusta, indent = 4)  
    return json_object
   
#########################################
if __name__ == '__main__':
    
    app.run()