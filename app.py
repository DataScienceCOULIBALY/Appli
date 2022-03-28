import numpy as np
import pandas as pd
import re
import streamlit as st
#from transformers import pipeline
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preProcess_data(text): #cleaning the data    
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text

data = pd.read_csv('train.csv')
tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(data['message'].values)

def my_pipeline(text): #pipeline
  text_new = preProcess_data(text)
  X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
  X = pad_sequences(X, maxlen=28)
  return X



st.title('Application d\'analyse de sentiment en Bambara-Francais')
st.write('Cette application repond au besoin d\'analyser les opinions du citoyen lambda s\'exprimant en langues burkinabe et africaines; la communication communautaire se deroulant en plusieurs langues.')
st.write(
    'Lien de notre notebook d\'analyse: https://github.com/DataScienceCOULIBALY/Analyse-de-sentiment-.git')


form = st.form(key='sentiment-form')
user_input = form.text_area('Enter votre texte')
submit = form.form_submit_button('Soumettre')

if submit:
    def predict(user_input):
        clean_text = my_pipeline(user_input) #cleaning and preprocessing of the texts
        loaded_model = load_model('LSTM_Mono.h5') #loading the saved model
        predictions = loaded_model.predict(clean_text) #making predictions
        sentiment = int(np.argmax(predictions)) #index of maximum prediction
        probability = max(predictions.tolist()[0]) #probability of maximum prediction
        if sentiment==-1: #assigning appropriate name to prediction
            st.error('Sentiment negatif')
            st.write("Polarite : {} ".format(sentiment))
            st.write("Score de confidence': {:.2f}".format(probability))
        elif sentiment==0:
            st.write('Sentiment neutre')
            st.write("Polarite : {} ".format(sentiment))
            st.write("Score de confidence': {:.2f}".format(probability))
        elif sentiment==1:
            st.success('Sentiment postif')
            st.success("Polarite : {} ".format(sentiment))
            st.success("Score de confidence : {:.2f}".format(probability))
        else:
            st.warning("Difficile! Essayer d\'ajouter plus de mots s\'il vous plait")
    st.write('*Note: Le traitement pourrait prendre 30 secondes.*')    
    predict(user_input)
    