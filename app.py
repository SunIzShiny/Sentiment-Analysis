import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os

import nltk
nltk.download('stopwords')
model_filename = 'model.sav'
vectorizer_filename = 'vectorizer.sav'

with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_filename, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

st.title('Sentiment Analysis with Logistic Regression')


user_input = st.text_area("Enter your text here:")

if st.button('Predict'):
    if user_input:

        def preprocess_text(content):
            import re
            from nltk.corpus import stopwords
            from nltk.stem.porter import PorterStemmer

            port_stem = PorterStemmer()

            content = re.sub('[^a-zA-Z]', ' ', content)
            content = content.lower()
            content = content.split()

            content = [port_stem.stem(word) for word in content if not word in stopwords.words('english')]
            return ' '.join(content)


        preprocessed_input = preprocess_text(user_input)


        transformed_input = vectorizer.transform([preprocessed_input])


        prediction = model.predict(transformed_input)


        if prediction == 1:
            st.write("Sentiment: Positive")
        else:
            st.write("Sentiment: Negative")
    else:
        st.write("Please enter some text.")
