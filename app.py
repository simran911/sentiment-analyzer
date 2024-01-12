import pickle 
import streamlit as st
import re
from nltk.corpus import stopwords  

# Loading the saved model
trained_model = pickle.load(open('trained_model.sav', 'rb'))
porter = pickle.load(open('porter_Stemmer.pkl', 'rb'))
tfidf = pickle.load(open('vectorize.pkl', 'rb'))

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [porter.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

st.title("SIMRAN'S - SENTIMENT ANALYZER")
input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    # Stemming
    stemmed = stemming(input_sms)

    # Vectorizing
    vector_input = tfidf.transform([stemmed])

    # Predict
    prediction = trained_model.predict(vector_input)

    # Display result in the Streamlit app
    if prediction[0] == 1:
        st.write("Positive review")
    else:
        st.write("Negative review")
