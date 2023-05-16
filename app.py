from flask import Flask, render_template, request
from flask import Blueprint
import re
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

bp = Blueprint('static', __name__, static_folder='static')

filename = 'model/model_naive_bayes.pkl'
mnb = pickle.load(open(filename, 'rb'))
count_vect = pickle.load(open("model/count_vect.pkl", 'rb'))
tfidf_transformer = pickle.load(open("model/tfidf_transformer.pkl", 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    # kode preprocessing dan prediksi seperti yang di atas
    stop_words = set(stopwords.words('indonesian'))
    test = request.form['kalimat']
    text = re.sub(r'[^\w\s]','', test)
    text = text.encode('ascii', 'replace').decode('ascii')
    text = re.sub(r'\d+', '', text)
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    text = text.lower()
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.casefold() not in stop_words]
    text = ' '.join(filtered_words)
    my_list_test = [text]
    predicted = []

    for i in my_list_test:
        a = count_vect.transform([i])
        X_coba = tfidf_transformer.fit_transform(a).toarray()
        y_pred = mnb.predict(X_coba)
        acc = mnb.predict_proba(X_coba)
        probmnb = acc.max(axis=1)
        pd.options.display.float_format = '{:,.2f}%'.format
        predicted.append({"tweet": test, "stopwords": i, "tokens": filtered_words, "label" : y_pred[0], "probability" : probmnb[0]*100})

    return render_template('index.html', predicted=predicted)

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
