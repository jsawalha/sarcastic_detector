from flask import Flask, render_template, url_for, request
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# load the model from disk
clf = pickle.load(open('trained_model.pkl', 'rb'))
tf_idf= pickle.load(open('tf_idf_vectorizer.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        my_prediction = clf.predict(TfidfVectorizer(vocabulary=tf_idf.get_feature_names()).fit_transform(data).toarray())

    return render_template('results.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)