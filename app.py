# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk for spam detector
filename_spam = 'spam-sms-mnb-model.pkl'
classifier_spam = pickle.load(open(filename_spam, 'rb'))
cv_spam = pickle.load(open('cv-transform_spam.pkl','rb'))

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk for resturant review classification
filename_rest = 'restaurant-sentiment-mnb-model.pkl'
classifier_rest = pickle.load(open(filename_rest, 'rb'))
cv_rest = pickle.load(open('cv-transform_rest.pkl','rb'))

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk for movie genre classification
filename_movie = 'movie-genre-mnb-model.pkl'
classifier_movie = pickle.load(open(filename_movie, 'rb'))
cv_movie = pickle.load(open('cv-transform_movie.pkl','rb'))

# Load the Random Forest CLassifier model
filename_dia = 'diabetes-prediction-rfc-model.pkl'
classifier_dia = pickle.load(open(filename_dia, 'rb'))

app = Flask(__name__)

@app.route('/')
def start():
    return render_template('index.html')   

@app.route('/spam')
def home_spam():
	return render_template('home_spam.html')

@app.route('/predict_spam',methods=['POST'])
def predict_spam():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv_spam.transform(data).toarray() 
    	my_prediction = classifier_spam.predict(vect)
    	return render_template('result_spam.html', prediction=my_prediction)


@app.route('/rest')
def home_rest():
	return render_template('home_rest.html')

@app.route('/predict_rest', methods=['POST'])
def predict_rest():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv_rest.transform(data).toarray()
    	my_prediction = classifier_rest.predict(vect)
    	return render_template('result_rest.html', prediction=my_prediction)


@app.route('/movie')
def home_movie():
	return render_template('home_movie.html')

@app.route('/predict_movie',methods=['POST'])
def predict_movie():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv_movie.transform(data).toarray()
    	my_prediction = classifier_movie.predict(vect)
    	return render_template('result_movie.html', prediction=my_prediction)


@app.route('/diabetes')
def home_dia():
	return render_template('home_dia.html')

@app.route('/predict_dia', methods=['POST'])
def predict_dia():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier_dia.predict(data)
        
        return render_template('result_dia.html', prediction=my_prediction)



if __name__ == '__main__':
	app.run(debug=True)



#scikit-learn>=0.18