import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('train/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    scaled_prediction = prediction[0]

    if scaled_prediction >= 1e12:
        output = "Rp{:.2f} Triliun".format(scaled_prediction / 1e12)
    elif scaled_prediction >= 1e9:
        output = "Rp{:.2f} Miliar".format(scaled_prediction / 1e9)
    else:
        output = "Rp{:.2f} Juta".format(scaled_prediction / 1e6)

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run()