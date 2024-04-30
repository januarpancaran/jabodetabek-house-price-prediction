import numpy as np
from flask import Flask, request, render_template
import pickle
from train.train import df

app = Flask(__name__)

model = pickle.load(open('train/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('pages/hero.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
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

        return render_template('pages/predict.html', prediction_text=output)
    else:
        return render_template('pages/predict.html')

if __name__ == "__main__":
    app.run(debug=True)