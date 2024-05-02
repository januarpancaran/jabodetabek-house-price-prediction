import numpy as np
from flask import Flask, request, render_template
import pickle
from train.train import df
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

model = pickle.load(open('train/model.pkl', 'rb'))

matplotlib.use('Agg')

def histogram(df):
    data = df.groupby('price_in_rp')['building_age'].sum()

    plt.figure(figsize=(10, 6))
    data.plot(kind='hist', color='#E0ECE4')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return img_base64


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
    
@app.route('/chart')
def chart():
    df_histogram = histogram(df)

    return render_template('pages/chart.html', df_histogram=df_histogram)

if __name__ == "__main__":
    app.run(debug=True)