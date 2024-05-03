import numpy as np
from flask import Flask, request, render_template
import pickle
from train.train import df
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns

app = Flask(__name__)

model = pickle.load(open('train/model.pkl', 'rb'))

matplotlib.use('Agg')

def histogram(df):
    plt.figure(figsize=(6, 6))
    plt.hist(df['land_size_m2'], bins=10, alpha=0.45, color='red')
    plt.hist(df['building_size_m2'], bins=25, alpha=0.45, color='blue')
    plt.legend(['Luas Tanah', 'Luas Bangunan'])
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return img_base64

def scatter_plot(df):
    fig, axes = plt.subplots(6, 1, figsize=(6, 24))

    # Bedrooms scatter plot
    scatter_bedrooms = sns.scatterplot(data=df, x='bedrooms', y='price_in_rp', ax=axes[0])
    axes[0].set_xlabel('Jumlah Kamar Tidur')
    axes[0].set_ylabel('Harga Bangunan')

    # Bathrooms scatter plot
    scatter_bathrooms = sns.scatterplot(data=df, x='bathrooms', y='price_in_rp', ax=axes[1])
    axes[1].set_xlabel('Jumlah Kamar Mandi')
    axes[1].set_ylabel('Harga Bangunan')

    scatter_land_size = sns.scatterplot(data=df, x='land_size_m2', y='price_in_rp', ax=axes[2])
    axes[2].set_xlabel('Luas Tanah')
    axes[2].set_ylabel('Harga Bangunan')

    scatter_building_size = sns.scatterplot(data=df, x='building_size_m2', y='price_in_rp', ax=axes[3])
    axes[3].set_xlabel('Luas Bangunan')
    axes[3].set_ylabel('Harga Bangunan')

    scatter_floors = sns.scatterplot(data=df, x='floors', y='price_in_rp', ax=axes[4])
    axes[4].set_xlabel('Jumlah Lantai')
    axes[4].set_ylabel('Harga Bangunan')

    scatter_building_age = sns.scatterplot(data=df, x='building_age', y='price_in_rp', ax=axes[5])
    axes[5].set_xlabel('Umur Bangunan')
    axes[5].set_ylabel('Harga Bangunan')

    # Convert the figure to BytesIO
    bytes_io_figure = BytesIO()
    plt.savefig(bytes_io_figure, format='png')
    bytes_io_figure.seek(0)

    # Close the plot to prevent displaying it
    plt.close(fig)

    # Convert BytesIO object to base64 string
    img_figure_base64 = base64.b64encode(bytes_io_figure.getvalue()).decode()

    return img_figure_base64


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
    df_scatter_plot = scatter_plot(df)
    
    return render_template('pages/chart.html', df_histogram=df_histogram, df_scatter_plot=df_scatter_plot)

if __name__ == "__main__":
    app.run(debug=True)