import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

file_path = 'dataset/jabodetabek_house_price.csv'
df = pd.read_csv(file_path, usecols=['price_in_rp', 'bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2', 'floors', 'building_age'])

df = df.dropna()

x_df = df.drop(columns=['price_in_rp'], axis=1)
y_df = df['price_in_rp']

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.25, random_state=42)

model = LinearRegression()

model.fit(x_train, y_train)

prediction_test = model.predict(x_test)

import pickle
pickle.dump(model, open('train/model.pkl', 'wb'))

model = pickle.load(open('train/model.pkl', 'rb'))