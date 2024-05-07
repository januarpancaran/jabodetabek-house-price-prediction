import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

file_path = 'dataset/jabodetabek_house_price.csv'
df = pd.read_csv(file_path, usecols=['price_in_rp', 'bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2', 'floors', 'building_age'])

df.dropna(inplace=True)

x = df.drop(['price_in_rp'], axis=1)
y = df['price_in_rp']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

train_df = x_train.join(y_train)

model = LinearRegression()

x_train, y_train = train_df.drop(['price_in_rp'], axis=1), train_df['price_in_rp']
model.fit(x_train, y_train)

pickle.dump(model, open('train/model.pkl', 'wb'))