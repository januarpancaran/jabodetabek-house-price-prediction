import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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

# train_score = model.score(x_train, y_train)
# print(f'Training score: {train_score}')

# y_pred = model.predict(x_test)

# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# print(f"RMSE: {rmse}")

# # Calculate mean and max of the target variable
# mean_y = y.mean()
# max_y = y.max()

# # RMSE as a percentage of the mean and max
# rmse_percentage_of_mean = (rmse / mean_y) * 100
# rmse_percentage_of_max = (rmse / max_y) * 100

# print(f"Mean of target variable: {mean_y}")
# print(f"Max of target variable: {max_y}")
# print(f"RMSE as a percentage of the mean: {rmse_percentage_of_mean:.2f}%")
# print(f"RMSE as a percentage of the max: {rmse_percentage_of_max:.2f}%")

pickle.dump(model, open('train/model.pkl', 'wb'))