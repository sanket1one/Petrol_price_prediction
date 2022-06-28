import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


car_dataset = pd.read_csv('car data.csv')
car_dataset.head()   #prints first 5 rows of dataset
car_dataset.shape #gives the number of coulmns and rows in dataset
car_dataset.info() # getting some information about the dataset

# checking the number of missing values
car_dataset.isnull().sum()

# checking the distribution of categorical data
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())

# encoding "Fuel_Type" Column
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# encoding "Seller_Type" Column
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# encoding "Transmission" Column
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

car_dataset.head()

X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y = car_dataset['Selling_Price']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

# loading the linear regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)

# prediction on Training data
training_data_prediction = lin_reg_model.predict(X_train)


# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

# plt.scatter(Y_train, training_data_prediction)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title(" Actual Prices vs Predicted Prices")
# plt.show()


# prediction on Training data
g = [2014, 5.59, 24000, 0, 0, 0, 0]
test_data_prediction = lin_reg_model.predict(X_test)
s = lin_reg_model.predict(pd.DataFrame([g],
                                           columns=['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type',
                                                    'Transmission', 'Owner']))
# plt.scatter(Y_test, test_data_prediction)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title(" Actual Prices vs Predicted Prices")
# plt.show()

