from flask import Flask, render_template, request, jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

app = Flask(__name__)

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
model = LinearRegression()
model.fit(X_train,Y_train)

# prediction on Training data
training_data_prediction = model.predict(X_train)


# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

# plt.scatter(Y_train, training_data_prediction)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title(" Actual Prices vs Predicted Prices")
# plt.show()


# prediction on Training data

test_data_prediction = model.predict(X_test)
#s = model.predict(pd.DataFrame([g],
                                           #columns=['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type',
                                                  #  'Transmission', 'Owner']))
# plt.scatter(Y_test, test_data_prediction)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title(" Actual Prices vs Predicted Prices")
# plt.show()


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Present_Price=float(request.form['Present_Price'])
        Kms_Driven=int(request.form['Kms_Driven'])
        #Kms_Driven2=np.log(Kms_Driven)
        Owner=int(request.form['Owner'])
        Fuel_Type=request.form['Fuel_Type']
        if(Fuel_Type=='Petrol'):
                Fuel_Type=0

        elif(Fuel_Type=='Diesel'):
            Fuel_Type=1

        else:
            Fuel_Type = 2

        Seller_Type=request.form['Seller_Type']
        if(Seller_Type=='Individual'):
            Seller_Type=1
        else:
            Seller_Type=0
        Transmission=request.form['Transmission']
        if(Transmission=='Mannual'):
            Transmission=0
        else:
            Transmission=1
        prediction=model.predict([[Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type,
                                                   Transmission, Owner]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction_text="You Can Sell The Car at {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)