from django.shortcuts import render, redirect
from django.contrib import messages
import numpy as np
import pandas as pd
from matplotlib import image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


def ANNSigmoiod(request):
    context = {}
    context["deeplearningPage"] = True
    context['title'] = 'DL | ANN Signoiod'
    context['title2'] = 'Artificial Neural Network (Binary Output)'
    template_name = "Deep Learning/Sub-Categories/Artificial NN(Sigmoid).html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        input_csv_file = request.FILES['input_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'data file is not a valid CSV file')
            return redirect('ann')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request, 'input file is not a valid CSV file')
            return redirect('ann')
        try:
            dataset = pd.read_csv(csv_file)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, :])
            X[:, :] = imputer.transform(X[:, :])
            lst = pd.read_csv(input_csv_file)
            labelencoder_X_1 = LabelEncoder()
            y = labelencoder_X_1.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            ann = Sequential()
            ann.add(Dense(units=16, activation='relu'))
            ann.add(Dense(units=16, activation='relu'))
            ann.add(Dense(units=1, activation='sigmoid'))
            ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            ann.fit(X_train, y_train, batch_size=100, epochs=150)
            result = ann.predict_classes(sc.transform(lst))  
            f_data = []
            for res in result:
                f_data.append(res)
            context['result'] = f_data
        except Exception as e:
            return HttpResponse("Error Occured , Reason : " + str(e))
    return render(request, template_name, context)


def ANNSoftmax(request):
    context = {}
    context["deeplearningPage"] = True
    context['title'] = 'DL | ANN Softmax'
    context['title2'] = 'Artificial Neural Network (Non-Binary Output)'
    template_name = "Deep Learning/Sub-Categories/Artificial NN(Sigmoid).html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        input_csv_file = request.FILES['input_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'data file is not a valid CSV file')
            return redirect('ann-softmax')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request, 'input file is not a valid CSV file')
            return redirect('ann-softmax')
        try:
            dataset = pd.read_csv(csv_file)
            lst = pd.read_csv(input_csv_file)
            X, y = dataset.iloc[:, :-1].values, dataset.iloc[:, -1].values
            l = len(X[1, :])
            p = (l+1)/2
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, :])
            X[:, :] = imputer.transform(X[:, :])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=0)
            sc = StandardScaler()
            X_train, X_test = sc.fit_transform(X_train), sc.transform(X_test)
            ann = tf.keras.models.Sequential()
            ann.add(tf.keras.layers.Dense(units=p, activation='relu'))
            ann.add(tf.keras.layers.Dense(units=p, activation='relu'))
            ann.add(tf.keras.layers.Dense(units=1, activation='softmax'))
            ann.compile(optimizer='adam', loss='categorical_crossentropy',
                        metrics=['accuracy'])
            ann.fit(X_train, y_train, batch_size=32, epochs=100)
            result = ann.predict_classes(sc.transform(lst))
            f_data = []
            for res in result:
                f_data.append(res)
            context['result'] = f_data
        except Exception as e:
            return HttpResponse("Error Occured , Reason : " + str(e))
    return render(request, template_name, context)
