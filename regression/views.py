from django.shortcuts import render, redirect
from django.contrib import messages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def DecisionTreeRegression(request):
    context = {}
    context["regressionPage"] = True
    context['title'] = 'Decission Tree Regression'
    template_name = "Regression/Sub-Categories/Decision Tree Regression.html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        input_csv_file = request.FILES['input_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'data file is not a valid CSV file')
            return redirect('decision-tree-regression')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request, 'input file is not a valid CSV file')
            return redirect('decision-tree-regression')
        try:
            dataset = pd.read_csv(csv_file)
            lst = pd.read_csv(input_csv_file)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, :])
            X[:, :] = imputer.transform(X[:, :])
            regressor = DecisionTreeRegressor(random_state=0)
            regressor.fit(X, y)
            result = regressor.predict(lst)
            f_data = []
            for res in result:
                f_data.append(res)
            context['result'] = f_data
        except Exception as e:
            return HttpResponse("Error Occured , Reason : " + str(e))
    return render(request, template_name, context)


def MultipleLinearRegression(request):
    context = {}
    context["regressionPage"] = True
    context['title'] = 'Multiple Linear Regression'
    template_name = "Regression/Sub-Categories/Multiple Linear Regression.html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        input_csv_file = request.FILES['input_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'data file is not a valid CSV file')
            return redirect('multiple-linear-regression')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request, 'input file is not a valid CSV file')
            return redirect('multiple-linear-regression')
        try:
            dataset = pd.read_csv(csv_file)
            lst = pd.read_csv(input_csv_file)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, :])
            X[:, :] = imputer.transform(X[:, :])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=0)
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
            result = regressor.predict(lst)
            f_data = []
            for res in result:
                f_data.append(res)
            context['result'] = f_data
        except Exception as e:
            return HttpResponse("Error Occured , Reason : " + str(e))
    return render(request, template_name, context)


def PolynomialRegression(request):
    context = {}
    context["regressionPage"] = True
    context['title'] = 'Polynomial Regression'
    template_name = "Regression/Sub-Categories/Polynomial Regression.html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        input_csv_file = request.FILES['input_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'data file is not a valid CSV file')
            return redirect('polynomial-regression')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request, 'input file is not a valid CSV file')
            return redirect('polynomial-regression')
        try:
            dataset = pd.read_csv(csv_file)
            lst = pd.read_csv(input_csv_file)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, :])
            X[:, :] = imputer.transform(X[:, :])
            lin_reg = LinearRegression()
            lin_reg.fit(X, y)
            poly_reg = PolynomialFeatures(degree=4)
            X_poly = poly_reg.fit_transform(X)
            lin_reg_2 = LinearRegression()
            lin_reg_2.fit(X_poly, y)
            result = lin_reg_2.predict(poly_reg.fit_transform(lst))
            f_data = []
            for res in result:
                f_data.append(res)
            context['result'] = f_data
        except Exception as e:
            return HttpResponse("Error Occured , Reason : " + str(e))
    return render(request, template_name, context)


def RandomForestRegression(request):
    context = {}
    context["regressionPage"] = True
    context['title'] = 'Random Forest Regression'
    template_name = "Regression/Sub-Categories/Random Forest Regression.html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        input_csv_file = request.FILES['input_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'data file is not a valid CSV file')
            return redirect('random-forest-regression')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request, 'input file is not a valid CSV file')
            return redirect('random-forest-regression')
        try:
            dataset = pd.read_csv(csv_file)
            lst = pd.read_csv(input_csv_file)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, :])
            X[:, :] = imputer.transform(X[:, :])
            regressor = RandomForestRegressor(n_estimators=10, random_state=0)
            regressor.fit(X, y)
            result = regressor.predict(lst)
            f_data = []
            for res in result:
                f_data.append(res)
            context['result'] = f_data
        except Exception as e:
            return HttpResponse("Error Occured , Reason : " + str(e))
    return render(request, template_name, context)


def SimpleLinearRegression(request):
    context = {}
    context["regressionPage"] = True
    context['title'] = 'SImple Linear Regression'
    template_name = "Regression/Sub-Categories/Simple Linear Regression.html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        input_csv_file = request.FILES['input_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'data file is not a valid CSV file')
            return redirect('simple-linear-regression')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request, 'input file is not a valid CSV file')
            return redirect('simple-linear-regression')
        try:
            dataset = pd.read_csv(csv_file)
            lst = pd.read_csv(input_csv_file)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, :])
            X[:, :] = imputer.transform(X[:, :])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1/3, random_state=0)
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
            result = regressor.predict(lst)
            f_data = []
            for res in result:
                f_data.append(res)
            context['result'] = f_data
        except Exception as e:
            return HttpResponse("Error Occured , Reason : " + str(e))
    return render(request, template_name, context)


def SupportVectorRegression(request):
    context = {}
    context["regressionPage"] = True
    context['title'] = 'Support Vector Regression'
    template_name = "Regression/Sub-Categories/Support Vector Regression.html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        input_csv_file = request.FILES['input_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'data file is not a valid CSV file')
            return redirect('support-vector-regression')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request, 'input file is not a valid CSV file')
            return redirect('support-vector-regression')
        try:
            dataset = pd.read_csv(csv_file)
            lst = pd.read_csv(input_csv_file)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            y = y.reshape(len(y), 1)
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, :])
            X[:, :] = imputer.transform(X[:, :])
            sc_X = StandardScaler()
            sc_y = StandardScaler()
            X = sc_X.fit_transform(X)
            y = sc_y.fit_transform(y)
            regressor = SVR(kernel='rbf')
            regressor.fit(X, y.ravel())
            result = sc_y.inverse_transform(regressor.predict(sc_X.transform(lst)))
            f_data = []
            for res in result:
                f_data.append(res)
            context['result'] = f_data
        except Exception as e:
            return HttpResponse("Error Occured , Reason : " + str(e))
    return render(request, template_name, context)
