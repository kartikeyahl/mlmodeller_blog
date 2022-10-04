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
from datetime import datetime
import math

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def DecisionTreeRegression(request):
    context = {}
    context["regressionPage"] = True
    context['title'] = 'Decission Tree Regression'
    template_name = "Regression/Sub-Categories/Decision Tree Regression.html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'data file is not a valid CSV file')
            return redirect('decision-tree-regression')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request, 'input file is not a valid CSV file')
            return redirect('decision-tree-regression')
        try:
            dataset = pd.read_csv(csv_file)
            dataset.rename(columns={'IN/OUT': 'IN_OUT'}, inplace=True)
            dataset=dataset[dataset.IN_OUT!=1]
            j,k=0,0
            for i in dataset.iloc[:,3]:
              dataset.Date[j]=datetime.strptime(str(i),'%Y%m%d').date()
              j+=1

            dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')
            l=[]
            for i in range(len(dataset)-1):
              j=i+1
              l.append([dataset.iloc[i]['E.Code'], datetime.combine(dataset.Date[0], datetime.strptime(str(dataset.iloc[j]['Time']),'%H%M%S').time()) - datetime.combine(dataset.Date[0], datetime.strptime(str(dataset.iloc[i]['Time']),'%H%M%S').time())])  
              i=i+2
            for i in range(len(l)):
                l[i].insert(1,dataset.iloc[i]['Date'].date())
            for i in range(len(l)):
              l[i][2] = l[i][2].__str__().replace(":",".")
              s=l[i][2]
              l[i][2]=s[0:-3]
            l2=[]
            for i in range(len(l)):
              if 'd' not in str(l[i][2]):
                l2.append(l[i])
            for i in range(len(l2)):
                l2[i][2]=float("{:.2f}".format(float(l2[i][2])))
            l3=[]
            for i in range(len(l2)):
                frac, whole = math.modf(l2[i][2])
                mins=int(whole*60+frac*100)
                if mins>570:
                  mins=mins-570
                  hours=mins//60
                  minutes=mins%60
                  if minutes<10:
                    ot_time = float("{}.0{}".format(hours, minutes))
                  else:
                    ot_time = float("{}.{}".format(hours, minutes))
                  l2[i].append(ot_time)
                  l3.append(l2[i])
            df = pd.DataFrame(l3, columns=['E.Code','Date','Total Time','OT'])
            df=df.drop(['Total Time'], axis=1)
            context['df'] = df
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
