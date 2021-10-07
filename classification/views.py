from django.shortcuts import render ,redirect
from django.http import HttpResponse
from django.contrib import messages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def DecisionTree(request):
    context = {}
    context['classificationPage'] = True
    context['title'] = 'Decision Tree Classification'
    template_name = "Classification/Sub-Categories/Decision Tree.html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        input_csv_file = request.FILES['input_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request,'data file is not a valid CSV file')
            return redirect('decision-tree')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request,'input file is not a valid CSV file')
            return redirect('decision-tree')
        try:
            dataset = pd.read_csv(csv_file)
            lst = pd.read_csv(input_csv_file)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, :])
            X[:, :] = imputer.transform(X[:, :])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
            classifier.fit(X_train, y_train)
            result = classifier.predict(sc.transform(lst))
            f_data = []
            for res in result:
                f_data.append(res)
            context['result'] = f_data
        except Exception as e:
            return HttpResponse("Error Occured , Reason : " + str(e))
    return render(request,template_name,context)

def KNearestNeighbors(request):
    context = {}
    context['classificationPage'] = True
    context['title'] = 'KNN Classification'
    template_name = "Classification/Sub-Categories/KNN.html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        input_csv_file = request.FILES['input_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request,'data file is not a valid CSV file')
            return redirect('k-nearest-neighbors')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request,'input file is not a valid CSV file')
            return redirect('k-nearest-neighbors')
        try:
            dataset = pd.read_csv(csv_file)
            lst = pd.read_csv(input_csv_file)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, :])
            X[:, :] = imputer.transform(X[:, :])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
            classifier.fit(X_train, y_train)
            result = classifier.predict(sc.transform(lst))
            f_data = []
            for res in result:
                f_data.append(res)
            context['result'] = f_data
        except Exception as e:
            return HttpResponse("Error Occured , Reason : " + str(e))
    return render(request,template_name,context)


def KernelSVM(request):
    context = {}
    context['classificationPage'] = True
    context['title'] = 'Kernel SVM Classification'
    template_name = "Classification/Sub-Categories/Kernel SVM.html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        input_csv_file = request.FILES['input_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request,'data file is not a valid CSV file')
            return redirect('kernel-svm')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request,'input file is not a valid CSV file')
            return redirect('kernel-svm')
        try:
            dataset = pd.read_csv(csv_file)
            lst = pd.read_csv(input_csv_file)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, :])
            X[:, :] = imputer.transform(X[:, :])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            classifier = SVC(kernel = 'rbf', random_state = 0)
            classifier.fit(X_train, y_train)
            result = classifier.predict(sc.transform(lst))
            f_data = []
            for res in result:
                f_data.append(res)
            context['result'] = f_data
        except Exception as e:
            return HttpResponse("Error Occured , Reason : " + str(e))
    return render(request,template_name,context)


# def LogisticRegression(request):
#     context = {}
#     context['classificationPage'] = True
#     context['title'] = 'Logistic Regression Classification'
#     template_name = "Classification/Sub-Categories/Logistic Regression.html"
#     if request.method == 'POST':
#         csv_file = request.FILES['data_file']
#         input_csv_file = request.FILES['input_file']    
#         if not csv_file.name.endswith('.csv'):
#             messages.error(request,'data file is not a valid CSV file')
#             return redirect('logistic-regression')
#         if not input_csv_file.name.endswith('.csv'):
#             messages.error(request,'input file is not a valid CSV file')
#             return redirect('logistic-regression')
#         try:
    #         dataset = pd.read_csv(csv_file)
    #         lst = pd.read_csv(input_csv_file)
    #         X = dataset.iloc[:, :-1].values
    #         y = dataset.iloc[:, -1].values
    #         imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    #         imputer.fit(X[:, :])
    #         X[:, :] = imputer.transform(X[:, :])
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    #         sc = StandardScaler()
    #         X_train = sc.fit_transform(X_train)
    #         X_test = sc.transform(X_test)
    #         classifier = LogisticRegression(random_state = 0)
    #         classifier.fit(X_train, y_train)
    #         result = classifier.predict(sc.transform(lst))
    #         f_data = []
    #         for res in result:
    #             f_data.append(res)
    #          context['result'] = f_data
    #     except Exception as e:
    #         return HttpResponse("Error Occured , Reason : " + str(e))
#     return render(request,template_name,context)

def NaiveBayes(request):
    context = {}
    context['classificationPage'] = True
    context['title'] = 'Naive Bayes Classification'
    template_name = "Classification/Sub-Categories/Naive Bayes.html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        input_csv_file = request.FILES['input_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request,'data file is not a valid CSV file')
            return redirect('naive-bayes')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request,'input file is not a valid CSV file')
            return redirect('naive-bayes')
        try:
            dataset = pd.read_csv(csv_file)
            lst = pd.read_csv(input_csv_file)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, :])
            X[:, :] = imputer.transform(X[:, :])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            classifier = GaussianNB()
            classifier.fit(X_train, y_train)
            result = classifier.predict(sc.transform(lst))
            f_data = []
            for res in result:
                f_data.append(res)
            context['result'] = f_data
        except Exception as e:
            return HttpResponse("Error Occured , Reason : " + str(e))
    return render(request,template_name,context)


def RandomForest(request):
    context = {}
    context['classificationPage'] = True
    context['title'] = 'Random Forest Classification'
    template_name = "Classification/Sub-Categories/Random Forest.html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        input_csv_file = request.FILES['input_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request,'data file is not a valid CSV file')
            return redirect('random-forest')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request,'input file is not a valid CSV file')
            return redirect('random-forest')
        try:
            dataset = pd.read_csv(csv_file)
            lst = pd.read_csv(input_csv_file)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, :])
            X[:, :] = imputer.transform(X[:, :])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
            classifier.fit(X_train, y_train)
            result = classifier.predict(sc.transform(lst))
            f_data = []
            for res in result:
                f_data.append(res)
            context['result'] = f_data
        except Exception as e:
            return HttpResponse("Error Occured , Reason : " + str(e))
    return render(request,template_name,context)


def SupportVector(request):
    context = {}
    context['classificationPage'] = True
    context['title'] = 'Support Vector Classification'
    template_name = "Classification/Sub-Categories/Linear SVM.html"
    if request.method == 'POST':
        csv_file = request.FILES['data_file']
        input_csv_file = request.FILES['input_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request,'data file is not a valid CSV file')
            return redirect('support-vector')
        if not input_csv_file.name.endswith('.csv'):
            messages.error(request,'input file is not a valid CSV file')
            return redirect('support-vector')
        try:
            dataset = pd.read_csv(csv_file)
            lst = pd.read_csv(input_csv_file)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, :])
            X[:, :] = imputer.transform(X[:, :])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            classifier = SVC(kernel = 'linear', random_state = 0)
            classifier.fit(X_train, y_train)
            result = classifier.predict(sc.transform(lst))
            f_data = []
            for res in result:
                f_data.append(res)
            context['result'] = f_data
            y_pred = classifier.predict(X_test)
            from sklearn.metrics import accuracy_score
            acc=accuracy_score(y_test, y_pred)
            context['accuracy']= acc
        except Exception as e:
            return HttpResponse("Error Occured , Reason : " + str(e))
    return render(request,template_name,context)
