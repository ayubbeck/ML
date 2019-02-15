import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

random_seed = 7
validation_size = 0.30
scoring = 'accuracy'

def learn(df):
    array = df.values
    X = array[:,0:len(df.columns)-1]
    Y = array[:,len(df.columns)-1]

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
        X, Y, test_size=validation_size, random_state=random_seed)

    scaler = preprocessing.StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train.astype("float64")),
        columns=list(df.columns)[:-1])
    X_validation = scaler.transform(X_validation.astype("float64"))

    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=random_seed)
        cv_results = model_selection.cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring=scoring)
        print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
