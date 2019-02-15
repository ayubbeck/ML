import pandas
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
         'occupation','relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
         'hours-per-week', 'native-country', 'salary']

orig_data = pandas.read_csv(url, index_col=False, header=None, names=names)

# plot all columns in histogram format
def plot_hist(df):
    fig = plt.figure(figsize=(20,15))
    cols = 5
    rows = math.ceil(float(df.shape[1]) / cols)
    for i, column in enumerate(df.columns):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.set_title(column)
        if df.dtypes[column] == np.object:
            df[column].value_counts().plot(kind="bar", axes=ax)
        else:
            df[column].hist(axes=ax)
            plt.xticks(rotation="vertical")
    plt.subplots_adjust(hspace=0.7, wspace=0.2)
    plt.show()

# Encode the categorical features as numbers
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

# plot_hist(orig_data)

# encoded_data, _ = number_encode_features(orig_data)
# sns.heatmap(encoded_data.corr(), square=True)
# plt.show()

# heat map shows that there is high correlation between education and education-numself.
# so we are removing education column. we are keeping
# education-num because it ordered and numerical
del orig_data["education"]
# print(orig_data[["sex", "relationship"]].head(15))

# encoded_data, encoders = number_encode_features(orig_data)

encoded_data = pandas.get_dummies(orig_data)
# Let's fix the Target as it will be converted to dummy vars too
encoded_data["salary"] = encoded_data["salary_ >50K"]
del encoded_data["salary_ <=50K"]
del encoded_data["salary_ >50K"]

# plot_hist(encoded_data)
array = encoded_data.values
# print(list(encoded_data.columns))
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    array[:,0:len(list(encoded_data.columns))-1],
    array[:,len(list(encoded_data.columns))-1],
    train_size=0.70,
    test_size=0.30)
scaler = preprocessing.StandardScaler()
# print(X_train.shape)
# print(list(encoded_data.columns)[:-1])
X_train = pandas.DataFrame(
            scaler.fit_transform(X_train.astype("float64")),
            columns=list(encoded_data.columns)[:-1])
X_test = scaler.transform(X_test.astype("float64"))

logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)

conf_matrix = confusion_matrix(y_test, predictions)
# plt.figure(figsize=(12,12))
# plt.subplot(2,1,1)
#
# # sns.heatmap(conf_matrix,
# #             annot=True, fmt="d",
# #             xticklabels=encoders["salary"].classes_,
# #             yticklabels=encoders["salary"].classes_)
# # plt.ylabel("Real value")
# # plt.xlabel("Predicted value")
# print(conf_matrix)
print("Accuracy score: %f" % accuracy_score(y_test, predictions))
print("F1 score: %f" % f1_score(y_test, predictions))

# coefs = pandas.Series(logreg.coef_[0], index=list(encoded_data.columns)[:-1])
# coefs.sort_index()
# plt.subplot(2,1,2)
# coefs.plot(kind="bar")
# plt.show()


# binary_data = pandas.get_dummies(orig_data)
# # Let's fix the Target as it will be converted to dummy vars too
# binary_data["salary"] = binary_data["salary_ >50K"]
# del binary_data["salary_ <=50K"]
# del binary_data["salary_ >50K"]
# # plt.subplots(figsize=(20,20))
# # sns.heatmap(binary_data.corr(), square=True)
# # plt.show()
# array = binary_data.values
# X_train, X_test, y_train, y_test = model_selection.train_test_split(
#     array[:,0:len(list(binary_data.columns))-1],
#     array[:,len(list(binary_data.columns))-1],
#     train_size=0.70,
#     test_size=0.30)
# scaler = preprocessing.StandardScaler()
# X_train = pandas.DataFrame(
#     scaler.fit_transform(X_train),
#     columns=list(binary_data.columns)[:-1])
# X_test = scaler.transform(X_test)
