import sys
import math
import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine

RANDOM_STATE = 42
FIG_SIZE = (10, 7)
seed = 7
scoring = 'accuracy'
data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
feature_names = ['alcohol',
                'malic-acid',
             	'ash',
            	'alcalinity-of-ash',
             	'magnesium',
            	'total-phenols',
             	'flavanoids',
             	'nonflavanoid-phenols',
             	'proanthocyanins',
            	'color-intensity',
             	'hue',
             	'diluted-wines',
             	'proline',
                'target']

orig_data = pandas.read_csv(data_url, index_col=False, header=None, names=feature_names)

features, target = load_wine(return_X_y=True)

# print(features.columns)

# plot all columns in histogram format
# def plot_hist(dataframe):
#     fig = plt.figure(figsize=(20,15))
#     cols = 5
#     rows = math.ceil(float(dataframe.shape[1]) / cols)
#     for i, column in enumerate(dataframe.columns):
#         ax = fig.add_subplot(rows, cols, i+1)
#         ax.set_title(column)
#         if dataframe.dtypes[column] == np.object:
#             dataframe[column].value_counts().plot(kind="bar", axes=ax)
#         else:
#             dataframe[column].hist(axes=ax)
#             plt.xticks(rotation="vertical")
#     plt.subplots_adjust(hspace=0.7, wspace=0.2)
#     plt.show()

# plot_hist(orig_data)
# plt.subplots(figsize=(20,20))
# sns.heatmap(orig_data.corr(), square=True)
# plt.show()

# TODO looks 'flavanoids' and 'nonflavanoid-phenols' are highly correlated
# maybe we should keep only one

features_bad = orig_data.values[:,0:len(list(orig_data.columns))-1]
target_bad = orig_data.values[:,len(list(orig_data.columns))-1]
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    features_bad,
    target_bad,
    train_size=0.70,
    test_size=0.30,
    random_state=42)
print(X_train)
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, target,
                                                    test_size=0.30,
                                                    random_state=42)
print(X_train)
# Fit to data and predict using pipelined GNB and PCA.
unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB())
unscaled_clf.fit(X_train, y_train)
predictions = unscaled_clf.predict(X_test)
print("Accuracy score: %f" % accuracy_score(y_test, predictions))

# Fit to data and predict using pipelined scaling, GNB and PCA.
std_clf = make_pipeline(preprocessing.StandardScaler(), PCA(n_components=2), GaussianNB())
std_clf.fit(X_train, y_train)
predictions = std_clf.predict(X_test)
print("Accuracy score: %f" % accuracy_score(y_test, predictions))

# Extract PCA from pipeline
pca = unscaled_clf.named_steps['pca']
pca_std = std_clf.named_steps['pca']

# Show first principal components
print('\nPC 1 without scaling:\n', pca.components_[0])
print('\nPC 1 with scaling:\n', pca_std.components_[0])

# Use PCA without and with scale on X_train data for visualization.
X_train_transformed = pca.transform(X_train)
scaler = std_clf.named_steps['standardscaler']
X_train_std_transformed = pca_std.transform(scaler.transform(X_train))

# visualize standardized vs. untouched dataset with PCA performed
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)


for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax1.scatter(X_train_transformed[y_train == l, 0],
                X_train_transformed[y_train == l, 1],
                color=c,
                label='class %s' % l,
                alpha=0.5,
                marker=m
                )

for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax2.scatter(X_train_std_transformed[y_train == l, 0],
                X_train_std_transformed[y_train == l, 1],
                color=c,
                label='class %s' % l,
                alpha=0.5,
                marker=m
                )

ax1.set_title('Training dataset after PCA')
ax2.set_title('Standardized training dataset after PCA')

for ax in (ax1, ax2):
    ax.set_xlabel('1st principal component')
    ax.set_ylabel('2nd principal component')
    ax.legend(loc='upper right')
    ax.grid()

plt.tight_layout()

plt.show()

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(X_train[0])
# scaler = preprocessing.StandardScaler()
# X_train = pandas.DataFrame(
#     scaler.fit_transform(X_train),
#     columns=list(orig_data.columns)[:-1])
# X_test = scaler.transform(X_test)
# print(X_train.shape)
# print(X_train.columns)

# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# predictions = lr.predict(X_test)

# lda = LinearDiscriminantAnalysis()
# lda.fit(X_train, y_train)
# predictions = lda.predict(X_test)

# print(confusion_matrix(y_test, predictions))
# print("Accuracy score: %f" % accuracy_score(y_test, predictions))
# print("F1 score: %f" % f1_score(y_test, predictions))

# results = []
# names = []
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# # print(X_train.shape[0])
# n_splits = 10
# n_splits = X_train.shape[0]
# for name, model in models:
#     kfold = model_selection.KFold(n_splits=n_splits, random_state=seed)
#     cv_results = model_selection.cross_val_score(
#         model, X_train, y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
