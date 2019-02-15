import sys
sys.path.append('/Users/nh019849/Documents/workspace/study/ML/projects')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from helpers.encoders import label_encoder, one_hot_encoder

data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
names = ['buying', 'maint', 'doors', 'persons', 'lug-boot', 'safety', 'target']

orig_data = pd.read_csv(data_url, index_col=False, header=None, names=names)

label_encoded_data, label_encoders = label_encoder(orig_data)
one_hot_encoded_data = one_hot_encoder(orig_data, None)

for column in one_hot_encoded_data.columns:
    if 'target' in column:
        del one_hot_encoded_data[column]

X = one_hot_encoded_data
y = label_encoded_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, test_size=0.30)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train.astype("float64"))
X_test = scaler.transform(X_test.astype("float64"))


mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))
