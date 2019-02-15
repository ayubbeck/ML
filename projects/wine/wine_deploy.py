import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
names = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols",
         "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"]

orig_data = pd.read_csv(data_url, index_col=False, header=None, names=names)

X = orig_data.drop('Cultivator',axis=1)
y = orig_data['Cultivator']

X_train, X_test, y_train, y_test = train_test_split(
    orig_data.drop('Cultivator',axis=1),
    orig_data['Cultivator'],
    train_size=0.70,
    test_size=0.30)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train.astype('float64'))
X_test = scaler.transform(X_test.astype('float64'))

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

pickle.dump(mlp, open("model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))
