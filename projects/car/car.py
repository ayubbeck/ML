import sys
sys.path.append('/Users/nh019849/Documents/workspace/study/ML/projects')
import numpy as np
import pandas as pd
from helpers.plot import histogram, heatmap
from helpers.learn import learn
from helpers.encoders import label_encoder, one_hot_encoder

random_seed = 7
validation_size = 0.30
scoring = 'accuracy'
data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
names = ['buying', 'maint', 'doors', 'persons', 'lug-boot', 'safety', 'target']

orig_data = pd.read_csv(data_url, names=names)

label_encoded_data, label_encoders = label_encoder(orig_data)
one_hot_encoded_data = one_hot_encoder(orig_data, None)

for column in one_hot_encoded_data.columns:
    if 'target' in column:
        del one_hot_encoded_data[column]
one_hot_encoded_data['target'] = label_encoded_data['target']

print('With Label Encoded: ')
learn(label_encoded_data)
print('With One Hot Encoded: ')
learn(one_hot_encoded_data)
