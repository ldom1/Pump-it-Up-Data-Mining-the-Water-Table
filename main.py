#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:30:43 2018

@author: louisgiron
"""
import pandas as pd
import os
import numpy as np
from codes.map_design import displayMap
from codes.predicteurs import Predict
from codes.data_processing import processData

# Projet
path = os.getcwd()

# Import des données
train_set = pd.read_csv(path + '/data/training_set_values.csv', sep=',')
test_set = pd.read_csv(path + '/data/test_set_values.csv', sep=',')
train_labels = pd.read_csv(path + '/data/training_set_labels.csv', sep=',')

# Display the map
data = pd.merge(train_set, train_labels, how='inner')
map_water_pumps = displayMap(data, 'status_group')
map_water_pumps.display_map()

# Select the id to submit
submit = pd.read_csv(path + '/data/SubmissionFormat.csv', sep=',')
submit_id = submit['id']
X = pd.concat([train_set, test_set])

# Define the density of population
X['density_population'] = X['population']/np.sum(X['population'])

# Define the dataset to submit
X_submit = X.loc[X['id'].isin(submit_id)]
X_learn = X.loc[~X['id'].isin(submit_id)]
X_learn = pd.merge(X_learn, train_labels, on='id', how='inner')

# Encode the dataset
data_transformer = processData(X_submit)

""" On sélectionne les données suivantes en fonction de la feature importance
quantity_group_dry                           0.229899
waterpoint_type_group_other                  0.153313
extraction_type_class_other                  0.109840
quantity_group_enough                        0.089407
longitude                                    0.036699
gps_height                                   0.033653
latitude                                     0.033092
extraction_type_class_handpump               0.028263
extraction_type_class_gravity                0.025792
waterpoint_type_group_hand pump              0.023272
water_quality_unknown                        0.022887
waterpoint_type_group_communal standpipe     0.017886
quantity_group_insufficient                  0.017754
source_type_shallow well                     0.016808
source_type_spring                           0.012250
density_population                           0.011516
"""

categorie_selection = ['quantity_group', 'waterpoint_type_group',
                       'extraction_type_class', 'water_quality', 'source_type']
numerical_selection = ['longitude', 'latitude', 'gps_height',
                       'density_population']
transformed = data_transformer.get_processData(categorie_selection,
                                               numerical_selection)
try:
    submit, y, code = transformed
except ValueError:
    submit = transformed


data_transformer = processData(X_learn)

transformed = data_transformer.get_processData(categorie_selection,
                                               numerical_selection)
try:
    X, y, code = transformed
except ValueError:
    X = transformed

"""TEST""" 
# Transform 3 classes problem in 2 classes problem
# class 0: functional
# class 1: functional needs repair
# class 2: non functional
data = X
data['labels'] = y

# We include functional needs repair in functional and predict
from sklearn.ensemble import RandomForestClassifier
import random as rd
from sklearn import model_selection
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

data['Class 1'] = data['labels'].apply(lambda x: 2 if x == 2 else 0)
X_1 = data[data.columns[:-2]]
y_1 = data['Class 1']
forest = RandomForestClassifier(max_depth=4, n_estimators=100)
r_s = rd.randint(0, 100)
(X_train, X_test,
 y_train, y_test) = model_selection.train_test_split(X_1, y_1,
                                                     test_size=0.33,
                                                     random_state=r_s)
forest.fit(X_train, y_train)
y1_pred = forest.predict(X_1)
data['Class 1 - pred'] = y1_pred

data_2 = data[data['labels'] != 2]
X_2 = data_2[data_2.columns[:-3]]
y_2 = data_2['labels']

plt.hist(y_2)

(X_train, X_test,
 y_train, y_test) = model_selection.train_test_split(X_2, y_2,
                                                     test_size=0.33,
                                                     random_state=r_s)

# Unbalanced data
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
plt.hist(y_train_res)

forest.fit(X_train_res, y_train_res)
y2_pred = forest.predict(X_2)
data_2['Class 2'] = y2_pred
data_2 = data_2[['Class 2']]

# result
result = pd.DataFrame()
result['labels'] = y
result['Class 1'] = np.array(data['Class 1 - pred'])
result = pd.merge(result, data_2, left_index=True, right_index=True,
                  how='outer')


# Definition de la prédiction
def concat_predict(class_1, class_2):
    if pd.isna(class_2) == True:
        return int(class_1)
    else:
        return int(class_2)


result['pred'] = result.apply(lambda row: concat_predict(row['Class 1'],
                                                         row['Class 2']),
                              axis=1)

# Erreur
err = 1 - np.sum(result['pred'] != result['labels'])/len(result['pred'])

"""FIN TEST"""

# Predict
predicteur = Predict(X, y, code)

XGBoost_predictions = predicteur.make_prediction('XgBoost', submit)
RandomForest_predictions = predicteur.make_prediction('RandomForest', submit)
GradientBoosting_predictions = predicteur.make_prediction('GradientBoosting',
                                                          submit)
NeuralNetwork_predictions = predicteur.make_prediction('NeuralNetwork',
                                                       submit)
# Submissions
# Prédiction du dataset - Submit
path_submit = path + '/submissions'
# XGBoost
to_submit = pd.DataFrame()
to_submit['id'] = X_submit['id']
to_submit['status_group'] = XGBoost_predictions[0]
# CSV
to_submit.to_csv(path_submit + '/Submissions_xgboost.csv', sep=",",
                 header=True, index=False)
# RandomForest
to_submit = pd.DataFrame()
to_submit['id'] = X_submit['id']
to_submit['status_group'] = RandomForest_predictions[0]
# CSV
to_submit.to_csv(path_submit + '/Submissions_rf.csv', sep=",", header=True,
                 index=False)
# GradinBoosting
to_submit = pd.DataFrame()
to_submit['id'] = X_submit['id']
to_submit['status_group'] = GradientBoosting_predictions[0]
# CSV
to_submit.to_csv(path_submit + '/Submissions_gboost.csv', sep=",", header=True,
                 index=False)
# XGBoost
to_submit = pd.DataFrame()
to_submit['id'] = X_submit['id']
to_submit['status_group'] = NeuralNetwork_predictions[0]
# CSV
to_submit.to_csv(path_submit + '/Submissions_nn.csv', sep=",", header=True,
                 index=False)
