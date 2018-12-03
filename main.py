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
from imblearn.over_sampling import SMOTE

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

# Predict
sm = SMOTE(random_state=2)
X_sm, y_sm = sm.fit_sample(X, y.ravel())
predicteur = Predict(X_sm, y_sm)

ArbreDecision_predictions = predicteur.make_prediction('ArbreDecision',
                                                       np.array(submit))
XGBoost_predictions = predicteur.make_prediction('XgBoost', np.array(submit))
RandomForest_predictions = predicteur.make_prediction('RandomForest',
                                                      np.array(submit))
GradientBoosting_predictions = predicteur.make_prediction('GradientBoosting',
                                                          np.array(submit))
NeuralNetwork_predictions = predicteur.make_prediction('NeuralNetwork',
                                                       np.array(submit))

# Scores
print('Score Arbre de décision: ', ArbreDecision_predictions[1])
print('Score XGBoost: ', XGBoost_predictions[1])
print('Score GradientBoosting: ', GradientBoosting_predictions[1])
print('Score RandomForest: ', RandomForest_predictions[1])

# Submissions
# Prédiction du dataset - Submit
path_submit = path + '/submissions'

# ArbreDecision
to_submit = pd.DataFrame()
to_submit['id'] = X_submit['id']
to_submit['status_group'] = code.inverse_transform(ArbreDecision_predictions[0])
# CSV
to_submit.to_csv(path_submit + '/Submissions_arbre.csv', sep=",",
                 header=True, index=False)
# XGBoost
to_submit = pd.DataFrame()
to_submit['id'] = X_submit['id']
to_submit['status_group'] = code.inverse_transform(XGBoost_predictions[0])
# CSV
to_submit.to_csv(path_submit + '/Submissions_xgboost.csv', sep=",",
                 header=True, index=False)
# RandomForest
to_submit = pd.DataFrame()
to_submit['id'] = X_submit['id']
to_submit['status_group'] = code.inverse_transform(RandomForest_predictions[0])
# CSV
to_submit.to_csv(path_submit + '/Submissions_rf.csv', sep=",", header=True,
                 index=False)
# GradinBoosting
to_submit = pd.DataFrame()
to_submit['id'] = X_submit['id']
to_submit['status_group'] = code.inverse_transform(GradientBoosting_predictions[0])
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
