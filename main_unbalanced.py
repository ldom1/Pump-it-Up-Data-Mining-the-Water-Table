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
from codes.data_processing import processData
from codes.unbalanced_predictions import unbalancedPrediction

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

# Predict with the SMOTE method
# ArbreDecision
ArbreDecision = unbalancedPrediction(X, y, np.array(submit), 'ArbreDecision')
ArbreDecision_pred = ArbreDecision.processDataPredicted()

# XGBoost
XGBoost = unbalancedPrediction(X, y, np.array(submit), 'XgBoost')
XGBoost_pred = XGBoost.processDataPredicted()

# RandomForest
RandomForest = unbalancedPrediction(X, y, np.array(submit), 'RandomForest')
RandomForest_pred = RandomForest.processDataPredicted()

# GradientBoosting
GradientBoosting = unbalancedPrediction(X, y, np.array(submit),
                                        'GradientBoosting')
GradientBoosting_pred = GradientBoosting.processDataPredicted()

# Score
print('Score Arbre de décision: ', ArbreDecision_pred[1])
print('Score XGBoost: ', XGBoost_pred[1])
print('Score GradientBoosting: ', GradientBoosting_pred[1])
print('Score RandomForest: ', RandomForest_pred[1])

# Submissions
# Prédiction du dataset - Submit
path_submit = path + '/submissions'

# ArbreDecision
to_submit = pd.DataFrame()
to_submit['id'] = X_submit['id']
to_submit['status_group'] = code.inverse_transform(ArbreDecision_pred[0])
# CSV
to_submit.to_csv(path_submit + '/Submissions_arbre_smote.csv', sep=",",
                 header=True, index=False)
# XGBoost
to_submit = pd.DataFrame()
to_submit['id'] = X_submit['id']
to_submit['status_group'] = code.inverse_transform(XGBoost_pred[0])
# CSV
to_submit.to_csv(path_submit + '/Submissions_xgboost_smote.csv', sep=",",
                 header=True, index=False)
# RandomForest
to_submit = pd.DataFrame()
to_submit['id'] = X_submit['id']
to_submit['status_group'] = code.inverse_transform(RandomForest_pred[0])
# CSV
to_submit.to_csv(path_submit + '/Submissions_rf_smote.csv', sep=",",
                 header=True, index=False)
# GradientBoosting
to_submit = pd.DataFrame()
to_submit['id'] = X_submit['id']
pred = unbalancedPrediction(X, y, np.array(submit), 'GradientBoosting')
to_submit['status_group'] = code.inverse_transform(GradientBoosting_pred[0])
# CSV
to_submit.to_csv(path_submit + '/Submissions_gboost_smote.csv', sep=",",
                 header=True, index=False)
"""
# NN
to_submit = pd.DataFrame()
to_submit['id'] = X_submit['id']
to_submit['status_group'] = NeuralNetwork_predictions[0]
# CSV
to_submit.to_csv(path_submit + '/Submissions_nn.csv', sep=",", header=True,
                 index=False)
"""
