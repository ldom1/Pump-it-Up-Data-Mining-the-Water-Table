#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:08:38 2018

@author: louisgiron
"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

from codes.predicteurs import Predict


class unbalancedPrediction:
    # Transform 3 classes problem in 2 classes problem
    # class 0: functional
    # class 1: functional needs repair
    # class 2: non functional

    def __init__(self, x_learn, y_learn, x_predict, predictor):

        self.predictor = predictor
        self.x_data = x_learn
        self.x_data['labels'] = y_learn
        self.x_predict = x_predict

    def predict_class_1(self):
        """Predict the first classes 0 or 2, within the class 0, there are
        the classes 0 and 1"""
        df = pd.DataFrame()
        self.x_data['Class 1'] = self.x_data['labels'].apply(lambda x: 2 if
                                                             x == 2 else 0)

        X_1 = self.x_data[self.x_data.columns[:-2]]
        y_1 = self.x_data['Class 1']

        predicteur = Predict(np.array(X_1), np.array(y_1))

        df_pred = pd.DataFrame()
        df_score = pd.DataFrame()
        df_score['labels'] = np.array(self.x_data['labels'])

        df_pred['Class 1'] = predicteur.make_prediction(self.predictor,
                                                        np.array(self.x_predict))[0]
        df_score['Class 1'] = predicteur.make_prediction(self.predictor,
                                                         np.array(X_1))[0]
        return df_pred, df_score

    def predict_class_2(self):
        """Predict the classes within the class 0, where there are
        the classes 0 and 1"""
        df_pred, df_score = unbalancedPrediction.predict_class_1(self)
        df_temp = self.x_data[self.x_data['labels'] != 2]

        X_2 = df_temp[df_temp.columns[:-2]]
        y_2 = df_temp['labels']

        # Smote method to adress the issue of unbalanced data
        sm = SMOTE(random_state=2)
        X_2_sm, y_2_sm = sm.fit_sample(X_2, y_2.ravel())

        # Prediction on the values on balanced
        predicteur = Predict(X_2_sm, y_2_sm)
        y2_score = predicteur.make_prediction(self.predictor, np.array(X_2))[0]
        y2_pred = predicteur.make_prediction(self.predictor, np.array(self.x_predict))[0]

        # Merge on the id
        df_temp['Class 2'] = y2_score
        df_score = pd.merge(df_score, df_temp[['Class 2']], how='outer',
                            left_index=True, right_index=True)

        # Define predictions
        df_pred['Class 2'] = y2_pred

        return df_pred, df_score

    def processDataPredicted(self):
        """Process the all dataframe with all the predictions and display
        the predictions"""
        df_pred, df_score = unbalancedPrediction.predict_class_2(self)

        # Definition de la prédiction
        def concat_predict_pred(class_1, class_2):
            if class_1 == 2:
                return int(class_1)
            else:
                return int(class_2)

        def concat_predict_score(class_1, class_2):
            if pd.isna(class_2):
                return int(class_1)
            else:
                return int(class_2)

        df_pred['pred'] = df_pred.apply(lambda row: concat_predict_pred(row['Class 1'],
                                                                        row['Class 2']),
                                        axis=1)
        df_score['pred'] = df_score.apply(lambda row: concat_predict_score(row['Class 1'],
                                                                           row['Class 2']),
                                          axis=1)
        score = 1 - np.sum(df_score['pred'] !=
                           df_score['labels'])/len(df_score['pred'])
        return np.array(df_pred['pred']), score
