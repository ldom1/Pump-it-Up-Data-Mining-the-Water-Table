#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 21:48:16 2018

@author: louisgiron
"""
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection
import random as rd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class Predict:

    def __init__(self, X, y):
        """class prediction"""
        r_s = rd.randint(0, 100)
        (X_train, X_valid,
         y_train, y_valid) = model_selection.train_test_split(X, y,
                                                              test_size=0.33,
                                                              random_state=r_s)
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def make_prediction(self, predictor, X_to_predict, display=True):

        def plot_cnf_matrix(pred_valid, name):
            try:
                labels = ["Functionnal", "Need reparations", "Non functionnal"]
                cfn = confusion_matrix(self.y_valid, pred_valid,
                                       labels=labels)
                cfn = cfn/len(pred_valid)*100
                df_cm = pd.DataFrame(cfn, index=[i for i in labels],
                                     columns=[i for i in labels])
            except Exception:
                cfn = confusion_matrix(self.y_valid, pred_valid)
                cfn = cfn/len(pred_valid)*100
                df_cm = pd.DataFrame(cfn)
            plt.figure(figsize=(10, 7))
            plt.title('Confusion matrix - ' + str(name) + ' classifier (%)')
            sns.heatmap(df_cm, annot=True)
            plt.show()

        if predictor == 'ArbreDecision':
            pred = Predict.ArbreDecision(self).predict(X_to_predict)
            pred_valid = Predict.ArbreDecision(self).predict(self.X_valid)
            score = Predict.ArbreDecision(self).score(self.X_valid,
                                                      self.y_valid)
            if display:
                plot_cnf_matrix(pred_valid, 'Decision Tree')
            return pred, score

        if predictor == 'RandomForest':
            pred = Predict.RandomForest(self).predict(X_to_predict)
            pred_valid = Predict.RandomForest(self).predict(self.X_valid)
            score = Predict.RandomForest(self).score(self.X_valid,
                                                     self.y_valid)
            if display:
                plot_cnf_matrix(pred_valid, 'Random Forest')
            return pred, score

        if predictor == 'GradientBoosting':
            pred = Predict.RandomForest(self).predict(X_to_predict)
            pred_valid = Predict.RandomForest(self).predict(self.X_valid)
            score = Predict.RandomForest(self).score(self.X_valid,
                                                     self.y_valid)
            if display:
                plot_cnf_matrix(pred_valid, 'Gradient Boosting')
            return pred, score

        if predictor == 'XgBoost':
            pred = Predict.XgBoost(self).predict(X_to_predict)
            pred_valid = Predict.XgBoost(self).predict(self.X_valid)
            score = Predict.XgBoost(self).score(self.X_valid, self.y_valid)
            if display:
                plot_cnf_matrix(pred_valid, 'XgBoost')
            return pred, score

    def ArbreDecision(self):
        """Random Forest"""
        # Check and determine best max depth
        min_depth = 4
        max_depth = 10

        parameters = {'max_depth': np.arange(min_depth, max_depth)}
        clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=3,
                           n_jobs=-1)
        clf.fit(X=self.X_train, y=self.y_train)

        # Define the best model
        tree_model = tree.DecisionTreeClassifier(**clf.best_params_)
        tree_model.fit(self.X_train, self.y_train)
        return tree_model

    def RandomForest(self):
        """Random Forest"""
        # Check and determine best max depth
        min_depth = 4
        max_depth = 10

        parameters = {'max_depth': np.arange(min_depth, max_depth),
                      'n_estimators': np.arange(50, 200, 50),
                      'class_weight': np.array(['balanced',
                                                'balanced_subsample'])}
        forest = GridSearchCV(RandomForestClassifier(), parameters, cv=3,
                              n_jobs=-1)
        forest.fit(X=self.X_train, y=self.y_train)

        # Define the best model
        forest = RandomForestClassifier(**forest.best_params_)
        forest.fit(self.X_train, self.y_train)
        return forest

    def GradientBoosting(self):
        """GradientBoosting"""
        # Split dataset into test / training
        GradientBoosting = GradientBoostingClassifier(loss="deviance",
                                                      learning_rate=0.1,
                                                      n_estimators=100,
                                                      subsample=1.0,
                                                      criterion='friedman_mse',
                                                      min_samples_split=2,
                                                      min_samples_leaf=1,
                                                      min_weight_fraction_leaf=0.0,
                                                      max_depth=3,
                                                      min_impurity_decrease=0.0,
                                                      min_impurity_split=None,
                                                      init=None,
                                                      random_state=None,
                                                      max_features=None,
                                                      verbose=0,
                                                      max_leaf_nodes=None,
                                                      warm_start=False,
                                                      presort='auto')

        # Check and determine best max depth
        min_depth = 5
        max_depth = 10
        parameters = {'max_depth': np.arange(min_depth, max_depth),
                      'class_weight': np.array(['balanced',
                                                'balanced_subsample'])}
        gb_cv = GridSearchCV(GradientBoosting, parameters, cv=3, n_jobs=-1)
        gb_cv.fit(X=self.X_train, y=self.y_train)

        # Return the best max depth
        model = GradientBoosting(**gb_cv.best_params_)
        model.fit(self.X_train, self.y_train)
        return model

    def XgBoost(self):
        """XgBoost"""
        # Define parameters
        param = {}
        param['booster'] = 'gbtree'
        param['objective'] = 'binary:logistic'

        # Define the model
        model = XGBClassifier()

        # Check and determine best max depth
        min_depth = 10
        max_depth = 15
        parameters = {'max_depth': np.arange(min_depth, max_depth),
                      'min_child_weight': np.arange(1, 6, 2),
                      'class_weight': np.array(['balanced',
                                                'balanced_subsample'])}
        xgb_cv = GridSearchCV(model, parameters, cv=3, n_jobs=-1)
        xgb_cv.fit(X=self.X_train, y=self.y_train)

        # Return the best max depth
        model = XGBClassifier(**xgb_cv.best_params_)
        model.fit(self.X_train, self.y_train)
        return model
