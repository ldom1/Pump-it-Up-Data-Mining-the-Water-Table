#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 21:48:16 2018

@author: louisgiron
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection
import random as rd
import tensorflow as tf
from tensorflow import keras
from xgboost import XGBClassifier


class Predict:

    def __init__(self, X, y, code):
        """class prediction"""
        r_s = rd.randint(0, 100)
        (X_train, X_test,
         y_train, y_test) = model_selection.train_test_split(X, y,
                                                             test_size=0.33,
                                                             random_state=r_s)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.code = code

    def make_prediction(self, predictor, X_to_predict):
        if predictor == 'RandomForest':
            pred = Predict.RandomForest(self).predict(X_to_predict)
            pred = self.code.inverse_transform(pred)
            score = Predict.RandomForest(self).score(self.X_test, self.y_test)
            return pred, score

        if predictor == 'GradientBoosting':
            pred = Predict.RandomForest(self).predict(X_to_predict)
            pred = self.code.inverse_transform(pred)
            score = Predict.RandomForest(self).score(self.X_test, self.y_test)
            return pred, score

        if predictor == 'XgBoost':
            pred = Predict.XgBoost(self).predict(X_to_predict)
            pred = self.code.inverse_transform(pred)
            score = Predict.XgBoost(self).score(self.X_test, self.y_test)
            return pred, score

        if predictor == 'NeuralNetwork':
            mean = self.X_train.mean(axis=0)
            std = self.X_train.std(axis=0)
            test_data = np.array((self.X_test - mean) / std)
            test_labels = np.array(self.y_test)
            pred = Predict.NeuralNetwork(self).predict(X_to_predict)
            pred = np.argmax(pred, axis=1)
            pred = self.code.inverse_transform(pred)
            (test_loss,
             test_acc) = Predict.NeuralNetwork(self).evaluate(test_data,
                                                              test_labels)
            return pred, test_acc

    def RandomForest(self):
        """Random Forest"""
        # Check and determine best max depth
        min_depth = 4
        max_depth = 10
        scores = []
        depth_tree = np.arange(min_depth, max_depth)
        for i in depth_tree:
            grid = {'max_depth': i}
            model = RandomForestClassifier(**grid)
            model.fit(self.X_train, self.y_train)
            scores.append(model.score(self.X_test, self.y_test))

        # Define the best model
        forest = RandomForestClassifier(max_depth=np.argmax(scores) + min_depth,
                                        n_estimators=100)
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
        min_depth = 4
        max_depth = 8
        scores = []
        depth_tree = np.arange(min_depth, max_depth)
        for i in depth_tree:
            grid = {'max_depth': i}
            model = GradientBoosting(**grid)
            model.fit(self.X_train, self.y_train)
            scores.append(model.score(self.X_test, self.y_test))

        # Return the best max depth
        grid = {'max_depth': np.argmax(scores) + min_depth}
        model = GradientBoosting(**grid)
        model.fit(self.X_train, self.y_train)
        return model

    def XgBoost(self):
        """GradientBoosting"""
        # Define parameters
        param = {}
        param['booster'] = 'gbtree'
        param['objective'] = 'binary:logistic'

        # Define the model
        model = XGBClassifier()

        # Check and determine best max depth
        min_depth = 4
        max_depth = 10
        scores = []
        depth_tree = np.arange(min_depth, max_depth)
        for i in depth_tree:
            grid = {'max_depth': i}
            model = XGBClassifier(**grid)
            model.fit(self.X_train, self.y_train)
            scores.append(model.score(self.X_test, self.y_test))

        # Return the best max depth
        grid = {'max_depth': np.argmax(scores) + min_depth}
        model = XGBClassifier(**grid)
        model.fit(self.X_train, self.y_train)
        return model

    def NeuralNetwork(self):
        """GradientBoosting"""
        # Split dataset into test / training
        mean = self.X_train.mean(axis=0)
        std = self.X_train.std(axis=0)

        train_data = np.array((self.X_train - mean) / std)
        train_labels = np.array(self.y_train)

        # Build the model
        nb_classe = len(np.unique(train_labels))

        model = keras.Sequential([
                keras.layers.Dense(128, activation=tf.nn.relu),
                keras.layers.Dense(nb_classe, activation=tf.nn.softmax)
                ])

        # Compile the model
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Fit the model
        nb_epochs = 20

        model.fit(train_data, train_labels, epochs=nb_epochs)
        return model
