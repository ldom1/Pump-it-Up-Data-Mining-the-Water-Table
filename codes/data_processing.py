#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:34:07 2018

@author: louisgiron
"""
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class processData:

    def __init__(self, data):
        self.data = data

    def define_label(self):
        """encode the labels"""
        # Encoding labels
        code = LabelEncoder()
        code.fit(['functional', 'non functional', 'functional needs repair'])

        return code.transform(self.data['status_group']), code

    def select_categorial(self, categories):
        """categorial data"""
        return pd.get_dummies(self.data[categories])

    def select_numerical(self, numerical):
        """numerical data"""
        return self.data[numerical]

    def get_processData(self, categories, numerical):
        """processed data"""
        df = processData.select_categorial(self, categories)
        df[numerical] = processData.select_numerical(self, numerical)
        try:
            labels, code = processData.define_label(self)
            return df, labels, code
        except KeyError:
            return df
