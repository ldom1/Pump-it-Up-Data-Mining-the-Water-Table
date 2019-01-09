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

    def get_processData(self, categories, numerical):
        """processed data"""
        df = self.data
        try:
            df = df.drop(['status_group'], axis=1)
            return pd.get_dummies(df)
        except KeyError:
            df = pd.get_dummies(df)
        try:
            labels, code = processData.define_label(self)
            return pd.get_dummies(df), labels, code
        except KeyError:
            return pd.get_dummies(df)
