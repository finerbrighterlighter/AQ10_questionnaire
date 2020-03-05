#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 00:32:27 2020

@author: hteza
"""

import data_preprocessing
from sklearn.model_selection import train_test_split
from recursive_feature_extraction import rfe

########################################################################################################

df_aad = data_preprocessing.df_aad
features = data_preprocessing.features

########################################################################################################

# Splitting the Dataset
X = df_aad.loc[:, features]
y = df_aad ["class"]
# feature extraction
final_pred = rfe(X,y,features).return_features()

# data for model
X= df_aad.loc[:, final_pred]
y = y
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=50, stratify=y)

########################################################################################################