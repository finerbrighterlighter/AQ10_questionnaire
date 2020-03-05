#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 00:46:39 2020

@author: hteza
"""

import feature_selection
import models as md

########################################################################################################

X_train = feature_selection.X_train
X_test = feature_selection.X_test
y_train = feature_selection.y_train
y_test = feature_selection.y_test

########################################################################################################

#MLP
MLP_TP, MLP_FP, MLP_TN, MLP_FN, mlpclassifier = md.mlp(X_train, X_test, y_train, y_test).performance()

print ("")
print ("-----------------------------------")
print ("")
print ("Multi-layer Perceptron classifier")
print ("")
print("Precision : ",MLP_TP/float(MLP_TP+MLP_FP))
print ("")
print("Accuracy : ", (MLP_TP+MLP_TN)/float(MLP_TP+MLP_TN+MLP_FP+MLP_FN))
print ("")
print("Sensitivity : ", MLP_TP/float(MLP_TP+MLP_FN))
print ("")
print("Specificity : ", MLP_TN/float(MLP_TN+MLP_FP))
print ("")
print ("-----------------------------------")
print ("")

md.mlp(X_train, X_test, y_train, y_test).plot()

########################################################################################################

import pickle
pickle.dump(mlpclassifier, open("multilayer_perceptron.pkl","wb"))