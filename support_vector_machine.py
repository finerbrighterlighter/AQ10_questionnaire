#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 00:44:23 2020

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

#SVM
SVM_TP, SVM_FP, SVM_TN, SVM_FN, svclassifier = md.svm(X_train, X_test, y_train, y_test).performance()

print ("")
print ("-----------------------------------")
print ("")
print ("Support Vector Machine")
print ("")
print("Precision : ",SVM_TP/float(SVM_TP+SVM_FP))
print ("")
print("Accuracy : ", (SVM_TP+SVM_TN)/float(SVM_TP+SVM_TN+SVM_FP+SVM_FN))
print ("")
print("Sensitivity : ", SVM_TP/float(SVM_TP+SVM_FN))
print ("")
print("Specificity : ", SVM_TN/float(SVM_TN+SVM_FP))
print ("")
print ("-----------------------------------")
print ("")

########################################################################################################

import pickle
pickle.dump(svclassifier, open("support_vector_classifier.pkl","wb"))