#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 00:40:04 2020

@author: hteza
"""

import feature_selection
import models as md

########################################################################################################

X = feature_selection.X
y = feature_selection.y
X_train = feature_selection.X_train
X_test = feature_selection.X_test
y_train = feature_selection.y_train
y_test = feature_selection.y_test

########################################################################################################

# Logistic Regression
LOGREG_TP, LOGREG_FP, LOGREG_TN, LOGREG_FN, logreg_C, logreg_solver, lr_model = md.logreg(X_train, X_test, y_train, y_test).performance()

md.logreg(X_train, X_test, y_train, y_test).plot()
print ("-----------------------------------")


print ("")
print ("Logistic Regression")
print ("")
print("Precision : ",LOGREG_TP/float(LOGREG_TP+LOGREG_FP))
print ("")
print("Accuracy : ", (LOGREG_TP+LOGREG_TN)/float(LOGREG_TP+LOGREG_TN+LOGREG_FP+LOGREG_FN))
print ("")
print("Sensitivity : ", LOGREG_TP/float(LOGREG_TP+LOGREG_FN))
print ("")
print("Specificity : ", LOGREG_TN/float(LOGREG_TN+LOGREG_FP))
print ("")
print ("-----------------------------------")
print ("")

########################################################################################################

# logistic regression cross validation

LOGREG_TP, LOGREG_FP, LOGREG_TN, LOGREG_FN, cv_score, mean_cv, lr_cv = md.logreg_cv(X,y,logreg_C,logreg_solver).performance()

md.logreg_cv(X,y,logreg_C,logreg_solver).plot()
print ("-----------------------------------")


print ("")
print ("Logistic Regression with Repeated Stratified K fold")
print ("")
print("Precision : ",LOGREG_TP/float(LOGREG_TP+LOGREG_FP))
print ("")
print("Accuracy : ", (LOGREG_TP+LOGREG_TN)/float(LOGREG_TP+LOGREG_TN+LOGREG_FP+LOGREG_FN))
print ("")
print("Sensitivity : ", LOGREG_TP/float(LOGREG_TP+LOGREG_FN))
print ("")
print("Specificity : ", LOGREG_TN/float(LOGREG_TN+LOGREG_FP))
print ("")

print("Cv",cv_score,
      "\nMean CV Score", mean_cv)
print ("")
print ("-----------------------------------")
print ("")

########################################################################################################

import pickle
pickle.dump(lr_model, open("logistic_regression.pkl","wb"))
pickle.dump(lr_cv, open("logistic_regression_cv.pkl","wb"))

########################################################################################################
