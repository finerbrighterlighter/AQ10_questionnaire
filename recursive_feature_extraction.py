#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:08:46 2020

@author: hteza
"""

import numpy as np
import pandas as pd

class rfe:
    
    def __init__(self,X=None,y=None, columns=None):
        self.X=X
        self.y=y
        self.columns=columns
    
    def rfe(self,X,y,columns):
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_selection import RFE
        self.X=X
        self.y=y
        self.columns=columns
        #no of features
        nof_list=np.arange(1,13)            
        high_score=0
        #Variable to store the optimum features
        nof=0           
        score_list =[]
        for n in range(len(nof_list)):
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0, stratify=y)
            model = LogisticRegression(solver="liblinear")
            rfe = RFE(model,nof_list[n])
            X_train_rfe = rfe.fit_transform(X_train,y_train)
            X_test_rfe = rfe.transform(X_test)
            model.fit(X_train_rfe,y_train)
            score = model.score(X_test_rfe,y_test)
            score_list.append(score)
            if(score>high_score):
                high_score = score
                nof = nof_list[n]
        # after determining the optimum number of features
        #Initializing RFE model
        rfe = RFE(model, nof)
        #Transforming data using RFE
        X_rfe = rfe.fit_transform(X,y)  
        #Fitting the data to model
        model.fit(X_rfe,y)
        temp = pd.Series(rfe.support_,index = columns)
        selected_features_rfe = temp[temp==True].index
        print("This function applies Logistic Regression as model and liblinear as solver")
        print("Optimum number of features: %d" %nof)
        print("Score with %d features: %f" % (nof, high_score))
        # print(rfe.support_)
        # print(rfe.ranking_)
        print (list(selected_features_rfe.values))
        return list(selected_features_rfe.values)
        
    def return_features(self):
        X=self.X
        y=self.y
        columns=self.columns
        features = self.rfe(X,y,columns)
        return features
