#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:11:05 2020

@author: hteza
"""
from sklearn.metrics import confusion_matrix

########################################################################################################

class logreg:
    
    def __init__(self,X_train=None, X_test=None, y_train=None, y_test=None):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        
        
    def logreg(self,X_train, X_test, y_train, y_test):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import RandomizedSearchCV
        lr = {}
        # random search
        lr["hyperparams"] = {"C": [0.0001,0.001,0.01,0.1,1],
                             "solver": ["lbfgs","liblinear","sag","saga"]}
        lr["randomcv"]= RandomizedSearchCV(estimator=LogisticRegression(max_iter=1000000,
                                                                        penalty="l2"),
                                           param_distributions = lr["hyperparams"],
                                           cv=10,
                                           iid=True)
        lr["learn"]=lr["randomcv"].fit(X_train,y_train)
        # tuning parameter
        lr["focal_params"] = {"C": lr["learn"].best_estimator_.get_params()["C"],
                              "solver" : lr["learn"].best_estimator_.get_params()["solver"]}
        # grid search
        lr["fine_tuned"] =  {"C": [lr["focal_params"]["C"]*i/10. for i in (1,10)]+
                             [lr["focal_params"]["C"]*10. for i in (1,10)],
                             "solver":[lr["focal_params"]["solver"]]}
        lr["gridcv"]= RandomizedSearchCV(estimator=LogisticRegression(max_iter=1000000,
                                                                penalty="l2"),
                                         param_distributions = lr["fine_tuned"],
                                         cv=10,
                                         iid=True,
                                         return_train_score=True)
        lr["tuned_model"]=lr["gridcv"].fit(X_train,y_train)
        print("")
        print("The hyperparameters used for Logistic Regression")
        print("")
        print ("C : "+ str(lr["tuned_model"].best_estimator_.get_params()["C"]),
              "\n",
              "\nSolver : "+ str(lr["tuned_model"].best_estimator_.get_params()["solver"]),
              "\n")
        tuned_C=lr["tuned_model"].best_estimator_.get_params()["C"]
        tuned_solver=lr["tuned_model"].best_estimator_.get_params()["solver"]
        tuned_lr = LogisticRegression(C=tuned_C, solver= tuned_solver)
        tuned_lr.fit(X_train,y_train)
        y_pred = tuned_lr.predict(X_test)
        y_score = tuned_lr.predict_proba(X_test)[:,1]
        return (y_pred,y_score,tuned_C,tuned_solver,tuned_lr)

    
    def performance(self):
        X_train=self.X_train
        X_test=self.X_test
        y_train=self.y_train
        y_test=self.y_test
        y_pred, y_score,tuned_C,tuned_solver,lr_model = self.logreg(X_train, X_test, y_train, y_test)
        CM = confusion_matrix(y_test, y_pred)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        return (TP, FP, TN, FN, tuned_C,tuned_solver, lr_model)
    
    def plot(self):
        X_train=self.X_train
        X_test=self.X_test
        y_train=self.y_train
        y_test=self.y_test
        y_pred, y_score, tuned_C, tuned_solver, lr_model = self.logreg(X_train, X_test, y_train, y_test)
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, threshold = roc_curve(y_test, y_score)
        auc = auc(fpr,tpr)
        import matplotlib.pyplot as plt
        plt.title("Reciever Operating Characteristics")
        plt.plot(fpr,tpr,"r",label = "Logistic Regression without Cross Validation = %0.2f" % auc)
        plt.legend(loc="lower right")
        plt.plot([0,1],[0,1],"b--")
        plt.ylabel("True positive rate")
        plt.xlabel("False positive rate")
        plt.savefig("logreg.svg")
        
########################################################################################################
        
class logreg_cv:
    
    def __init__(self,X=None,y=None,C=None,solver=None):
        self.X=X
        self.y=y
        self.C=C
        self.solver=solver
    
    def logreg_cv(self,X,y,C,solver):
        import numpy as np
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        fold = RepeatedStratifiedKFold(n_splits=10)
        cv_score =[]
        i=1
        for train_index,test_index in fold.split(X,y):
            X_train,X_test = X.loc[train_index],X.loc[test_index]
            y_train,y_test = y.loc[train_index],y.loc[test_index]
            #model
            lr = LogisticRegression(C=C,solver=solver)
            lr.fit(X_train,y_train)
            score = roc_auc_score(y_test,lr.predict(X_test))
            cv_score.append(score)
            i+=1
        y_pred = lr.predict(X_test)
        y_score = lr.predict_proba(X_test)[:,1]
        mean_cv= np.mean(cv_score)
        lr_cv=lr
        return (y_test,y_pred,y_score,cv_score,mean_cv,lr_cv)

    def performance(self):
        X=self.X
        y=self.y
        C=self.C
        solver=self.solver
        y_test,y_pred,y_score,cv_score,mean_cv, lr_cv = self.logreg_cv(X,y,C,solver)
        CM = confusion_matrix(y_test, y_pred)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        return (TP, FP, TN, FN, cv_score, mean_cv, lr_cv)
    
    def plot(self):
        X=self.X
        y=self.y
        C=self.C
        solver=self.solver
        y_test,y_pred,y_score,cv_score,mean_cv, lr_cv = self.logreg_cv(X,y,C,solver)
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, threshold = roc_curve(y_test, y_score)
        auc = auc(fpr,tpr)
        import matplotlib.pyplot as plt
        plt.title("Reciever Operating Characteristics")
        plt.plot(fpr,tpr,"y",linestyle="-.",label = "Logistic Regression with Cross Validation = %0.2f" % auc)
        plt.legend(loc="lower right")
        plt.plot([0,1],[0,1],"b--")
        plt.ylabel("True positive rate")
        plt.xlabel("False positive rate")
        plt.savefig("logreg_with_cv.svg")
    
########################################################################################################
        
class svm:
    
    def __init__(self,X_train=None, X_test=None, y_train=None, y_test=None):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        
    def svm(self,X_train, X_test, y_train, y_test):
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        # tuning hyper parameter
        tuned_parameters = [{"kernel": ["rbf"],
                             "gamma": [1e-2, 1e-3, 1e-4, 1e-5],
                             "C": [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}, 
                            {"kernel": ["sigmoid"], 
                             "gamma": [1e-2, 1e-3, 1e-4, 1e-5],
                             "C": [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                            {"kernel": ["linear"],
                             "C": [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]
        clf = GridSearchCV(SVC(), tuned_parameters, n_jobs=-1, cv=10, iid=True)
        clf.fit(X_train, y_train)
        print("")
        print("Optimum hyperparameters used for SVM")
        print("")
        print ("C : "+str(clf.best_params_["C"]),
               "\n",
               "\nGamma : "+str(clf.best_params_["gamma"]),
               "\n",
               "\nKernel : "+str(clf.best_params_["kernel"]),
               "\n")
        svclassifier = SVC(C=clf.best_params_["C"],
                           kernel=clf.best_params_["kernel"],
                           gamma=clf.best_params_["gamma"])
        svclassifier.fit(X_train,y_train)
        y_pred= svclassifier.predict(X_test)
        return (y_pred, svclassifier)
    
    def performance(self):
        X_train=self.X_train
        X_test=self.X_test
        y_train=self.y_train
        y_test=self.y_test
        y_pred, svclassifier = self.svm(X_train, X_test, y_train, y_test)
        CM = confusion_matrix(y_test, y_pred)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        return (TP, FP, TN, FN, svclassifier)
    
########################################################################################################
         
class mlp:
    
    def __init__(self,X_train=None, X_test=None, y_train=None, y_test=None):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test   
        
    def mlp(self,X_train, X_test, y_train, y_test):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(max_iter=1000)
        tuned_parameters = {"hidden_layer_sizes": [(50,50,50), (50,100,50), (100,)],
                           "activation": ["tanh", "relu", "identity", "logistic"],
                           "solver": ["lbfgs", "sgd", "adam"],
                           "alpha": [0.0001, 0.05],
                           "learning_rate": ["constant","adaptive","invscaling"]
                           }
        from sklearn.model_selection import GridSearchCV
        clf = GridSearchCV(mlp, tuned_parameters, n_jobs=-1, cv=10, iid=True)
        clf.fit(X_train, y_train)
        print("")
        print("Optimum hyperparameters used for MLP")
        print("")
        print ("Hidden Layers : "+str(clf.best_params_["hidden_layer_sizes"]),
               "\n",
               "\nActivation : "+str(clf.best_params_["activation"]),
               "\n",
               "\nSolver : "+str(clf.best_params_["solver"]),
               "\n",
               "\nAlpha : "+str(clf.best_params_["alpha"]),
               "\n",
               "\nLearning Rate : "+str(clf.best_params_["learning_rate"]),
               "\n")
        mlpclassifier = MLPClassifier(hidden_layer_sizes=clf.best_params_["hidden_layer_sizes"],
                                      activation=clf.best_params_["activation"],
                                      solver=clf.best_params_["solver"],
                                      alpha=clf.best_params_["alpha"],
                                      learning_rate=clf.best_params_["learning_rate"])
        mlpclassifier.fit(X_train,y_train)
        y_pred = mlpclassifier.predict(X_test)
        y_score = mlpclassifier.predict_proba(X_test)
        return (y_pred, y_score, mlpclassifier)
    
    def performance(self):
        X_train=self.X_train
        X_test=self.X_test
        y_train=self.y_train
        y_test=self.y_test
        y_pred, y_score, mlpclassifier = self.mlp(X_train, X_test, y_train, y_test)
        CM = confusion_matrix(y_test, y_pred)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        return (TP, FP, TN, FN, mlpclassifier)

    def plot(self):
        X_train=self.X_train
        X_test=self.X_test
        y_train=self.y_train
        y_test=self.y_test
        y_pred, y_score, mlpclassifier = self.mlp(X_train, X_test, y_train, y_test)
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, threshold = roc_curve(y_test, y_score[:,1])
        auc = auc(fpr,tpr)
        import matplotlib.pyplot as plt
        plt.title("Reciever Operating Characteristics")
        plt.plot(fpr,tpr,"g",linestyle=":",label = "Multi-Layer Perceptron = %0.2f" % auc)
        plt.legend(loc="lower right")
        plt.plot([0,1],[0,1],"b--")
        plt.ylabel("True positive rate")
        plt.xlabel("False positive rate")
        plt.savefig("mlp.svg")
 ########################################################################################################   
    
    
        
        
        