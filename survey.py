#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:13:45 2020

@author: hteza
"""
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import pickle

########################################################################################################

app = Flask(__name__)

logreg=pickle.load(open("logistic_regression.pkl","rb"))
logreg_cv=pickle.load(open("logistic_regression_cv.pkl","rb"))
svm=pickle.load(open("support_vector_classifier.pkl","rb"))
# mlp=pickle.load(open("multilayer_perceptron.pkl","rb"))
# MLP somehow gets too sensitive and assume everyone positive, which is why I will not deploy it in application

########################################################################################################

information = {"Age":[],
            "Sex":[],
            "Family History":[],
            "Total Score": []
            }

final_print = {"Age":[],
            "Sex":[],
            "Family History":[]
            }

personal = {"How old is the child ?":[12,13,14,15,16],
            "What is the gender of the child ?":["Male","Female"],
            "Is there a family member or a relative with ASD ?":["Yes","No"]
            }

questions = {"S/he notices patterns in things all the time .":['Definitely Agree','Slightly Agree','Slightly Disagree','Definitely Disagree'],
             "S/he usually concentrates more on the whole picture, rather than the small details .":['Definitely Agree','Slightly Agree','Slightly Disagree','Definitely Disagree'],
             "In a social group, s/he can easily keep track of several different people’s conversations .":['Definitely Agree','Slightly Agree','Slightly Disagree','Definitely Disagree'],
             "If there is an interruption, s/he can switch back to what s/he was doing very quickly .":['Definitely Agree','Slightly Agree','Slightly Disagree','Definitely Disagree'],
             "S/he frequently finds that s/he doesn’t know how to keep a conversation going .":['Definitely Agree','Slightly Agree','Slightly Disagree','Definitely Disagree'],
             "S/he is good at social chit-chat .":['Definitely Agree','Slightly Agree','Slightly Disagree','Definitely Disagree'],
             "When s/he was younger, s/he used to enjoy playing games involving pretending with other children .":['Definitely Agree','Slightly Agree','Slightly Disagree','Definitely Disagree'],
             "S/he finds it difficult to imagine what it would be like to be someone else .":['Definitely Agree','Slightly Agree','Slightly Disagree','Definitely Disagree'],
             "S/he finds social situations easy .":['Definitely Agree','Slightly Agree','Slightly Disagree','Definitely Disagree'],
             "S/he finds it hard to make new friends .":['Definitely Agree','Slightly Agree','Slightly Disagree','Definitely Disagree']
             }

group1 = [0,4,7,9]
group2 = [1,2,3,5,6,8]
choices = ['Definitely Agree','Slightly Agree','Slightly Disagree','Definitely Disagree']

models = {"Choose one machine learning model to apply .":["Logistic Regression", \
                                                          "Logistic Regression with Cross Validation", \
                                                              "Support Vector Machine"]}

    


########################################################################################################

@app.route('/')
def index():
    return render_template('intro.html')  
    
@app.route('/begin', methods=['POST'])
def childinfo():
    for i in personal.keys():
        return render_template('personal.html', q = personal, o = personal)
        
@app.route('/information', methods=['POST'])
def info():
    age=request.form['How old is the child ?']
    information["Age"]=int(age)
    sex=request.form['What is the gender of the child ?']
    if sex=="Male":
        information["Sex"]=1
    else:
        information["Sex"]=0
    fhistory=request.form['Is there a family member or a relative with ASD ?']
    if fhistory=="Yes":
        information["Family History"]=1
        final_print["Family History"]="positive"
    else:
        information["Family History"]=0
        final_print["Family History"]="no"
    final_print["Age"]=request.form['How old is the child ?']
    final_print["Sex"]=request.form['What is the gender of the child ?']
    return redirect('/quiz')
    
@app.route('/quiz')
def aq10():
    for i in questions.keys():
        return render_template('question.html', q = questions, o = questions)

@app.route('/score', methods=["POST"])
def score():
    ans=[]
    score = [None]*10
    for i in questions.keys():
        answered = request.form[i]
        ans.append(answered)
    for j in group1:
        if ans[j]==choices[0] or ans[j]==choices[1]:
            score[j]=1
        else:
            score[j]=0
    for j in group2:
        if ans[j]==choices[2] or ans[j]==choices[3]:
            score[j]=1
        else:
            score[j]=0
    information["Total Score"] = sum(score)
    return redirect("/model_selection")

@app.route("/model_selection")
def model_choice():
    for i in models.keys():
        return render_template('models.html', q = models, o = models)

@app.route("/models", methods=['POST'])
def model_selection():
    sel_model = request.form["Choose one machine learning model to apply ."]
    return redirect(url_for('.predict', sel_model=sel_model))
    # return (sel_model)

@app.route("/predict",methods=["GET"])
def predict():
    sel_model = request.args['sel_model']
    if sel_model == "Logistic Regression":
        model = logreg
    elif sel_model == "Logistic Regression with Cross Validation":
        model = logreg_cv
    elif sel_model == "Support Vector Machine":
        model = svm
    # elif sel_model == "Multi-Layer Perceptron":
    #     model = mlp
    features=[]
    for key, value in information.items():
        features.append(value)
    fprint=[]
    for key, value in final_print.items():
        fprint.append(value)
    X_new = np.array(features).reshape(1,-1)
    # return X_values for this subject
    y_pred=model.predict(X_new)
    if y_pred==1:
        outcome = "on the spectrum"
    else:
        outcome = "normal"
    if model == svm:
        probability = "not applicable"
    else:
        prob = model.predict_proba(X_new)
        probability = str(round(prob[0][1],3))
    return "<h1>AQ-10 ( Adolescent Version )</h1>\
        <h2>Autism Spectrum Quotient</h2>\
            <h3><u>Results</u></h3>\
                The age of the child is <u>"+str(fprint[0])+"</u> .<br />\
          The gender of the child is <u>"+str(fprint[1])+"</u> .<br />\
              There is <u>"+str(fprint[2])+"</u> family history.<br />\
                  The AQ-10 score is <u>"+str(features[3])+"</u> .<br />\
                      Probability of being on the spectrum is <u>"+probability+ \
                          "</u> and <br /> this child is considered <u>"+ outcome+"</u> ."

if __name__ == '__main__':
    app.run(debug=True)
