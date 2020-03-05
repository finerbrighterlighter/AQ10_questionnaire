# AQ10_questionnaire

This project is the Assignment of RADS 602 : Data Mining and Machine Learning Class 
at Department of Clinical Epidemiology and Biostatistics, Faculty of Medicine Ramathibodhi Hospital.

While the models are based on AQ10 questionnaire and real life data set from UCI Machine Learning Repository,
this is not to be taken of any diagnostical values.

[AQ10 for Adolescent](/roc/AQ10-Adolescent.jpg)

[UCI dataset](https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult)

[Web application deployed at Heroku](https://aq10-questionnaires.herokuapp.com)

* Data Preprocessing
* Feature Selection
* Model Deployment
  * Logistic Regression
  * Logistic Regression with Cross Validation
  * Support Vector Machine
  * Multi-layer Perceptron
* Flask Application Deployment
* Heroku Web Application Deployment

## Data Preprocessing

Spelling Errors are corrected.
Categorical variables are changed into dichotomous variables, 0 and 1.
Dummy variables are created from categorical variables with more than two distinct values.
Missing Values are dropped.

## Feature Selection

Feature Selection is done in two ways.
  * Conventional and Medical Knowledge
  * Recursive Feature Extraction
  
 ### Conventional and Medical Knowledge
 
 Variables A1 to A10 are the scores of the questionnaire.
 Results is the cummulative score.
 Variables "A1" to "A10" are dropped from final model.
 
 Country of Residence is considered non-significant unless for epidemiological purposes.
 Variable "Country of Residence" is dropped from final model, in favor of "Ethnicity".
 
 Previous experience with the questionnaire ( application ) is considered non-significant unless for familiarity purposes.
 Variable "Used App before" is dropped from final model.
 
 Age group is non significant since the data set itself is limited to adolescent age group ( Age 12-16 ).
 The values are redundant.
 Variable "Age group" is dropped from final model, in favor of "Age".
 
Relationship between the interviewee and the person of interest is considered non-significant unless for examining person-of-interest's ability to take the questionnaire by self.
Variable "relation" is dropped from final model.

### Recursive Feature Extraction

recursive_feature_extraction.py is submitted separately in favor of Object Oriented Programming.

The model applies Logistic Regression as Model and *liblinear* as solver.
The optimum number of features is 4.
The model eliminates "Ethnicity" from final model.

Final features for the data set are "Age", "Sex", "Family History", and "Total Score from Questionnaire".
Final sample size is 98.

## Model Deployment

models.py is submitted separately in favor of Object Oriented Programming.

### Logistic Regression 

Hyperparameters are searched.

C|Solver
---|----
1.0|lbfgs

Hyperparametes are tuned.

### Logistic Regression with Cross Validation

For Cross Validation, Repeated Stratified K-fold is used.
Number of splits = 10

The same hyperparameters as Logistic Regression are used.
Hyperparametes are tuned.

![Receiver operating characteristic of Logistic Regression](/roc/logreg_with_cv.svg)

### Support Vector Machine

Hyperparameters are searched.

C|Gamma|Kernel
-|-|-
10|0.01|rbf

Hyperparametes are tuned.

### Multi-layer Perceptron

Hyperparameters are searched.

Hidden Layers|Activation|Solver|Alpha|Learning Rate
-|-|-|-|-
(50,50,50)|tanh|lbfgs|0.0001|invscaling

Hyperparametes are tuned.

![Receiver operating characteristic of MLP](/roc/mlp.svg)

Models|Precision|Accuracy|Sensitivity|Specificity
------|---------|--------|-----------|-----------
Logistic Regression|1.0|1.0|1.0|1.0
Logistic Regression with Cross Validation|1.0|1.0|1.0|1.0
Support Vector Machine|1.0|1.0|1.0|1.0
Multi-layer Perceptron|1.0|1.0|1.0|1.0

## Flask Application Deployment

The process spans over 5 page.
* Discalimer
* Patient's History
* Questionnaires
* Machine Learning Model to be applied
* Results

All models are dumped into respective pickle files.
MLP classifier is not deployed in the application.
MLP classifier got too sensitive and assume everyone to be on the spectrum.
Logistic regressions and Support Vector Machine are deployed.

### Patient's History

Age, gender and family history are collected.

### Questionnaires

All ten separate questionnaires are taken NHS source.
Separate answers are collected.

Scoring scheme is deployed per guideline.
Total score is calculated by summation.

### Machine Learning Model to be applied

Model Selection is done by the interviewee.

### Results

Results are given in "Normal" or "On the spectrum".
Logitstic Regression Models also give "Probability of being on the spectrum".

## Heroku Web Application Deployment

All python files are uploaded to Github as per Assignment's instructions.
Pickle files and flask file are uploaded for Heroku.
Requirement txt is created.
Procfile is created.

Web application is deployed on [Heroku](https://aq10-questionnaires.herokuapp.com)
