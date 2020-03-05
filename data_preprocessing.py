#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 00:30:59 2020

@author: hteza
"""

from scipy.io import arff
import pandas as pd

########################################################################################################

# Loading Dataset
data = arff.loadarff('/Users/hteza/Desktop/Class/RADS602/aj_pok_assignment/Autism_Adolescent_Data/Autism_Adolescent_Data.arff')
df_aad_byte = pd.DataFrame(data[0])
# df_aad_byte.head()

########################################################################################################

# Data Preprocessing

# since we are getting byte strings
# I am going to decode this to utf-8
# I will use applymap to see if any of the variables inside are byte
# if so, decode it (default utf-8)
# else return original

df_aad = df_aad_byte.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
# df_aad.head()

# since the column name have a backslash in it, and in the next lines, when I try to binary it, it shows error
# here I will copy the column rename it, and concat it in the df
df_aad= pd.concat([df_aad.rename(columns={"Class/ASD": "asdclass"})],axis=1)

# we need to chang these categorical variables into (0,1) binary
df_aad["sex"] = df_aad.apply(lambda row: 1 if row.gender=="m" else 0 , axis=1)
df_aad["jud"] = df_aad.apply(lambda row: 1 if row.jundice=="yes" else 0 , axis=1)
df_aad["fhistory"] = df_aad.apply(lambda row: 1 if row.austim=="yes" else 0 , axis=1)
df_aad["class"] = df_aad.apply(lambda row: 1 if row.asdclass=="YES" else 0 , axis=1)

# i don't want a space in my values, which will later become a column name
df_aad["ethnicity"] = df_aad["ethnicity"].str.replace("Middle Eastern", "mideast")
# but I still did, you will see

# it also have 8 categories and some ? values
# df_aad['ethnicity'].value_counts()

# I have missing values coded ?
# so I am dropping those values
df_aad=df_aad[df_aad.ethnicity!="?"].reset_index(drop=True)

########################################################################################################

# the features that I wil be including in the model are
# age ( as the real numbers, not as categories )
# gender ( as dichotomous variable, sex )
# jaundice ( mispelled as jundice, as dichotomous variable, jud)
# family history ( coded in data set as autism )
# ethnicity ( a copy before making dummies )
df_aad["ethnicity_ori"] = df_aad["ethnicity"]

df_aad = pd.get_dummies(df_aad,prefix=["ethnicity"], columns = ["ethnicity"], drop_first=False) 
# somehow I am getting a pace behind 'mideast' like 'mideast '
# just can't figure out how

features= ["age", "sex",
           "ethnicity_Asian", "ethnicity_Black", "ethnicity_Hispanic", 
           "ethnicity_Latino", "ethnicity_mideast ", "ethnicity_Others", 
           "ethnicity_South Asian", "ethnicity_White-European",
           "jud", "fhistory", "result"]

# according to the data set decription 'results' column is the final score calculated
# the summation of all A1 to A10 scores
# I will just include the 'results' value in the model

########################################################################################################