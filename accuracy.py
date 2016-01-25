# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:41:05 2015

@author: PrasanthS

This program calculates the accuracy of the naive bayes calssfier for the data used.
This program was taken from http://scikit-learn.org/ and optimised for my data.
"""

import glob,pandas as pd,numpy as np
from sklearn.naive_bayes import GaussianNB

#Only a single file is taken for the evaluation
list_ = []
allFiles = glob.glob("./Incidents/Incidents_2014.csv")
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0, nrows=100)
    list_.append(df)
frame = pd.concat(list_)
y = range(len(frame))

#Selecting only the necesssary columns
frame=frame[["DC_DIST","DISPATCH_TIME","TEXT_GENERAL_CODE"]]
frame['DISPATCH_TIME'] = frame['DISPATCH_TIME'].str.extract('(\d+):\d+:\d+')
frame['DISPATCH_TIME'] = frame['DISPATCH_TIME'].astype(float)
frame['DC_DIST'] = frame['DC_DIST'].astype(int)
cols = frame.groupby("TEXT_GENERAL_CODE").size()
cols = cols.to_dict()

#Creating a numpy matrix from the dataframe
xtest = frame.as_matrix(columns = ["DC_DIST","DISPATCH_TIME"])

i = 1
for col in cols:
    cols[col] = i;
    i += 1

for col in cols:
    frame.loc[frame.TEXT_GENERAL_CODE == col,'TEXT_GENERAL_CODE'] = cols[col]

#Creating a list from the dataframe
yt = np.array(frame["TEXT_GENERAL_CODE"])
yt = list(yt)

#Applying the guassian naive bayes classifier.
clf = GaussianNB()

fit_test = clf.fit(xtest, yt).predict(xtest)

print("Number of mislabeled points out of a total %d points : %d"% (xtest.shape[0],(yt != fit_test).sum()))

mislabel = (yt != fit_test).sum()

acc = mislabel/len(frame)

print("Accuracy:",acc*100,"%")
