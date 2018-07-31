# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:07:34 2018

@author: Erman
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import statistics

class HCandTrump():
    
    def __init__(self):
        
        self.df=pd.read_csv("hate_crimes.csv")
        self.X=None
        self.y=[]
        self.n_estimators=100
        self.clf=None
        
        self.splitRatio=0.2
        self.trainX=[]
        self.trainY=[]
        self.testX=[]
        self.testY=[]
        self.validationAccuracies=[]
        self.kFold=3
        
        self.models=[]
       
        self.finalAccuracy=0
        
        
        
    def cleanseDf(self):
        
        self.df.drop(['gini_index','hate_crimes_per_100k_splc','state','share_non_citizen'], axis=1,inplace=True)
        
        toBeDeleted=[]
        
        for ind in range(len(self.df)):
            if pd.isnull(self.df['avg_hatecrimes_per_100k_fbi'][ind])  :
                toBeDeleted.append(ind)
#        print(toBeDeleted, self.df['state'][toBeDeleted])
        self.df.drop(self.df.index[toBeDeleted], inplace=True)
        self.df.reset_index(inplace=True)
        self.df.drop(['index'], axis=1,inplace=True)
#        print(self.df)
        
    def labelHateCrimes(self):
        
        lowestRate=min(list(self.df['avg_hatecrimes_per_100k_fbi']))
        highestRate=max(list(self.df['avg_hatecrimes_per_100k_fbi']))
        
#        segmentNumber=2
#        segments=[]
#        
#        for segment in range(segmentNumber):
#            segments.append(lowestRate+segment*(highestRate-lowestRate)/float(segmentNumber))
        threshold=statistics.median(list(self.df['avg_hatecrimes_per_100k_fbi']))
##        print(segments)
##        print(lowestRate,highestRate)
#        
#        for ind in range(len(self.df)):
#            for segment in range(segmentNumber):
#                
#                if self.df['avg_hatecrimes_per_100k_fbi'][ind]>=segments[-1]:
#                    self.y.append(segmentNumber-1)
#                    break
##                elif self.df['avg_hatecrimes_per_100k_fbi'][ind]>=segments[segment] and self.df['avg_hatecrimes_per_100k_fbi'][ind]<segments[segment+1] :
##                    self.y.append(segment)
##                    break
#                else:
#                    self.y.append(segment)
        for ind in range(len(self.df)):
            if self.df['avg_hatecrimes_per_100k_fbi'][ind]>=threshold:
                self.y.append(1)
            else:
                self.y.append(0)
        
#        print(self.y)
#        print(self.df['avg_hatecrimes_per_100k_fbi']) 


    def labelTrump(self):

        threshold=0.5
        self.y=[]
        for ind in range(len(self.df)):
            if self.df['share_voters_voted_trump'][ind]>threshold:
                self.y.append(1)
            else:
                self.y.append(0)
                
#        print(self.y)
#        print(self.df['share_voters_voted_trump'])
#        
    def getXY(self):
        
        choice=int(input("Enter 1 for predicting hate crimes and 2 for Trump votes: "))
        self.kFold=int(input("Enter the validation number: "))
        
        if choice==1:
            self.labelHateCrimes()
        elif choice==2:
            self.labelTrump()
        
        
        self.X=self.df.drop(['avg_hatecrimes_per_100k_fbi','share_voters_voted_trump'], axis=1).values.tolist()
                
    def trainTestSplit(self):

        self.trainX, self.testX,self.trainY, self.testY = train_test_split(self.X, self.y, test_size=self.splitRatio, random_state=42)
    

    
    def trainAndValidate(self):

            
            validationRatio=1/float(self.kFold)
            
            for validation in range(self.kFold):
               print("Validation number : ", validation)
               clf=RandomForestClassifier(n_estimators=60)
               
#               clf_svm=GridSearchCV(svm,{'kernel':['linear','poly','rbf'],'C':[0.1,1,10]})
#               clf_svm=GridSearchCV(svm,{'kernel':['linear','poly']})
               self.trainX, self.validateX,self.trainY, self.validateY = train_test_split(self.trainX, self.trainY, test_size=validationRatio)
               clf.fit(self.trainX,self.trainY)
               outcome=clf.predict(self.validateX)
                   
               self.validationAccuracies.append(accuracy_score(outcome,self.validateY))
               self.models.append(clf)
        
    # Choose the model that is the least biased of all validated models.        
            self.clf=self.models[self.validationAccuracies.index(max(self.validationAccuracies))]
    
    # Release the memory
            del self.models[:]
            
            print("Validation Accuracies: ")
            print(self.validationAccuracies)
            
    def test(self):
        self.results=self.clf.predict( self.testX)
        self.finalAccuracy=accuracy_score(self.results,self.testY) 
        
    def predictAndScore(self):
#        self.results=self.model.predict(self.testX)
        print("Accuracy Score: ", accuracy_score(self.results,self.testY ))
        print("Confusion Matrix: ")
        print( confusion_matrix(self.results,self.testY ))

    
        
    def printResults(self):
       
       for ii in range(len(self.results)):
           print(self.testY[ii],self.results[ii])  
           
    def plot_coefficients(self):
        coef = self.clf.feature_importances_
 
         # create plot
        importances = pd.DataFrame({'feature':self.df.drop(['avg_hatecrimes_per_100k_fbi','share_voters_voted_trump'], axis=1).columns.values,'importance':np.round(coef,3)})
        importances = importances.sort_values('importance',ascending=False).set_index('feature')
        print( importances)
        importances.plot.bar()
            

    
        
        
        
        
        
        
        
        
        
if __name__ == '__main__':    
    
    HCT=HCandTrump()
    HCT.cleanseDf()
    HCT.getXY()
    HCT.trainTestSplit()
    HCT.trainAndValidate()
    HCT.test()
    HCT.predictAndScore()
    HCT.plot_coefficients()
 