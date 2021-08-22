import math
import numpy as np 
from numpy import linalg as LA
import pandas as pd
import random
import copy
import multiprocessing as mp

from sklearn import ensemble
from sklearn.metrics.pairwise import rbf_kernel
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.feature_selection import mutual_info_regression, SelectFromModel
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import plot_roc_curve, roc_curve, auc, roc_auc_score, mean_squared_error, r2_score, explained_variance_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KernelDensity
from sklearn.metrics import log_loss
from programs.functions import *

############################################################

#Definindo função
def evaluate_Logreg_w(c, X_train, y_train, X_test, y_test, reg):
    model = LogisticRegression(penalty=reg, C=c, random_state=42, solver='liblinear', n_jobs=-1)
    model.fit(X_train,y_train)
    return [c, log_loss(y_test, model.predict_proba(X_test)[:,1])]
    
def valid_Logreg_w(self, X, y, reg='l1', set_cv=[(10**-4,.5),(.501,2),(2.001,5)]): 
    #Arrumando dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

    #Fazendo testes
    for interval in set_cv:
       
        #pool = mp.Pool(mp.cpu_count())
        #output = pool.starmap(evaluate_Logreg_w, [(c, X_train, y_train, X_test, y_test, reg) for c in np.linspace(interval[0],interval[1],12)]) 
        #pool.close()
        
        output=[evaluate_Logreg_w(c, X_train, y_train, X_test, y_test, reg) for c in np.linspace(interval[0],interval[1],8)]
        
        #Output
        output=np.array(output)
        index=np.argmin(output[:,1])
        
        if output[index][0]==interval[1]: continue
        else: break
  
    return output[index][0], output[index][1]

class Logreg_w:
    def __init__(self):
        self.eps=10**-5
    
    def fit(self,X0 ,X1, reg='l1'):
        
        #Arrumando dados
        X=np.vstack((X0, X1))
        y=np.hstack((np.zeros(np.shape(X0)[0]),np.ones(np.shape(X1)[0])))
        
        #Pegando melhores hiper.
        self.best_c, self.log_loss = valid_Logreg_w(self, X, y, reg)
        
        #Treinando modelo final
        model=LogisticRegression(penalty=reg,C=self.best_c,random_state=0,solver='liblinear', n_jobs=-1)
        model.fit(X,y)
        self.model=model
    
    def predict(self, X):
        return (self.model.predict_proba(X)[:,0])/(self.model.predict_proba(X)[:,1] + self.eps)
    
    def predict_mar(self, X):
        return 1/(self.model.predict_proba(X)[:,1]+ self.eps)
    
    def get_log_loss(self): return self.log_loss
 

class pca_98:
    def __init__(self):
        pass
    
    def fit(self,X):
        pca = PCA()
        pca.fit(X)
        cumsum=np.cumsum(pca.explained_variance_ratio_)
        index=1
        while cumsum[index]<.99:
            index+=1
        
        pca = PCA(n_components=index+1)
        pca.fit(X)
        return pca

###
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def unique_columns(a):
    return unique_rows(a.astype(float).T).T

class Poly_Logreg_w:
    def __init__(self, degree=2):
        self.degree=degree
        self.eps=10**-5
    
    def fit(self,X0 ,X1, reg='l1'):
        
        #Arrumando dados
        X=np.vstack((X0, X1))
        y=np.hstack((np.zeros(np.shape(X0)[0]),np.ones(np.shape(X1)[0])))
        poly = PolynomialFeatures(self.degree)
        X = poly.fit_transform(X)
        #X = unique_columns(X)
        #self.pca=pca_98().fit(X)
        #X = self.pca.transform(X)
        
        #Pegando melhores hiper.
        self.best_c, self.log_loss = valid_Logreg_w(self, X, y, reg)
        #print(self.best_c)
      
        #Treinando modelo final
        model=LogisticRegression(penalty=reg,C=self.best_c,random_state=0,solver='liblinear', n_jobs=-1)
        model.fit(X,y)
        self.model=model
    
    def predict(self, X):
        poly = PolynomialFeatures(self.degree)
        X = poly.fit_transform(X)
        #X = unique_columns(X)
        #X = self.pca.transform(X)
        return (self.model.predict_proba(X)[:,0])/(self.model.predict_proba(X)[:,1] + self.eps)
    
    def predict_mar(self, X):
        poly = PolynomialFeatures(self.degree)
        X = poly.fit_transform(X)
        #X = unique_columns(X)
        #X = self.pca.transform(X)
        return 1/(self.model.predict_proba(X)[:,1]+ self.eps)
    
    def calc_log_loss(self, X): 
        poly = PolynomialFeatures(self.degree)
        X = poly.fit_transform(X)
        #X = unique_columns(X)
        #X = self.pca.transform(X)
        y = np.zeros(np.shape(X)[0])
        y[0]=1 #gambs
        return log_loss(y, self.model.predict_proba(X)[:,1])
         
    def get_log_loss(self): return self.log_loss