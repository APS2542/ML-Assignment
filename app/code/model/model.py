from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import joblib
import warnings
import mlflow
class LinearRegression(object):
    
    kfold = KFold(n_splits=3)
        
    def __init__(self,
        regularization=None,
        lr=0.001,
        method='batch',
        weight_init_mode='zeros',
        use_momentum=True,
        momentum=0.9,
        degree=1, #for polynomial features
        num_epochs=500,
        batch_size=50,
        cv=kfold,):
        self.regularization = regularization
        self.lr         = lr
        self.method     = method
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.cv         = cv 
        self.weight_init_mode = weight_init_mode
        self.use_momentum   = use_momentum
        self.momentum   = momentum
        self.degree = degree
        self.prev_step = 0
#metrics 
    #function mse that compute the mse score
    def mse(self, ytrue, ypred):
        ytrue = np.asarray(ytrue, dtype=float).reshape(-1)   #(m,)
        ypred = np.asarray(ypred, dtype=float).reshape(-1)
        m = ytrue.size
        if m == 0:
            raise ValueError("Empty ytrue in mse")
        else:
            return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0] 
    
    #add a function r2 that compute the r2 score 
    def r2(self, ytrue, ypred):
        ss_res = np.sum((ytrue - ypred) ** 2)
        ss_tot = np.sum((ytrue - ytrue.mean()) ** 2)
        r2_score = 1.0 - (ss_res / ss_tot)
        return r2_score
#initializers
    #add a function xavier initialization method to calculated as a random number with uniform prob dist 
    def xavier_init(self,n_input,n_output):
        lower,upper = -(1.0/np.sqrt(n_input)), (1.0/np.sqrt(n_input))
        numbers = np.random.rand(n_output)
        scaled =lower + numbers*(upper-lower)
        return scaled

    def fit(self, X_train, y_train):
        
        self.kfold_scores = list()
        #reset val loss
        self.val_loss_old = np.inf
        #cross-validation
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            

            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]

            if self.degree > 1:
                X_cross_train = self.polynomial_features(X_cross_train, self.degree)
                X_cross_val = self.polynomial_features(X_cross_val, self.degree)

            if self.weight_init_mode in ('zero', 'zeros'):
                self.theta = np.zeros(X_cross_train.shape[1])
            elif self.weight_init_mode in ('xavier', 'xavier_init', 'xavier_weigh_init'):
                self.theta = self.xavier_init(X_cross_train.shape[0],X_cross_train.shape[1])
            else: 
                raise ValueError(f"unknown weight_init: {self.weight_init_mode}")

            #define X_cross_train as only a subset of the data
            #how big is this subset?  => mini-batch size ==> 50
            
            
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__}
                mlflow.log_params(params=params)
                
                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]
                    
                    if self.method == 'stochastic':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = y_cross_train[[batch_idx]] 
                            train_loss = self._train(X_method_train, y_method_train)
                    elif self.method == 'mini-batch':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    else: #batch
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    val_r2 = self.r2(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    mlflow.log_metric(key="val_r2", value=val_r2, step=epoch)
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
            
                self.kfold_scores.append(val_loss_new)
                print(f"Fold {fold}: MSE Score - {val_loss_new}, R2 Score - {val_r2}")
    
    #update
    def _train(self, X, y):
        yhat = self.predict(X)
        m    = X.shape[0]
        
        if self.regularization is not None:
            reg_grad = self.regularization.derivation(self.theta)
        else: reg_grad = 0        
        grad = (1/m) * X.T @(yhat - y) + reg_grad
        #add momentum from pseudocode
        if self.use_momentum:
            step = self.lr*grad
            self.theta = self.theta - step + self.momentum*self.prev_step 
            self.prev_step = step
        else:
            self.theta = self.theta - self.lr * grad
        return self.mse(y, yhat)
    
    def predict(self, X, poly=False):
        if X.ndim == 1:
            X = X.reshape(1, -1)                 # (1, n)
        if poly and self.degree > 1:
            X = self.polynomial_features(X, self.degree)
        return X @ self.theta

    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]
    
    #polynomial features
    def polynomial_features(self,X, degree):
        X_poly = np.ones((X.shape[0], 1))  
        for d in range(1, degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly
    
class Polynomial(LinearRegression):
    def __init__(self, method='batch', lr=0.001, weight_init_mode='zeros',use_momentum=True, momentum=0.9, degree=2):
        super().__init__(regularization=None,lr=lr,method=method,weight_init_mode=weight_init_mode, use_momentum=use_momentum,momentum=momentum,degree=degree)