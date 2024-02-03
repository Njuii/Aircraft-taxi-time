import enum
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import model_selection

from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.sampling.rnd import FloatRandomSampling

from pymoo.optimize import minimize
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.termination import get_termination

from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

import matlab
import matlab.engine

from scipy import stats

def ANFIS_model():
    # Get the data --> drop the unnamed column (copy of the index)
    data = pd.read_csv('./final.csv')
    data.drop(columns='Unnamed: 0',inplace=True)

    # K-Fold cross validator
    kf = model_selection.KFold(n_splits=5,shuffle=True,random_state=3)

    X = data.values

    X_train = []

    X_test = []

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_batch_train = []

        for j in train_index:
            X_batch_train.append(X[j])
        X_train.append(X_batch_train)


        X_batch_test = []

        for k in test_index:
            X_batch_test.append(X[k])
        X_test.append(X_batch_test)


    # Save the different groups into separate dataframes
    columns = ['taxi_time', 
            'distance_gate', 
            'QDepArr', 
            'week_day', 
            'hourly_congestion', 
            'distance PC1', 
            'distance PC2', 
            'congest PC1', 
            'congest PC2']

    train_data_arr = []
    train_data_arr.append(pd.DataFrame(X_train[0],columns=columns))
    train_data_arr.append(pd.DataFrame(X_train[1],columns=columns))
    train_data_arr.append(pd.DataFrame(X_train[2],columns=columns))
    train_data_arr.append(pd.DataFrame(X_train[3],columns=columns))
    train_data_arr.append(pd.DataFrame(X_train[4],columns=columns))

    test_data_arr = []
    test_data_arr.append(pd.DataFrame(X_test[0], columns=columns))
    test_data_arr.append(pd.DataFrame(X_test[1], columns=columns))
    test_data_arr.append(pd.DataFrame(X_test[2], columns=columns))
    test_data_arr.append(pd.DataFrame(X_test[3], columns=columns))
    test_data_arr.append(pd.DataFrame(X_test[4], columns=columns))


    for i in range(len(train_data_arr)):
        train_data_arr[i].to_csv(f'ANFIS_Train_{i}.csv', index = False)
        test_data_arr[i].to_csv(f'ANFIS_Test_{i}.csv', index = False)


    # Start the Matlab part
    import matlab.engine
    eng = matlab.engine.start_matlab()
    # this runs trainingScript.m MATLAB script for training and generating the prediction model
    eng.trainingScript(nargout=0);
    # this runs testingScript.m MATLAB script for testing the prediction model
    eng.testingScript(nargout=0);
    # this prints out the root-mean-square error for the training
    training_RMSE = []
    for i in range(4,11):
        file1 = open(f"trainRMSE-result{float(i/10)}.txt", "r")
        number1 = file1.readline()
        print(f"Training {float(i/10)} RMSE: ", number1)
        training_RMSE.append(float(number1[0:12]))
        file1.close()
    # this prints out the root-mean-square error for the testing
    testing_RMSE = []
    for i in range(4,11):
        file2 = open(f"testRMSE-result{float(i/10)}.txt", "r")
        number2 = file2.readline()
        print(f"Testing {float(i/10)} RMSE: ", number2)
        testing_RMSE.append(float(number2[0:12]))
        file2.close()
    # To change which group is used for cross validation (k-fold cross validation),
    # the data called should be manually changed in trainingScript.m testingScript.m (the MATLAB files with where the anfis model was made)
        # Change number at the end of the different file names (in range [0,4])
        # e.g. ANFIS_Test_0.csv or ANFIS_Train_0.csv ---> ANFIS_Test_1.csv or ANFIS_Train_1.csv


if __name__ == "__main__":
    ANFIS_model()
