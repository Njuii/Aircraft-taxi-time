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

class DNN:
    class NeuralNetwork:
        class ActivationType(enum.Enum):
            ReLU = 0
            Linear = 1

        def __init__(self):
            self.debug_tol = 50

            self.Biases = []
            self.Linear = []
            self.Weights = []
            self.NonLinear = []
            self.DerivativeBiases = []
            self.DerivativeLinear = []
            self.DerivativeWeights = []

            self.MSE_AcrossIter = []
            self.MSE_AcrossEpochs = []
            self.MeanAbsoluteErrorAcrossIter = []
            self.MeanAbsoluteErrorAcrossEpochs = []

            self.iter_in_epoch = 0


        def BuildWithOptimal(self, optimised, debug=False):
            layer_depths = np.array([1, optimised[4]+2], dtype=int)
            layer_depths[0] = 8
            layer_depths[-1] = 1
            layer_depths[1:-1] = optimised[0]
            batch_size = optimised[2]
            self.Build(layer_depths, batch_size, epochs=optimised[3], 
                       learning=optimised[1], debug=debug,
                       hidden_activation=DNN.NeuralNetwork.ActivationType.ReLU, 
                       final_activation=DNN.NeuralNetwork.ActivationType.Linear)

        def Build(self, layer_depths, n_in_iter, epochs=1, learning=0.1,  debug=True,
                     hidden_activation=ActivationType.ReLU, final_activation=ActivationType.Linear):
            self.debug = debug
            self.epochs = epochs
            self.Learning = learning
            self.ParallelCases = n_in_iter
            self.LayerDepths = layer_depths
            self.HiddenLayers = len(layer_depths) - 2
            self.SetActivationFunction(hidden_activation, final_type=final_activation)
            self.Reset()

        def Reset(self):
            #fixed_random = np.random.default_rng()
            for i in range(self.HiddenLayers + 1):
                self.Biases.append(np.random.randn(self.LayerDepths[i+1], 1))
                self.Weights.append(np.random.randn(self.LayerDepths[i+1], self.LayerDepths[i]))

        def SetFeaturesAndLabels(self, features, labels):
            indices = np.arange(0, np.size(labels), 1, dtype=int)
            np.random.shuffle(indices)
            self.Labels = labels[indices]
            self.Features = features[indices]
        
        def SetTestingSet(self, testing_features, testing_labels):
            self.TestingLabels = testing_labels
            self.TestingFeatures = testing_features

        def SetActivationFunction(self, hidden_type, final_type=ActivationType.Linear):
            self.activation_type = hidden_type
            self.final_type = final_type
        
        def ActivationFunction(self, linear_output, final=False):
            if not final:
                match self.activation_type:
                    case DNN.NeuralNetwork.ActivationType.ReLU:
                        return np.maximum(0, linear_output)
                    
                    case DNN.NeuralNetwork.ActivationType.Sigmoid:
                        return 1 / (1 + np.exp(-linear_output))
            else:
                match self.final_type:
                    case DNN.NeuralNetwork.ActivationType.Linear:
                        return linear_output
        
        def DerivativeActivation(self, linear_output, final=False):
            if not final:
                match self.activation_type:
                    case DNN.NeuralNetwork.ActivationType.ReLU:
                        return np.array(linear_output > 0, dtype=float)
                    
                    case DNN.NeuralNetwork.ActivationType.Sigmoid:
                        tmp = 1 - self.ActivationFunction(linear_output)
                        return self.ActivationFunction(linear_output) * tmp
            else:
                match self.activation_type:
                    case DNN.NeuralNetwork.ActivationType.Linear:
                        return 0
        
        def ForwardPropogation(self):
            self.Linear = []
            self.NonLinear = []
            for i in range(self.HiddenLayers + 1):
                biases = np.array(self.Biases[i])
                weights = np.array(self.Weights[i])

                index = self.iter_in_epoch * self.ParallelCases
                if i == 0:
                    linear = weights.dot(self.Features[index:index+self.ParallelCases, :].T) + biases
                else:
                    linear = weights.dot(self.NonLinear[-1]) + biases
                self.Linear.append(linear)

                if i < self.HiddenLayers + 1:
                    nonlinear = self.ActivationFunction(self.Linear[-1])
                else:
                    nonlinear = self.ActivationFunction(self.Linear[-1], final=True)

                self.NonLinear.append(nonlinear)
            
        def BackwardPropogation(self):
            self.DerivativeBiases = []
            self.DerivativeLinear = []
            self.DerivativeWeights = []
            index = self.iter_in_epoch * self.ParallelCases

            reversed_linear = self.Linear[::-1]
            reversed_weights = self.Weights[::-1]
            reversed_nonlinear = self.NonLinear[::-1]

            for i in range(self.HiddenLayers + 1):
                if i == 0:
                    loss = reversed_nonlinear[i] - self.Labels[index:index+self.ParallelCases]
                    self.DerivativeLinear.append(loss)
                    self.MSE_AcrossIter.append(1/self.ParallelCases * np.sum(loss**2))
                    self.MeanAbsoluteErrorAcrossIter.append(1/self.ParallelCases * np.sum(np.abs(loss)))
            
                else:
                    cmp = self.DerivativeLinear[-1]
                    dln = self.DerivativeActivation(reversed_linear[i])
                    self.DerivativeLinear.append(reversed_weights[i-1].T.dot(cmp) * dln)
                
                if i == self.HiddenLayers:
                    self.DerivativeWeights.append(1 / self.ParallelCases * self.DerivativeLinear[-1].dot(self.Features[index:index+self.ParallelCases, :]))
                else:
                    self.DerivativeWeights.append(1 / self.ParallelCases * self.DerivativeLinear[-1].dot(reversed_nonlinear[i-1].T))
                
                self.DerivativeBiases.append(1 / self.ParallelCases * np.sum(self.DerivativeLinear[-1], 1))

        def UpdateConnections(self):
            reversed_dweights = self.DerivativeWeights[::-1]
            reversed_dbias = self.DerivativeBiases[::-1]

            for i in range(self.HiddenLayers + 1):
                self.Weights[i] -= self.Learning * reversed_dweights[i]
                self.Biases[i] -= self.Learning * reversed_dbias[i].reshape(reversed_dbias[i].shape[0], 1)
            self.iter_in_epoch += 1

        def GradientDescent(self):
            self.TestingLossAcrossEpochs = []
            max = int(np.floor(self.Labels.size / self.ParallelCases)) - 1
            for epoch in range(self.epochs):
                self.MSE_AcrossIter = []
                self.MeanAbsoluteErrorAcrossIter = []
                self.DescendWithTesting()
                self.SetFeaturesAndLabels(self.Features, self.Labels)
                for i in range(int(np.floor(self.Labels.size / self.ParallelCases))):
                #for i in range(10): 
                    self.ForwardPropogation()
                    self.BackwardPropogation()
                    self.UpdateConnections()
                    if self.debug:
                        if ((i+1)%self.debug_tol == 0 or i == max):
                            print("===========Training Set============")
                            print("MSE For iteration:", self.MSE_AcrossIter[i])
                            print("MAE For iteration", self.MeanAbsoluteErrorAcrossIter[i])
                            print("===================================")
                self.MSE_AcrossEpochs.append(1 / self.iter_in_epoch * np.sum(np.array(self.MSE_AcrossIter)))
                self.MeanAbsoluteErrorAcrossEpochs.append(1 / self.iter_in_epoch * np.sum(np.array(self.MeanAbsoluteErrorAcrossIter)))
                if self.debug:
                    print("\x1b[1;31m" + "epoch: ", epoch)
                    print("average testing MSE: ", self.TestingLossAcrossEpochs[-1])
                    print("average testing MAE: ", np.sqrt(self.TestingLossAcrossEpochs[-1]))
                    print("average training MSE: ", self.MSE_AcrossEpochs[-1])
                    print("average training MAE: ", self.MeanAbsoluteErrorAcrossEpochs[-1])
                    print("\x1b[0m===================================")
                self.iter_in_epoch = 0

        def DescendWithTesting(self):
            self.TestingLoss = []
            max = int(np.floor(self.Labels.size / self.ParallelCases)) - 1
            self.SetFeaturesAndLabels(self.TestingFeatures, self.TestingLabels)
            for i in range(int(np.floor(self.Labels.size / self.ParallelCases))):
                self.ForwardPropogation()
                index = self.iter_in_epoch * self.ParallelCases
                loss = self.NonLinear[-1] - self.Labels[index:index+self.ParallelCases]
                self.TestingLoss.append(1/self.ParallelCases * np.sum(loss**2))
                if self.debug:
                    if ((i+1)%self.debug_tol == 0 or i == max):
                        print("Iteration: ", i)
                        print("============Testing Set============")
                        print("Average Loss: ", self.TestingLoss[-1])
                self.iter_in_epoch += 1
            self.TestingLossAcrossEpochs.append(1 / self.iter_in_epoch * np.sum(np.array(self.TestingLoss)))
            self.iter_in_epoch = 0

    class Optimisation(ElementwiseProblem):
        class AlgorithmType(enum.Enum):
            GA = 0
            NONE = 5
        
        def __init__(self, ranges, features_train, features_test, labels_train, labels_test, type=AlgorithmType.GA):
            super().__init__(n_var = 5, n_obj = 1, xl = ranges[:, 0], xu = ranges[:, 1])
            self.LabelsTest = labels_test
            self.LabelsTrain = labels_train
            self.FeaturesTest = features_test
            self.FeaturesTrain = features_train

            self.ranges = ranges # min max n neurons in hidden layer,
                                 # min max learning rate, min max batch size
                                 # min max number of epochs # min max no hidden layers 
            self.SetAlgorithmType(type)

        def _evaluate(self, x, out, *arg, **kwargs):
            tmp = np.array([1, int(np.floor(x[4]))+2], dtype=int)
            tmp[0] = 8
            tmp[-1] = 1
            tmp[1:-1] = int(np.floor(x[0]))
            nn = DNN.NeuralNetwork()
            nn.Build(tmp, int(np.floor(x[2])), epochs=int(np.floor(x[3])), 
                     learning=x[1], debug=False,
                     hidden_activation=DNN.NeuralNetwork.ActivationType.ReLU, 
                     final_activation=DNN.NeuralNetwork.ActivationType.Linear)
            nn.SetFeaturesAndLabels(self.FeaturesTrain, self.LabelsTrain)
            nn.SetTestingSet(self.FeaturesTest, self.LabelsTest)
            nn.GradientDescent()
            f1 = nn.MSE_AcrossEpochs[-1]
            out["F"] = [f1]
            nn.Reset()
        
        def SetAlgorithmType(self, type):
            match type:
                case DNN.Optimisation.AlgorithmType.GA:
                    self.algorithm = GA(pop_size=40, eliminate_duplicates=True)
                
                case _:
                    ValueError("No defined alogrithms match the passed algorithm")
        
        def FindOptimal(self):
            termination = get_termination("n_eval", 10)
            self.Optimised = minimize(self, self.algorithm, seed = 7, termination=termination)
            self.Optimised = [int(np.floor(self.Optimised.X[0])), 
                              self.Optimised.X[1], int(np.floor(self.Optimised.X[2])),
                              int(np.floor(self.Optimised.X[3])), int(np.floor(self.Optimised.X[4]))]
            return self.Optimised

    def __init__(self, ranges, features_train, features_test, labels_train, labels_test):
        self.LabelsTest = labels_test
        self.LabelsTrain = labels_train
        self.FeaturesTest = features_test
        self.FeaturesTrain = features_train
        self.optimised = DNN.Optimisation(ranges, features_train, features_test, labels_train, labels_test)
    
    def FindAndTestOptimalNetwork(self, debug=False):
        self.OptimisedNetwork = DNN.NeuralNetwork()
        optimal = self.optimised.FindOptimal()
        self.OptimisedNetwork.BuildWithOptimal(optimal, debug=debug)
        self.OptimisedNetwork.SetFeaturesAndLabels(self.FeaturesTrain, self.LabelsTrain)
        self.OptimisedNetwork.SetTestingSet(self.FeaturesTest, self.LabelsTest)
        self.OptimisedNetwork.GradientDescent()

        print("==========Hyperparameters==========")
        print("Number of hidden layers: ", optimal[4])
        print("Number of neurons in HL(s): ", optimal[0])
        print("Number of example in batch: ", optimal[2])
        print("Number of epochs iterated: ", optimal[3])
        print("Learning rate: ", optimal[1])
        print("-----------------------------------")

    
    def PerformK_FoldTest(self, df):
        # Get the data --> drop the unnamed column (copy of the index)
        # df.drop(columns='Unnamed: 0',inplace=True)

        # K-Fold cross validator
        kf = KFold(n_splits=5,shuffle=True,random_state=3)
        X = df.drop(columns = 'taxi_time').values
        y = df['taxi_time'].values
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        self.OptimisedNetwork.debug = False

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_batch_train = []
            y_batch_train = []

            for j in train_index:
                X_batch_train.append(X[j])
                y_batch_train.append(y[j])
            X_train.append(X_batch_train)
            y_train.append(y_batch_train)
            X_batch_test = []
            y_batch_test = []

            for k in test_index:
                X_batch_test.append(X[k])
                y_batch_test.append(y[k])
            X_test.append(X_batch_test)
            y_test.append(y_batch_test)

        X_test = np.array(X_test)
        X_train = np.array(X_train)

        y_test = np.array(y_test)
        y_train = np.array(y_train)

        MSE_kfold = []
        self.OptimisedNetwork.Reset()
        for index,value in enumerate(kf.split(X)):
            self.OptimisedNetwork.SetFeaturesAndLabels(X_train[index], y_train[index])
            self.OptimisedNetwork.SetTestingSet(X_test[index], y_test[index])
            self.OptimisedNetwork.GradientDescent()
            MSE_kfold.append(self.OptimisedNetwork.MSE_AcrossEpochs[-1])
            self.OptimisedNetwork.Reset()
        print("============K-Fold Test============")
        for index,value in enumerate(MSE_kfold):
            print("Iteration: ", index, " MSE: ", value)

def Sparse_NN():
    # Get the data --> drop the unnamed column (copy of the index)
    data = pd.read_csv('./final.csv')
    data.drop(columns='Unnamed: 0',inplace=True)

    data
    # K-Fold cross validator
    kf = KFold(n_splits=5,shuffle=True,random_state=3)

    X_dist = data[['distance PC1','distance PC2','distance_gate']].values
    X_congest = data[['congest PC1', 'congest PC2','QDepArr']].values
    X_time = data[['week_day','hourly_congestion']].values

    y = data['taxi_time'].values

    X_dist_train = []
    X_congest_train = []
    X_time_train = []
    y_train = []

    X_dist_test = []
    X_congest_test = []
    X_time_test = []
    y_test = []

    for i, (train_index, test_index) in enumerate(kf.split(X_dist)):
        dist_batch_train = []
        congest_batch_train = []
        time_batch_train = []
        y_batch_train = []

        for j in train_index:
            dist_batch_train.append(X_dist[j])
            congest_batch_train.append(X_congest[j])
            time_batch_train.append(X_time[j])
            y_batch_train.append(y[j])
        X_dist_train.append(dist_batch_train)
        X_congest_train.append(congest_batch_train)
        X_time_train.append(time_batch_train)
        y_train.append(y_batch_train)


        dist_batch_test = []
        congest_batch_test = []
        time_batch_test = []
        y_batch_test = []

        for k in test_index:
            dist_batch_test.append(X_dist[k])
            congest_batch_test.append(X_congest[k])
            time_batch_test.append(X_time[k])
            y_batch_test.append(y[k])
        X_dist_test.append(dist_batch_test)
        X_congest_test.append(congest_batch_test)
        X_time_test.append(time_batch_test)
        y_test.append(y_batch_test)

    # For GPU accelerated learning (didn't really work)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")
    # Convert arrays to tensors (for Pytorch)

    X_dist_train = torch.FloatTensor(X_dist_train)
    X_congest_train = torch.FloatTensor(X_congest_train)
    X_time_train = torch.FloatTensor(X_time_train)

    X_dist_test = torch.FloatTensor(X_dist_test)
    X_congest_test = torch.FloatTensor(X_congest_test)
    X_time_test = torch.FloatTensor(X_time_test)

    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)


    # Set Criterion 
    # (Mean Square Error Loss, L2 norm)
    criterion = nn.MSELoss()


    # Define Model

    # Distance layers
    class S_layers_D(nn.Module):
        def __init__(self):
            super().__init__()
            # self.flatten = nn.Flatten()
            self.sc1=nn.Linear(3, 12,bias=True)

        def forward(self, x):
            x = F.tanh(self.sc1(x))
            return x

    # Congestion layers
    class S_layers_C(nn.Module):
        def __init__(self):
            super().__init__()
            # self.flatten = nn.Flatten()
            self.sc1=nn.Linear(3, 12,bias=True)
        
        def forward(self, x):
            x = F.tanh(self.sc1(x))
            return x

    # Time layers
    class S_layers_T(nn.Module):
        def __init__(self):
            super().__init__()
            # self.flatten = nn.Flatten()
            self.sc1=nn.Linear(2, 8,bias=True)
        

        def forward(self, x):
            x = F.tanh(self.sc1(x))
            return x

    # Final fully connected layers (final model)
    class Fin_layers(nn.Module):
        def __init__(self):
            super().__init__()
            self.network_dist = S_layers_D()
            self.network_congest = S_layers_C()
            self.network_time = S_layers_T()

        
            self.fc2 = nn.Linear(32, 16, bias=True)
            self.out = nn.Linear(16, 1, bias=False)
        
        def forward(self, x1, x2, x3):
            x1 = F.relu(self.network_dist(x1))
            x2 = F.relu(self.network_congest(x2))
            x3 = F.relu(self.network_time(x3))

            x = torch.cat((x1, x2, x3), 1)
            x = F.relu(self.fc2(x))
            x = F.relu(self.out(x))
            return x
        

    # Train model

    torch.manual_seed(3)
    model = Fin_layers()

    # Select optimizer + learning rate 
    # commonly used ADAM-> type of stochastic gradient descent
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Keep track of how loss changes
    epochs = 100
    batch_size = 400
    batches_per_epoch = int(800/batch_size)
    losses=[]
    losses_test = []

    group = 4

    for i in range(epochs):

        # Shuffler for batches
        train_order = list(range(800))
        np.random.shuffle(train_order)
    
        for j in range(batches_per_epoch):
            # Training
            y_pred=model.forward(X_dist_train[group][train_order[int(j*batch_size):int(j*batch_size + batch_size+1)]],
                                X_congest_train[group][train_order[int(j*batch_size):int(j*batch_size + batch_size+1)]],
                                X_time_train[group][train_order[int(j*batch_size):int(j*batch_size + batch_size+1)]])
        
            loss = criterion(y_pred,
                            y_train[group][train_order[int(j*batch_size):int(j*batch_size + batch_size+1)]])

            # Comparing against withheld set
            prediction = model.forward(X_dist_test[group],
                                X_congest_test[group],
                                X_time_test[group])
        
            loss_test = criterion(prediction, y_test[group])

            if i % 10 == 0 and j == 0:
                print(f'Epoch: {i}')
                print(f'Train Loss: {loss}')
                print(f'Test Loss: {loss_test}')


            if j ==0:
                losses.append(loss.detach().numpy())
                losses_test.append(loss_test.detach().numpy())


            # Updating weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Graphs --> put graph of the losses from the withheld set to compare between the two (for cross validation)
    plt.figure(1)
    plt.plot(range(epochs),losses, 'blue', label = 'Training MSE')
    plt.plot(range(epochs),losses_test, 'red', label = 'Test MSE')
    plt.legend()
    plt.title('Sparsely connected NN, Training vs Testing MSE ')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    # plt.show()



if __name__ == "__main__":
    df = pd.read_csv("./final.csv")
    df = df.loc[:, "distance_gate":]
    arr = np.array(df)

    m, n = arr.shape
    df_test = arr[int(np.floor(m*0.2)):, :]
    labels_test = df_test[:, -1]
    features_test = df_test[:, :n-1]
    df_train = arr[:int(np.floor(m*0.8)), :]
    labels_train = df_train[:, -1]
    features_train = df_train[:, :n-1]

    ranges = np.array([[8, 16], [0.0001, 0.1], [8, 256], [2, 10], [1, 4]])
    dnn = DNN(ranges, features_train, features_test, labels_train, labels_test)
    dnn.FindAndTestOptimalNetwork(debug=True)
    dnn.PerformK_FoldTest(df)

    Sparse_NN()
   