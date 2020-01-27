import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import random

def flatten_list(X):
    n_col = np.size(X[0],1)
    Y = np.empty((0, n_col))
    for each in X:
        Y = np.vstack((Y, each))
    return Y

def form_decoder_dataset(spike, y, N):
    spike_N_lag = []
    emg_N_lag = []
    for i in range(N-1, np.size(spike, 0)-N):
        temp = spike[i-N+1:i+1, :]
        temp = temp.reshape((np.size(temp)))
        spike_N_lag.append(temp)
        emg_N_lag.append(y[i, :])
    return np.asarray(spike_N_lag), np.asarray(emg_N_lag)

def w_filter_train(spike, y, c=0):
    spike_plus_bias = np.c_[np.ones((np.size(spike, 0), 1)), spike]
    H_all = w_filter_fit(spike_plus_bias, y, c)
    return H_all

def w_filter_test(spike, H_all):
    spike_plus_bias = np.c_[np.ones((np.size(spike, 0), 1)), spike]
    y_pred = np.dot(spike_plus_bias, H_all)
    return y_pred

def w_filter_fit(X,Y, c):
    # c : L2 regularization coefficient
    # I : Identity Matrix
    # Linear Least Squares (code defaults to this if c is not passed)
    #   H = ( X^T * X )^-1 * X^T * Y
    # Ridge Regression
    #   R = c * I
    #   ridge regression doesn't penalize x
    #   R[0,0] = 0
    #   H = ( (X^T * X) + R )^-1 * X^T * Y
    X = X+np.random.uniform(0.0001,0.0002,(X.shape[0],X.shape[1]))
    R = c * np.eye( X.shape[1] )
    # To do 
    R[0,0] = 0;
    
    temp_before_inv = np.dot(X.T,X) + R
    shape_temp_before_inv = temp_before_inv.shape
    rand_matrix = np.random.uniform(0.0001,0.0002,shape_temp_before_inv)
    #np.random.rand(int(shape_temp_before_inv[0]),int(shape_temp_before_inv[1]))*0.001
    #print(rand_matrix[1,:])
    
    temp = np.linalg.inv(temp_before_inv)
    #获得np.dot(X.T, X) + R的shape
    #用np.rand.randn((shape,shape))*0.001
    temp2 = np.dot(temp,X.T)
    H = np.dot(temp2,Y)
    return H

def reg_sweep( flat_x, flat_y, C, kf ):
    reg_r2 = []
    print ('Sweeping ridge regularization using CV decoding on train data' )
    for c in C:
        print( 'Testing c= ' + str(c) )
        cv_r2 = []
        for train_indices, test_indices in kf.split(flat_x):
            # split data into train and test
            train_x, test_x = flat_x[train_indices,:], flat_x[test_indices,:]
            train_y, test_y = flat_y[train_indices,:], flat_y[test_indices,:]
            # fit decoder
            H = w_filter_train(train_x, train_y, c)
            #print( H.shape )
            # predict
            test_y_pred = w_filter_test(test_x, H)
            # evaluate performance
            cv_r2.append(r2_score(test_y, test_y_pred, multioutput='raw_values'))
        # append mean of CV decoding for output
        cv_r2 = np.asarray(cv_r2)
        reg_r2.append( np.mean( cv_r2, axis=0 ) )

    reg_r2 = np.asarray(reg_r2)        
    reg_r2 = np.mean( reg_r2, axis=1 )
    best_c = C[ np.argmax( reg_r2 ) ] 
    return best_c

def wiener_nonlinear(p, y):
    return p[0]+p[1]*y+p[2]*y*y
    
def wiener_nonlinear_res(p, y, z):
    return (wiener_nonlinear(p, y) - z).reshape((-1,))

from scipy.optimize import least_squares
def wiener_cascade_train(x, y, l2 = 0):
    if l2 == 1:
        n_l2 = 20
        C = np.logspace( 1, 5, n_l2 )
        kfolds = 4
        kf = KFold( n_splits = kfolds )
        best_c = reg_sweep( x, y, C, kf )
    else:
        best_c = 0
    H_reg = w_filter_train( x, y, best_c )
    y_pred = w_filter_test(x, H_reg)
    res_lsq = least_squares(wiener_nonlinear_res, [0.1,0.1,0.1], args = (y_pred, y))
    return H_reg, res_lsq
    
def wiener_only_train(x, y, l2 = 0):
    if l2 == 1:
        n_l2 = 20
        C = np.logspace( 1, 5, n_l2 )
        kfolds = 4
        kf = KFold( n_splits = kfolds )
        best_c = reg_sweep( x, y, C, kf )
    else:
        best_c = 0
    H_reg = w_filter_train( x, y, best_c )
    return H_reg

def wiener_cascade_test(x, H_reg, res_lsq):    
    y1 = w_filter_test(x, H_reg)
    y2 = wiener_nonlinear(res_lsq.x, y1)
    return y2
    
def vaf(x,xhat):
   x = x - x.mean(axis=0)
   xhat = xhat - xhat.mean(axis=0)
   return (1-(np.sum(np.square(x - xhat))/np.sum(np.square(x))))*100    
    
    
    
    
    
    


