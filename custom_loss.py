'''
DESCRIPTION
-----------
This script contains the implementation of the average euclidean distance loss function used to train predective ML models 
that produces K estimations for the target vector. 
'''


#%% code imports
import math
import numpy as np
import tensorflow as tf


#%% functions definition
def average_euclidean_distance(y_pred, y_true):
    """function to calculate the average euclidean distance between K estimates of target and actual target

    Parameters
    ----------
    y_pred : ndarray
        3D tensor of shape (m, k, o) where [m] is the batch size, [k] is the number of estimators, [o] is the output size 
    y_true : ndarray
        2D tensor of shape (m, o) where [m] is the batch size and [o] is the output size

    Returns
    -------
    loss : ndarray
        vector of the averages of euclidean distances between estimatros of target and the actual target for a batch of size m
      
    """
    y_true = tf.reshape(y_true, (y_true.shape[0], 1, y_true.shape[1]))
    diff_sqaured = tf.math.square(tf.math.subtract(y_pred, y_true))
    euclidean_distance = tf.math.sqrt(tf.reduce_sum(diff_sqaured, axis=2))
    loss = tf.reduce_mean(euclidean_distance, axis=1)
    return loss


#%% function testing
if(__name__=='__main__'):
    # start a tf session
    s = tf.Session()

    # test with m=1, k=5 and o=4
    y_pred = np.ones((1, 5, 4))
    y_true = np.ones((1, 4))*2
    loss = average_euclidean_distance(y_pred, y_true)
    assert list(s.run(loss))==[2], 'test_case_1_failed'

    # test with m=3, k=5 and o=4
    y_pred = np.ones((3, 5, 4))
    y_true = np.ones((3, 4))*2
    loss = average_euclidean_distance(y_pred, y_true)
    assert list(s.run(loss))==[2, 2, 2], 'test_case_2_failed'

    # test with m=1, k=3 and o=3
    y_pred = np.zeros((1, 3, 3), dtype=np.float64)
    y_pred[0][0] = [7, 2, 1]
    y_pred[0][1] = [3, 4, 5]
    y_pred[0][2] = [2, 6, 8]
    y_true = np.array([[9, 10, 2]], dtype=np.float64)
    loss = average_euclidean_distance(y_pred, y_true)
    assert math.isclose(list(s.run(loss))[0], 9.1188, abs_tol=0.01), 'test_case_3_failed' 

    # close tf session
    s.close()

