'''
DESCRIPTION
-----------
This script contains an example for using the custom loss fnction on a test neural network.
'''


# %% code imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
#from custom_loss import average_euclidean_distance


# %% main code section
if(__name__ == '__main__'):
    input_size = 3  # dimensions of input vector
    output_size = 3  # dimensions of output vector
    k_estimators = 5  # number of estimators for output
    batch_size = 32  # number of examples per each update step

    # create random data for training ad testing
    x_train, x_test = np.random.rand(100, input_size), np.random.rand(10, input_size)
    y_train, y_test = np.random.rand(100, output_size), np.random.rand(10, output_size)

    # build model
    main_input = Input(shape=(input_size,))
    neural_net = Dense(5, activation='relu')(main_input)
    neural_net = Dense(5, activation='relu')(neural_net)
    neural_net_outputs = [Dense(output_size, activation='softmax')(neural_net) for _ in range(k_estimators)] 
    model = tf.keras.models.Model(inputs=main_input,
                                  outputs=neural_net_outputs)

    # compile model using our custom made loss function 
    #model.compile('Adam', loss='#')
    
    # train 
    #model.fit(x_train, y_train, epochs=5)
    
    # validate
    #model.evaluate(x_test, y_test)
