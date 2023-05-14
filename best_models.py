from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

import numpy as np

from sklearn.linear_model import LinearRegression 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def lr_model(X_train, y_train, X_val, y_val, X_test, y_test):
    lr_model = LinearRegression()
    lr_model = MultiOutputRegressor(lr_model)
    X_train_val = np.concatenate((X_train, X_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    # fit the model on the training and validation data
    lr_model.fit(X_train_val, y_train_val)

    # make predictions on the test data, loop through it one by one and each 720 predictions retrain the model
    y_pred = []
    for i in range(len(X_test)):
        # make prediction
        yhat = lr_model.predict(X_test.iloc[i].values.reshape(1, -1))
        # store prediction
        y_pred.append(yhat)
        # retrain the model at the end of each 30 days with the new data
        if (i + 1) % 720 == 0:
            print('Retraining model...')
            print('Progress =', i + 1, '/', len(X_test))
            X_train_val = np.concatenate((X_train_val, X_test.iloc[i-719:i+1].values), axis=0)
            y_train_val = np.concatenate((y_train_val, y_test.iloc[i-719:i+1].values), axis=0)
            print('X_train_val shape:', X_train_val.shape)
            print('y_train_val shape:', y_train_val.shape)
            lr_model.fit(X_train_val, y_train_val)

    return y_pred

def best_rf_model(X_train, y_train, X_val, y_val, X_test, y_test):
    # Instantiate the model WITH Best n_estimators: 500 Best max_depth: 10 Best min_samples_split: 10
    rf = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=10, random_state=42)
    #MultiOutputRegressor
    multi_rf = MultiOutputRegressor(rf)

    X_train_val = np.concatenate((X_train, X_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    # fit the model on the training and validation data
    multi_rf.fit(X_train_val, y_train_val)

    # make predictions on the test data, loop through it one by one and each 720 predictions retrain the model
    y_pred = []
    for i in range(len(X_test)):
        # make prediction
        yhat = multi_rf.predict(X_test.iloc[i].values.reshape(1, -1))
        # store prediction
        y_pred.append(yhat)
        # retrain the model at the end of each 30 days with the new data
        if (i + 1) % 720 == 0:
            print('Retraining model...')
            print('Progress =', i + 1, '/', len(X_test))
            X_train_val = np.concatenate((X_train_val, X_test.iloc[i-719:i+1].values), axis=0)
            y_train_val = np.concatenate((y_train_val, y_test.iloc[i-719:i+1].values), axis=0)
            print(len(X_train_val), len(y_train_val))
            multi_rf.fit(X_train_val, y_train_val)

    return y_pred

def best_svr_model(X_train, y_train, X_val, y_val, X_test, y_test):
    # Instantiate the model WITH Best C: 100000 Best gamma: Scale Best kernel: rbf Best epsilon: 0.1
    svr = SVR(C=100000, gamma='scale', kernel='rbf', epsilon=0.1)
    #MultiOutputRegressor
    multi_svr = MultiOutputRegressor(svr)

    X_train_val = np.concatenate((X_train, X_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    # fit the model on the training and validation data
    multi_svr.fit(X_train_val, y_train_val)
    
    # make predictions on the test data, loop through it one by one and each 720 predictions retrain the model
    y_pred = []
    for i in range(len(X_test)):
        # make prediction
        yhat = multi_svr.predict(X_test.iloc[i].values.reshape(1, -1))
        # store prediction
        y_pred.append(yhat)
        # retrain the model at the end of each 30 days with the new data
        if (i + 1) % 720 == 0:
            print('Retraining model...')
            print('Progress =', i + 1, '/', len(X_test))
            X_train_val = np.concatenate((X_train_val, X_test.iloc[i-719:i+1].values), axis=0)
            y_train_val = np.concatenate((y_train_val, y_test.iloc[i-719:i+1].values), axis=0)
            print(len(X_train_val), len(y_train_val))
            multi_svr.fit(X_train_val, y_train_val)

    return y_pred

def best_lstm_model(X_train, y_train, X_val, y_val, X_test, y_test):
    # transform the data to numpy arrays
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_val_np = np.array(X_val)
    y_val_np = np.array(y_val)

    # Define the LSTM model with best parameters
    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape=(X_train_np.shape[1], 1)))
    model.add(Dense(2))

    # Compile the model
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='mse')

    # Reshape the training and validation data for LSTM input
    X_train_reshaped = X_train_np.reshape((X_train_np.shape[0], X_train_np.shape[1], 1))
    X_val_reshaped = X_val_np.reshape((X_val_np.shape[0], X_val_np.shape[1], 1))

    # Train the model
    model.fit(X_train_reshaped, y_train_np, epochs=20, batch_size=32, validation_data=(X_val_reshaped, y_val_np))

    # transform the test data to numpy arrays
    X_test_np = np.array(X_test)

    # Reshape the test data for LSTM input
    X_test_reshaped = X_test_np.reshape((X_test_np.shape[0], X_test_np.shape[1], 1))

    # make predictions on the test data, loop through it one by one and each 720 predictions retrain the model
    y_pred = []
    for i in range(len(X_test)):
        # make prediction
        yhat = model.predict(X_test_reshaped[i].reshape(1, X_test_reshaped.shape[1], 1))
        # store prediction
        y_pred.append(yhat)
        # retrain the model at the end of each 30 days with the new data
        if (i + 1) % 720 == 0:
            print('Retraining model...')
            print('Progress =', i + 1, '/', len(X_test))
            X_train_reshaped = np.concatenate((X_train_reshaped, X_test_reshaped[i-719:i+1]), axis=0)
            y_train_np = np.concatenate((y_train_np, y_test.iloc[i-719:i+1].values), axis=0)
            print(len(X_train_reshaped), len(y_train_np))
            model.fit(X_train_reshaped, y_train_np, epochs=20, batch_size=32, validation_data=(X_val_reshaped, y_val_np))


    return y_pred

def oos_mse_score():
    '''
    This function will return the out-of-sample MSE score for each model.
    '''
    models = ['MA', 
              'Linear Regression', 
              'Random Forest', 
              'SVR', 
              'LSTM'
              ]
    high_mse = [1943724.378019002,
                1505.6940446134586,
                0,
                0,
                0
            ]
    low_mse = [1882402.1947497907,
               1695.6264541153896,
               0,
               0,
               0
               ]
