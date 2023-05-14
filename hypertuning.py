import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def finetuning_rf(X_train, y_train, X_val, y_val, n_estimators, max_depth, min_samples_split):
    """
    This function takes in a rf model and hyperparameters and returns the best model with the hyperparameters set based on R squared score.
    """
    # R-squared score, start with - infinity
    best_r2 = -np.inf

    best_n_estimators = None
    best_max_depth = None
    best_min_samples_split = None

    # Save the r squared score for each model in a dictionary
    r2_dict = {}

    total_combinations = len(n_estimators) * len(max_depth) * len(min_samples_split)

    # loop through all the hyperparameter combinations
    for estimator in n_estimators:
        for depth in max_depth:
            for split in min_samples_split:
                # Instantiate the model
                rf = RandomForestRegressor(n_estimators=estimator, max_depth=depth, min_samples_split=split, random_state=42)
                #MultiOutputRegressor
                multi_rf = MultiOutputRegressor(rf)

                # Fit the model on the training data
                multi_rf.fit(X_train, y_train)

                # Make predictions on the validation data
                y_pred = multi_rf.predict(X_val)

                # Calculate the R-squared score
                r2 = r2_score(y_val, y_pred)

                # Print the hyperparameters and R-squared score
                print('n_estimators:', estimator, 'max_depth:', depth, 'min_samples_split:', split, 'r2:', r2)
                print('Progress:', (len(r2_dict) + 1) / total_combinations * 100, '%')
                # Save the R-squared score for each model
                r2_dict[(estimator, depth, split)] = r2

                # If the R-squared score is greater than the previous best score
                if r2 > best_r2:
                    # Save the new best score
                    best_r2 = r2
                    # Save the new best model
                    best_model = multi_rf
                    # Save the best hyperparameters
                    best_n_estimators = estimator
                    best_max_depth = depth
                    best_min_samples_split = split

    # Print the best hyperparameters
    print("Best n_estimators:", best_n_estimators)
    print("Best max_depth:", best_max_depth)
    print("Best min_samples_split:", best_min_samples_split)
    print("Best r2:", best_r2)

    return r2_dict

def finetuning_svr(X_train, y_train, X_val, y_val, c_values, kernel_values, gamma_values, epsilon_values):
    """
    This function takes in a svr model and hyperparameters and returns the best model with the hyperparameters set based on R squared score.
    """
    # R-squared score, start with - infinity
    best_r2 = -np.inf

    best_c = None
    best_kernel = None
    best_gamma = None
    best_epsilon = None

    # Save the r squared score for each model in a dictionary
    r2_dict = {}

    total_combinations = len(c_values) * len(kernel_values) * len(gamma_values) * len(epsilon_values)

    # loop through all the hyperparameter combinations
    for c in c_values:
        for kernel in kernel_values:
            for gamma in gamma_values:
                for epsilon in epsilon_values:
                    # Instantiate the model
                    svr = SVR(C=c, kernel=kernel, gamma=gamma, epsilon=epsilon)
                    #MultiOutputRegressor
                    multi_svr = MultiOutputRegressor(svr)

                    # Fit the model on the training data
                    multi_svr.fit(X_train, y_train)

                    # Make predictions on the validation data
                    y_pred = multi_svr.predict(X_val)

                    # Calculate the R-squared score
                    r2 = r2_score(y_val, y_pred)

                    # Print the hyperparameters and R-squared score
                    print('c:', c, 'kernel:', kernel, 'gamma:', gamma, 'epsilon:', epsilon, 'r2:', r2)
                    print('Progress:', (len(r2_dict) + 1) / total_combinations * 100, '%')

                    # Save the R-squared score for each model
                    r2_dict[(c, kernel, gamma, epsilon)] = r2

                    # If the R-squared score is greater than the previous best score
                    if r2 > best_r2:
                        # Save the new best score
                        best_r2 = r2
                        # Save the new best model
                        best_model = multi_svr
                        # Save the best hyperparameters
                        best_c = c
                        best_kernel = kernel
                        best_gamma = gamma
                        best_epsilon = epsilon

    # Print the best hyperparameters
    print("Best c:", best_c)
    print("Best kernel:", best_kernel)
    print("Best gamma:", best_gamma)
    print("Best epsilon:", best_epsilon)
    print("Best r2:", best_r2)

    return r2_dict

num_units_array = [32, 64, 128, 256]
learning_rate_array = [0.001, 0.01, 0.1, 1.0]
batch_size_array = [16, 32, 64, 128, 256, 512]
epochs_array = [20, 40, 60, 80, 100]
activation_array = ['relu', 'tanh', 'sigmoid']

def finetuning_lstm(X_train, y_train, X_val, y_val, lstm_units, learning_rate, batch_size, epochs, activations):
    """
    This function takes in a LSTM model and hyperparameters and returns the best model with the hyperparameters set based on R squared score.
    """
    # R-squared score, start with - infinity
    best_r2 = -np.inf

    best_lstm_units = None
    best_learning_rate = None
    best_activation = None
    best_epochs = None
    best_batch_size = None

    # Save the r squared score for each model in a dictionary
    r2_dict = {}

    # transform the data to numpy arrays
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_val_np = np.array(X_val)
    y_val_np = np.array(y_val)


    total_combinations = len(lstm_units) * len(learning_rate) * len(batch_size) * len(epochs) * len(activations)

    # loop through all the hyperparameter combinations
    for lstm_unit in lstm_units:
        for epoch in epochs:
            for batch in batch_size:
                for activation in activations:
                    for learning in learning_rate:
                        # Define the LSTM model
                        model = Sequential()
                        model.add(LSTM(lstm_unit, activation=activation, input_shape=(X_train_np.shape[1], 1)))
                        model.add(Dense(2))

                        # Compile the model
                        adam = Adam(lr=learning)
                        model.compile(optimizer=adam, loss='mse')

                        # Reshape the training and validation data for LSTM input
                        X_train_reshaped = X_train_np.reshape((X_train_np.shape[0], X_train_np.shape[1], 1))
                        X_val_reshaped = X_val_np.reshape((X_val_np.shape[0], X_val_np.shape[1], 1))

                        # Train the model
                        model.fit(X_train_reshaped, y_train_np, epochs=epoch, batch_size=batch,validation_data=(X_val_reshaped, y_val_np))

                        # Make predictions on the validation data
                        y_pred = model.predict(X_val_reshaped)

                        # Calculate the R-squared score
                        r2 = r2_score(y_val_np, y_pred)


                        # Print the hyperparameters and R-squared score
                        print('lstm_units:', lstm_unit, 'epochs:', epoch, 'batch_size:', batch, 'learning_rate:', learning, 'activation:', activation, 'r2:', r2)
                        print('Progress:', (len(r2_dict) + 1) / total_combinations * 100, '%')

                        # Save the R-squared score for each model
                        r2_dict[(lstm_unit, epoch, batch, learning, activation)] = r2

                        # If the R-squared score is greater than the previous best score
                        if r2 > best_r2:
                            # Save the new best score
                            best_r2 = r2
                            # Save the new best model
                            best_model = model
                            # Save the best hyperparameters
                            best_lstm_units = lstm_unit
                            best_epochs = epoch
                            best_batch_size = batch
                            best_learning_rate = learning
                            best_activation = activation

    # Print the best hyperparameters
    print("Best lstm_units:", best_lstm_units)
    print("Best learning_rate:", best_learning_rate)
    print("Best activation:", best_activation)
    print("Best epochs:", best_epochs)
    print("Best batch_size:", best_batch_size)
    print("Best r2:", best_r2)

    return r2_dict
