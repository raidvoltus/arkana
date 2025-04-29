# model_utils.py

import optuna
import logging
import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# === LightGBM Trainer ===
def train_lightgbm(X: pd.DataFrame, y: pd.Series) -> LGBMRegressor:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                  early_stopping_rounds=20, verbose=False)
        preds = model.predict(X_valid)
        return mean_squared_error(y_valid, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    best_params = study.best_params
    logging.info(f"Best LightGBM params: {best_params}")

    model = LGBMRegressor(**best_params)
    model.fit(X, y)
    return model
    
# === XGBoost Trainer ===
def train_xgboost(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
            'random_state': 42
        }

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                  early_stopping_rounds=20, verbose=False)
        preds = model.predict(X_valid)
        return mean_squared_error(y_valid, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    best_params = study.best_params
    logging.info(f"Best XGBoost params: {best_params}")

    model = XGBRegressor(**best_params)
    model.fit(X, y)
    return model
    
# === LSTM Trainer ===
def train_lstm(X: pd.DataFrame, y: pd.Series) -> Sequential:
    def objective(trial):
        units = trial.suggest_int("units", 32, 128, step=32)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs = trial.suggest_int("epochs", 10, 30)

        X_np = X.values
        y_np = y.values
        X_seq = X_np.reshape((X_np.shape[0], 1, X_np.shape[1]))  # (samples, timesteps, features)

        X_train, X_val, y_train, y_val = train_test_split(X_seq, y_np, test_size=0.2, shuffle=False)

        model = Sequential()
        model.add(Input(shape=(X_seq.shape[1], X_seq.shape[2])))
        model.add(LSTM(units, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=epochs, batch_size=batch_size, verbose=0)

        val_loss = history.history["val_loss"][-1]
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    best_params = study.best_params
    logging.info(f"Best LSTM params: {best_params}")

    # Final training with best parameters
    X_np = X.values
    y_np = y.values
    X_seq = X_np.reshape((X_np.shape[0], 1, X_np.shape[1]))

    model = Sequential()
    model.add(Input(shape=(X_seq.shape[1], X_seq.shape[2])))
    model.add(LSTM(best_params["units"], return_sequences=False))
    model.add(Dropout(best_params["dropout"]))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=best_params["learning_rate"]), loss='mse')

    model.fit(X_seq, y_np, epochs=best_params["epochs"],
              batch_size=best_params["batch_size"], verbose=0)

    return model