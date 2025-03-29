import numpy as np
import pandas as pd
import optuna
import warnings
import json

from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.var import VAR
from sktime.forecasting.vecm import VECM
from sktime.forecasting.compose import make_reduction
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sktime.performance_metrics.forecasting import MeanSquaredError
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from src.utils.datetime_utils import ensure_datetime_index


class MultivariateForecaster:
    """
    An automated multivariate forecaster that uses an expanding window splitter 
    and Optuna to find the best model for forecasting multiple time series.
    """

    def __init__(
        self, 
        y: pd.DataFrame, 
        config_path: str = "config.json", 
        initial_window: int = 60, 
        step_length: int = 12, 
        fh: np.ndarray = np.arange(1, 13)
    ):
        """
        Initialize the MultivariateForecaster.

        Parameters:
        -----------
        y : pd.DataFrame
            The multivariate time series data to be forecasted, where each column is a different series.
        config_path : str
            Path to the configuration JSON file.
        initial_window : int
            The initial training window size.
        step_length : int
            The step length between each CV split.
        fh : np.ndarray
            Forecast horizon relative to the end of each training split.
        """
        self.y = ensure_datetime_index(y)
        self.cv = ExpandingWindowSplitter(
            fh=fh, 
            initial_window=initial_window, 
            step_length=step_length
        )
        self.study = None
        self.best_model = None

        # Load configuration from the JSON file
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Extract the list of models to include
        self.models_to_include = self.config.get("models", [])

    def create_model(self, trial: optuna.trial.Trial):
        """
        Create a multivariate forecasting model based on trial parameters suggested by Optuna.
        """
        # Use only the models specified in the "models" key of the config
        model_options = self.models_to_include
        if not model_options:
            raise ValueError("No models specified in the configuration file.")

        model_name = trial.suggest_categorical('model', model_options)
        model_params = self.config[model_name]

        model_constructors = {
            'RandomForestRegressor': lambda: make_reduction(
                RandomForestRegressor(
                    n_estimators=trial.suggest_int('n_estimators', *model_params['n_estimators']),
                    max_depth=trial.suggest_int('max_depth', *model_params['max_depth']),
                    min_samples_split=trial.suggest_int('min_samples_split', *model_params['min_samples_split']),
                    bootstrap=trial.suggest_categorical('bootstrap', model_params['bootstrap'])
                ),
                strategy="recursive"
            ),
            'KNeighborsRegressor': lambda: make_reduction(
                KNeighborsRegressor(
                    n_neighbors=trial.suggest_int('n_neighbors', *model_params['n_neighbors']),
                    weights=trial.suggest_categorical('weights', model_params['weights'])
                ),
                strategy="recursive"
            ),
            'XGBRegressor': lambda: make_reduction(
                XGBRegressor(
                    n_estimators=trial.suggest_int('n_estimators', *model_params['n_estimators']),
                    max_depth=trial.suggest_int('max_depth', *model_params['max_depth']),
                    learning_rate=trial.suggest_float('learning_rate', *model_params['learning_rate']),
                    subsample=trial.suggest_float('subsample', *model_params['subsample']),
                    colsample_bytree=trial.suggest_float('colsample_bytree', *model_params['colsample_bytree'])
                ),
                strategy="recursive"
            )
        }
        return model_constructors[model_name]()

    def evaluate_model(self, model) -> float:
        """
        Evaluate the model using ExpandingWindowSplitter and RMSE.

        Parameters:
        -----------
        model : forecasting model
            The multivariate forecasting model to be evaluated.

        Returns:
        --------
        float
            The mean RMSE of the model across all time series.
        """
        errors = []
        fh_range = np.arange(1, self.cv.fh[-1] + 1)
        for train_idx, test_idx in self.cv.split(self.y):
            try:
                model.fit(self.y.iloc[train_idx])
                y_pred = model.predict(fh_range[:len(test_idx)])
                error = MeanSquaredError(square_root=True, multioutput='uniform_average')
                error_value = error(self.y.iloc[test_idx], y_pred)
                errors.append(error_value)
            except Exception as e:
                print(f"Error in model evaluation: {str(e)}")
                errors.append(float('inf'))  # Penalize models that fail
        return np.mean(errors)

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Parameters:
        -----------
        trial : optuna.trial.Trial
            The trial object from Optuna.

        Returns:
        --------
        float
            The mean RMSE of the model.
        """
        model = self.create_model(trial)
        return self.evaluate_model(model)

    def optimize(self, n_trials: int = 50, n_jobs: int = -1) -> pd.DataFrame:
        """
        Optimize the model parameters using Optuna and return forecast from the 
        best model.

        Parameters:
        -----------
        n_trials : int
            The number of trials for optimization.
        n_jobs : int
            The number of jobs to run in parallel.

        Returns:
        --------
        pd.DataFrame
            The forecasted values from the best model.
        """
        self.study = optuna.create_study(
            direction='minimize', 
            sampler=TPESampler(), 
            pruner=MedianPruner()
        )
        try:
            self.study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)
        except Exception as e:
            print(f"⚠ Optimization failed: {str(e)}")
            raise RuntimeError("Optimization process failed. Check the data and model configurations.")

        if self.study.best_trial is None:
            raise RuntimeError("No valid model found during optimization.")

        self.train_best_model()
        return self.forecast()

    def train_best_model(self):
        """
        Retrain the best model found from optimization on the full dataset.
        """
        if self.study is None or self.study.best_trial is None:
            raise ValueError("No optimization study or best trial found. Run optimize() first.")

        best_params = self.study.best_trial.params
        model_name = best_params.get('model')
        if model_name is None:
            raise ValueError("Best trial parameters are invalid or incomplete.")

        # Create the best model based on the best parameters
        model_params = self.config[model_name]
        if model_name == 'RandomForestRegressor':
            self.best_model = make_reduction(
                RandomForestRegressor(
                    n_estimators=best_params['n_estimators'],
                    max_depth=best_params['max_depth'],
                    min_samples_split=best_params['min_samples_split'],
                    bootstrap=best_params['bootstrap']
                ),
                strategy="recursive"
            )
        elif model_name == 'KNeighborsRegressor':
            self.best_model = make_reduction(
                KNeighborsRegressor(
                    n_neighbors=best_params['n_neighbors'],
                    weights=best_params['weights']
                ),
                strategy="recursive"
            )
        elif model_name == 'XGBRegressor':
            self.best_model = make_reduction(
                XGBRegressor(
                    n_estimators=best_params['n_estimators'],
                    max_depth=best_params['max_depth'],
                    learning_rate=best_params['learning_rate'],
                    subsample=best_params['subsample'],
                    colsample_bytree=best_params['colsample_bytree']
                ),
                strategy="recursive"
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Fit the best model on full dataset
        self.best_model.fit(self.y)

    def forecast(self, fh: np.ndarray = np.arange(1, 13)) -> pd.DataFrame:
        """
        Generate forecasts from the trained best model.

        Parameters:
        -----------
        fh : np.ndarray
            Forecast horizon relative to the end of the training data.

        Returns:
        --------
        pd.DataFrame
            The forecasted values for all time series.
        """
        if self.best_model is None:
            raise ValueError("No best model trained. Run optimize() first.")
        
        # Use the entire dataset for forecasting
        return self.best_model.predict(fh, X=self.y)