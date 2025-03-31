import numpy as np
import pandas as pd
import optuna
import warnings
import json

from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.var import VAR
from sktime.forecasting.vecm import VECM
from sktime.forecasting.compose import make_reduction
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sktime.performance_metrics.forecasting import MeanSquaredError
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from src.utils.datetime_utils import ensure_datetime_index
from sklearn.preprocessing import OneHotEncoder


class MultivariateForecaster:
    """
    An automated multivariate forecaster that uses an expanding window splitter 
    and Optuna to find the best model for forecasting multiple time series.
    """

    def __init__(
        self, 
        data: pd.DataFrame, 
        target_columns: list, 
        config_path: str = "config.json", 
        initial_window: int = 60, 
        step_length: int = 12, 
        fh: np.ndarray = np.arange(1, 13)
    ):
        """
        Initialize the MultivariateForecaster.

        Parameters:
        -----------
        data : pd.DataFrame
            The input DataFrame containing both target variables and exogenous variables.
        target_columns : list
            List of column names to be used as target variables (`y`).
            All other columns will be treated as exogenous variables (`X`).
        config_path : str
            Path to the configuration JSON file.
        initial_window : int
            The initial training window size.
        step_length : int
            The step length between each CV split.
        fh : np.ndarray
            Forecast horizon relative to the end of each training split.
        """
        # Ensure the DataFrame has a datetime index
        data = ensure_datetime_index(data)

        # Split the data into y (targets) and X (exogenous variables)
        self.y = data[target_columns]
        self.X_original = data.drop(columns=target_columns) if len(data.columns) > len(target_columns) else None

        # Initialize encoder and encoded X
        self.encoder = None
        self.categorical_columns = []
        self.X_encoded = None
        
        # Process exogenous variables if they exist
        if self.X_original is not None:
            # Identify categorical columns
            self.categorical_columns = self.X_original.select_dtypes(include=['object', 'category']).columns.tolist()
            # Create and fit the encoder if categorical columns exist
            if len(self.categorical_columns) > 0:
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                # Process X to get encoded version
                self.X_encoded = self._encode_categorical_features(self.X_original)
            else:
                # No categorical columns, X_encoded is the same as X_original
                self.X_encoded = self.X_original.copy()

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

    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features in X using the fitted OneHotEncoder.
        
        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame containing the features to encode
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with categorical features encoded
        """
        if not self.categorical_columns:
            return X.copy()
            
        # Validate that all categorical columns are present
        missing_columns = [col for col in self.categorical_columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Categorical columns missing from X: {missing_columns}")
        
        # If encoder not yet fitted, fit it
        if not hasattr(self.encoder, 'feature_names_in_'):
            self.encoder.fit(X[self.categorical_columns])
        
        # Transform the categorical columns
        encoded = self.encoder.transform(X[self.categorical_columns])
        encoded_df = pd.DataFrame(
            encoded,
            index=X.index,
            columns=self.encoder.get_feature_names_out(self.categorical_columns)
        )
        
        # Combine with non-categorical columns
        result = pd.concat([
            X.drop(columns=self.categorical_columns), 
            encoded_df
        ], axis=1)
        
        return result

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
                # Extract and process training and testing data
                if self.X_encoded is not None:
                    X_train = self.X_encoded.iloc[train_idx]
                    X_test = self.X_encoded.iloc[test_idx]
                else:
                    X_train = None
                    X_test = None
                    
                # Fit the model and make predictions
                model.fit(self.y.iloc[train_idx], X=X_train)
                y_pred = model.predict(fh_range[:len(test_idx)], X=X_test)
                
                # Calculate error
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

        # Fit the best model on full dataset using encoded features
        self.best_model.fit(self.y, X=self.X_encoded)

    def forecast(self, fh: np.ndarray = np.arange(1, 13), X_future: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate forecasts from the trained best model.

        Parameters:
        -----------
        fh : np.ndarray
            Forecast horizon relative to the end of the training data.
        X_future : pd.DataFrame, optional
            Future values of exogenous variables. If not provided, the training X will be used.

        Returns:
        --------
        pd.DataFrame
            The forecasted values for all time series.
        """
        if self.best_model is None:
            raise ValueError("No best model trained. Run optimize() first.")
        
        # Process future exogenous variables if provided
        if X_future is not None and self.X_original is not None:
            # Encode the future exogenous variables
            X_future_encoded = self._encode_categorical_features(X_future)
            return self.best_model.predict(fh, X=X_future_encoded)
        
        # Use the training data for forecasting if no future X provided
        return self.best_model.predict(fh, X=self.X_encoded)