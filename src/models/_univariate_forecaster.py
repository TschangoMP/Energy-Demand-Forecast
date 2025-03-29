import numpy as np
import pandas as pd
import optuna
import warnings
import time

from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.performance_metrics.forecasting import MeanSquaredError
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from src.utils.datetime_utils import ensure_datetime_index

# Filter specific warnings
warnings.filterwarnings("ignore", message="No frequency information was provided")
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters")


class UnivariateForecaster:
    """
    An automated univariate forecaster that uses an expanding window splitter 
    and Optuna to find the best model.
    """

    def __init__(
        self, 
        y: pd.Series, 
        initial_window: int = 60, 
        step_length: int = 12, 
        fh: np.ndarray = np.arange(1, 13)
    ):
        """
        Initialize the UnivariateForecaster.

        Parameters:
        -----------
        y : pd.Series
            The time series data to be forecasted.
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
        self.tested_params = set()  # Track tested parameter combinations

    def create_model(self, trial: optuna.trial.Trial):
        """
        Create a model based on trial parameters suggested by Optuna.
        
        Parameters:
        -----------
        trial : optuna.trial.Trial
            The trial object from Optuna.
        
        Returns:
        --------
        model : sktime.forecasting.base.BaseForecaster
            The forecasting model.
        """
        has_non_positive = np.any(self.y <= 0)
    
        # Exclude ThetaForecaster if data contains non-positive values
        model_options = [
            'AutoARIMA', 'NaiveForecaster', 'PolynomialTrendForecaster', 
            'ExponentialSmoothing'
        ]
        
        if not has_non_positive:
            # Only include ThetaForecaster for strictly positive data
            model_options.append(['ThetaForecaster', 'TBATS'])
        
        model_name = trial.suggest_categorical('model', model_options)
        
        freq = self.y.index.freq
        if freq is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                freq = pd.infer_freq(self.y.index)
        if freq is None:
            print("⚠ Warning: No frequency detected. Defaulting to 'M'.")
            freq = 'M'
        
        try:
            sp_value = pd.tseries.frequencies.to_offset(freq).n
            if sp_value is None or sp_value <= 0:
                raise ValueError("Invalid seasonal period")
        except (ValueError, AttributeError):
            sp_value = 12
            print(f"⚠ Warning: Couldn't infer seasonal period. Defaulting to {sp_value}.")
              
        model_constructors = {
            'AutoARIMA': lambda: AutoARIMA(
                start_p=trial.suggest_int('start_p', 0, 3),
                d=trial.suggest_int('d', 0, 2),
                start_q=trial.suggest_int('start_q', 0, 3),
                seasonal=True, max_p=3, max_q=3, stepwise=True
            ),
            'ThetaForecaster': lambda: ThetaForecaster(
                sp=sp_value
            ),
            'NaiveForecaster': lambda: NaiveForecaster(
                strategy=trial.suggest_categorical(
                    'strategy', ['last', 'mean', 'drift']
                )
            ),
            'PolynomialTrendForecaster': lambda: PolynomialTrendForecaster(
                degree=trial.suggest_int('degree', 1, 4)
            ),
            'TBATS': lambda: TBATS(
                use_box_cox=trial.suggest_categorical(
                    'use_box_cox', [True, False]
                ),
                use_trend=trial.suggest_categorical(
                    'use_trend', [True, False]
                ),
                use_damped_trend=trial.suggest_categorical(
                    'use_damped_trend', [True, False]
                ),
                sp=sp_value if sp_value > 1 else None
            ),
            'ExponentialSmoothing': lambda: ExponentialSmoothing(
                trend=trial.suggest_categorical(
                    'trend', 
                    ['add', None] if has_non_positive else ['add', 'mul', None]
                ),
                seasonal=trial.suggest_categorical(
                    'seasonal', 
                    ['add', None] if has_non_positive else ['add', 'mul', None]
                ),
                sp=sp_value if sp_value > 1 else None
            )
        }
        return model_constructors[model_name]()

    def evaluate_model(self, model, trial) -> float:
        """
        Evaluate the model using ExpandingWindowSplitter and mean squared error (MSE).

        Parameters:
        -----------
        model : sktime.forecasting.base.BaseForecaster
            The forecasting model to be evaluated.
        trial : optuna.trial.Trial
            The trial object from Optuna.

        Returns:
        --------
        float
            The mean squared error of the model.
        """
        errors = []
        fh_range = np.arange(1, self.cv.fh[-1] + 1)
        start_time = time.time()  # Track the start time of the trial
        max_trial_time = 300  # Set a maximum time for the trial in seconds (e.g., 5 minutes)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for train_idx, test_idx in self.cv.split(self.y):
                model.fit(self.y.iloc[train_idx])
                y_pred = model.predict(fh_range[:len(test_idx)])
                error = MeanSquaredError(square_root=True)
                error_value = error(self.y.iloc[test_idx], y_pred)
                errors.append(error_value)

                # Report intermediate value
                if len(errors) > 0:
                    trial.report(np.mean(errors), step=len(errors))

                # Check if the trial should be pruned
                if trial.should_prune():
                    print("Pruning triggered for trial:", trial.number)
                    raise optuna.exceptions.TrialPruned()

                # Abort the trial if it exceeds the maximum allowed time
                elapsed_time = time.time() - start_time
                if elapsed_time > max_trial_time:
                    print(f"Trial {trial.number} aborted due to timeout ({elapsed_time:.2f}s).")
                    raise optuna.exceptions.TrialPruned()

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
            The mean absolute percentage error of the model.
        """
        params = trial.params
        params_tuple = tuple(sorted(params.items()))  # Convert params to a hashable form
        self.tested_params.add(params_tuple)
        model = self.create_model(trial)
        return self.evaluate_model(model, trial)

    def optimize(self, n_trials: int = 50, n_jobs: int = -1) -> pd.Series:
        """
        Optimize the model parameters using Optuna and return forecast from the 
        best model.
        """
        self.study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(),
            pruner=MedianPruner()  # Explicitly disable pruning
        )
        print(f"Using pruner: {self.study.pruner}")  # Debugging: Log the active pruner

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

        # Get frequency with better error handling
        freq = self.y.index.freq
        if freq is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                freq = pd.infer_freq(self.y.index)
        if freq is None:
            print("⚠ Warning: No frequency detected. Defaulting to 'M'.")
            freq = 'M'

        # Get the seasonal period value safely
        try:
            sp_value = pd.tseries.frequencies.to_offset(freq).n
            if sp_value is None or sp_value <= 0:
                raise ValueError("Invalid seasonal period")
        except (ValueError, AttributeError):
            sp_value = 12  # Assume monthly seasonality as a safe fallback
            print(f"⚠ Warning: Couldn't infer seasonal period. Defaulting to {sp_value}.")

        # Adjust seasonality for models that support it
        seasonal_param = best_params['seasonal'] if sp_value > 1 else None

        model_constructors = {
            'AutoARIMA': lambda: AutoARIMA(
                start_p=best_params['start_p'], 
                d=best_params['d'], 
                start_q=best_params['start_q'], 
                seasonal=True,
                max_p=3, 
                max_q=3, 
                stepwise=True
            ),
            'ThetaForecaster': lambda: ThetaForecaster(
                sp=sp_value
            ),
            'NaiveForecaster': lambda: NaiveForecaster(
                strategy=best_params['strategy']
            ),
            'PolynomialTrendForecaster': lambda: PolynomialTrendForecaster(
                degree=best_params['degree']
            ),
            'TBATS': lambda: TBATS(
                use_box_cox=best_params['use_box_cox'],
                use_trend=best_params['use_trend'],
                use_damped_trend=best_params['use_damped_trend'],
                sp=sp_value if sp_value > 1 else None
            ),
            'ExponentialSmoothing': lambda: ExponentialSmoothing(
                trend=best_params['trend'],
                seasonal=seasonal_param,
                sp=sp_value if sp_value > 1 else None
            )
        }

        self.best_model = model_constructors[model_name]()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.best_model.fit(self.y)

    def forecast(self, fh: np.ndarray = np.arange(1, 13)) -> pd.Series:
        """
        Generate forecasts from the trained best model.

        Parameters:
        -----------
        fh : np.ndarray
            Forecast horizon relative to the end of the training data.

        Returns:
        --------
        pd.Series
            The forecasted values.
        """
        if self.best_model is None:
            raise ValueError("No best model trained. Run optimize() first.")
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.best_model.predict(fh)