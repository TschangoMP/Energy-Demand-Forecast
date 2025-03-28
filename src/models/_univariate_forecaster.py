import numpy as np
import pandas as pd
import optuna
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

class UnivariateForecaster:
    """
    An automated univariate forecaster that uses an expanding window splitter and Optuna to find the best model.
    """

    def __init__(self, y: pd.Series, initial_window: int = 60, step_length: int = 12, fh: np.ndarray = np.arange(1, 13)):
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
        self.y = self._ensure_datetime_index(y)
        self.cv = ExpandingWindowSplitter(fh=fh, initial_window=initial_window, step_length=step_length)
        self.study = None
        self.best_model = None

    def _ensure_datetime_index(self, y: pd.Series) -> pd.Series:
        """
        Ensure the Series has a proper DatetimeIndex using multiple approaches.
        
        Parameters:
        -----------
        y : pd.Series
            The time series data.
            
        Returns:
        --------
        pd.Series
            The series with a proper DatetimeIndex.
        """
        print("\n🕒 PREPARING DATETIME INDEX")
        
        # First check if already a DatetimeIndex
        if isinstance(y.index, pd.DatetimeIndex):
            print("✓ Index is already a DatetimeIndex")
            return y
        
        try:
            # Approach 1: Direct conversion (most efficient)
            print("   Converting index to datetime...")
            y_converted = y.copy()
            y_converted.index = pd.to_datetime(y.index)
            print(f"✓ Successfully converted to DatetimeIndex")
            return y_converted
        except Exception as e:
            print(f"   → Direct conversion failed: {str(e)[:80]}...")
        
        try:
            # Approach 2: Reset index, convert column, set as new index
            print("   Trying alternative conversion method...")
            temp_df = y.reset_index()
            temp_df['index'] = pd.to_datetime(temp_df['index'])
            y_converted = temp_df.set_index('index').iloc[:, 0]
            print(f"✓ Successfully converted using alternative method")
            return y_converted
        except Exception as e:
            print(f"❌ All conversion methods failed: {str(e)[:80]}...")
            raise ValueError("Could not convert index to DatetimeIndex. Please provide data with a proper datetime index.")

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
        model_name = trial.suggest_categorical(
            'model', ['AutoARIMA', 'ThetaForecaster', 'NaiveForecaster',
                      'PolynomialTrendForecaster', 'TBATS', 'ExponentialSmoothing']
        )
        model_constructors = {
            'AutoARIMA': lambda: AutoARIMA(
                start_p=trial.suggest_int('start_p', 0, 4),
                d=trial.suggest_int('d', 0, 2),
                start_q=trial.suggest_int('start_q', 0, 4),
                seasonal=True, max_p=6, max_q=6, stepwise=True
            ),
            'ThetaForecaster': lambda: ThetaForecaster(sp=self.y.index.freq.n),
            'NaiveForecaster': lambda: NaiveForecaster(
                strategy=trial.suggest_categorical('strategy', ['last', 'mean', 'drift'])
            ),
            'PolynomialTrendForecaster': lambda: PolynomialTrendForecaster(
                degree=trial.suggest_int('degree', 1, 4)
            ),
            'TBATS': lambda: TBATS(
                use_box_cox=trial.suggest_categorical('use_box_cox', [True, False]),
                use_trend=trial.suggest_categorical('use_trend', [True, False]),
                use_damped_trend=trial.suggest_categorical('use_damped_trend', [True, False]),
                sp=self.y.index.freq.n
            ),
            'ExponentialSmoothing': lambda: ExponentialSmoothing(
                trend=trial.suggest_categorical('trend', ['add', 'mul', None]),
                seasonal=trial.suggest_categorical('seasonal', ['add', 'mul', None]),
                sp=self.y.index.freq.n
            )
        }
        return model_constructors[model_name]()

    def evaluate_model(self, model) -> float:
        """
        Evaluate the model using ExpandingWindowSplitter and mean absolute percentage error (MAPE).

        Parameters:
        -----------
        model : sktime.forecasting.base.BaseForecaster
            The forecasting model to be evaluated.

        Returns:
        --------
        float
            The mean absolute percentage error of the model.
        """
        errors = []
        fh_range = np.arange(1, self.cv.fh[-1] + 1)
        for train_idx, test_idx in self.cv.split(self.y):
            model.fit(self.y.iloc[train_idx])
            y_pred = model.predict(fh_range[:len(test_idx)])
            error = MeanSquaredError(square_root=True)
            error_value = error(self.y.iloc[test_idx], y_pred)
            errors.append(error_value)
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
        model = self.create_model(trial)
        return self.evaluate_model(model)

    def optimize(self, n_trials: int = 50, n_jobs: int = -1) -> pd.Series:
        """
        Optimize the model parameters using Optuna and return forecast from the best model.

        Parameters:
        -----------
        n_trials : int
            The number of trials for optimization.
        n_jobs : int
            The number of jobs to run in parallel.

        Returns:
        --------
        pd.Series
            The forecasted values from the best model.
        """
        self.study = optuna.create_study(direction='minimize', sampler=TPESampler(), pruner=MedianPruner())
        self.study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)
        
        self.train_best_model()
        return self.forecast()

    def train_best_model(self):
        """
        Retrain the best model found from optimization on the full dataset.
        """
        best_params = self.study.best_trial.params
        model_name = best_params['model']

        model_constructors = {
            'AutoARIMA': lambda: AutoARIMA(
                start_p=best_params['start_p'], d=best_params['d'], start_q=best_params['start_q'], seasonal=True
            ),
            'ThetaForecaster': lambda: ThetaForecaster(sp=self.y.index.freq.n),
            'NaiveForecaster': lambda: NaiveForecaster(strategy=best_params['strategy']),
            'PolynomialTrendForecaster': lambda: PolynomialTrendForecaster(degree=best_params['degree']),
            'TBATS': lambda: TBATS(
                use_box_cox=best_params['use_box_cox'],
                use_trend=best_params['use_trend'],
                use_damped_trend=best_params['use_damped_trend'],
                sp=self.y.index.freq.n
            ),
            'ExponentialSmoothing': lambda: ExponentialSmoothing(
                trend=best_params['trend'],
                seasonal=best_params['seasonal'],
                sp=self.y.index.freq.n
            )
        }

        self.best_model = model_constructors[model_name]()
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
        return self.best_model.predict(fh)