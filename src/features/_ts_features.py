import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import holidays

class TimeSeriesFeatureEngineering:
    """
    A class for generating engineered features from time series data.
    """

    def __init__(self, df, target_col, country="US"):
        """
        Initialize with time series data.

        Parameters:
        df (pd.DataFrame): DataFrame with a datetime index.
        target_col (str): Column name of the target variable (e.g., electricity demand).
        country (str): Country code for holiday feature extraction (default: 'US').
        """
        self.df = df.copy()
        self.target_col = target_col
        self.country = country

        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex.")

    def add_time_features(self):
        """Extracts time-based features."""
        self.df["hour"] = self.df.index.hour
        self.df["day_of_week"] = self.df.index.dayofweek  # Monday = 0, Sunday = 6
        self.df["month"] = self.df.index.month
        self.df["is_weekend"] = (self.df["day_of_week"] >= 5).astype(int)
        return self.df

    def add_holiday_feature(self):
        """Adds a binary feature for holidays based on the specified country."""
        holiday_calendar = holidays.country_holidays(self.country)
        self.df["is_holiday"] = self.df.index.map(lambda x: 1 if x in holiday_calendar else 0)
        return self.df

    def add_lag_features(self, lags=[1, 24, 168]):
        """
        Adds lagged versions of the target variable to capture past trends.

        Parameters:
        lags (list): List of time steps to lag (e.g., [1, 24, 168] for previous hour, previous day, previous week).
        """
        for lag in lags:
            self.df[f"lag_{lag}"] = self.df[self.target_col].shift(lag)
        return self.df

    def add_rolling_statistics(self, windows=[3, 7, 30]):
        """
        Adds rolling mean and standard deviation features to smooth out fluctuations.

        Parameters:
        windows (list): List of window sizes in days.
        """
        for window in windows:
            self.df[f"rolling_mean_{window}"] = self.df[self.target_col].rolling(window=window, min_periods=1).mean()
            self.df[f"rolling_std_{window}"] = self.df[self.target_col].rolling(window=window, min_periods=1).std()
        return self.df

    def scale_features(self, feature_cols):
        """
        Standardizes selected numerical features using StandardScaler.

        Parameters:
        feature_cols (list): List of column names to be scaled.
        """
        scaler = StandardScaler()
        self.df[feature_cols] = scaler.fit_transform(self.df[feature_cols])
        return self.df

    def generate_all_features(self, lags=[1, 24, 168], windows=[3, 7, 30]):
        """
        Runs all feature engineering steps.

        Parameters:
        lags (list, optional): Lag periods to include.
        windows (list, optional): Rolling statistics windows.

        Returns:
        pd.DataFrame: DataFrame with all engineered features.
        """
        self.add_time_features()
        self.add_holiday_feature()
        self.add_lag_features(lags)
        self.add_rolling_statistics(windows)

        return self.df
