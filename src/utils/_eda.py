import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from scipy.stats import zscore
from scipy.signal import periodogram

sns.set_style("darkgrid")

class TimeSeriesEDA:
    """
    TimeSeriesEDA class for performing exploratory data analysis on time series data.
    Methods
    -------
    __init__(self, ts, periodicity=None)
    _infer_periodicity(self)
        Infer periodicity using spectral density estimation.
    summary_statistics(self)
        Compute basic statistics for the time series.
    rolling_statistics(self, window=7)
        Compute rolling mean and standard deviation.
    test_stationarity(self)
        Perform the Augmented Dickey-Fuller (ADF) test for stationarity.
    detect_outliers(self, method="seasonal", threshold=3.0)
        Detect outliers in the time series using specified method.
    _detect_seasonal_outliers(self)
        Detect outliers using SeasonalESD.
    decompose(self)
        Perform STL decomposition to analyze trend and seasonality.
    plot_time_series(self, title="Time Series Data")
        Plot the time series data.
    plot_rolling_statistics(self)
        Plot rolling mean and standard deviation along with detected outliers.
    plot_decomposition(self)
        Plot the STL decomposition components.
    run_full_analysis(self)
        Run full EDA and display results.
    """
    
    def __init__(self, ts, periodicity=None):
        """
        Initialize TimeSeriesEDA with a time series and optional periodicity.
        
        Parameters:
        ts (pd.Series): Time series data.
        periodicity (int, optional): Periodicity for decomposition and rolling statistics.
        """
        if isinstance(ts, pd.DataFrame):
            if ts.shape[1] != 1:
                raise ValueError("DataFrame must have only one column for time series data.")
            ts = ts.squeeze()  # Convert single-column DataFrame to Series
        elif not isinstance(ts, pd.Series):
            raise ValueError("Input time series must be a pandas Series or a single-column DataFrame.")

        self.ts = ts
        self.periodicity = periodicity or self._infer_periodicity()
        if self.periodicity is None:
            raise ValueError("Could not infer periodicity. Please provide a valid periodicity.")

    def _infer_periodicity(self):
        """Infer periodicity using spectral density estimation."""
        freqs, spectrum = periodogram(self.ts.dropna(), detrend="linear")
        period = int(1 / freqs[np.argmax(spectrum)]) if np.any(spectrum > 0) else None
        return period if period and period > 1 else None

    def summary_statistics(self):
        """Compute basic statistics for the time series."""
        return {
            "mean": self.ts.mean(),
            "variance": self.ts.var(),
            "min": self.ts.min(),
            "max": self.ts.max(),
            "missing_values": self.ts.isna().sum(),
            "periodicity": self.periodicity,
        }

    def rolling_statistics(self, window=7):
        """Compute rolling mean and standard deviation."""
        rolling_mean = self.ts.rolling(window=window, min_periods=1).mean()
        rolling_std = self.ts.rolling(window=window, min_periods=1).std()
        return rolling_mean, rolling_std

    def test_stationarity(self):
        """Perform the Augmented Dickey-Fuller (ADF) test for stationarity."""
        try:
            result = adfuller(self.ts.dropna())
            return {
                "ADF Statistic": result[0],
                "p-value": result[1],
                "Lags Used": result[2],
                "Critical Values": result[4],
                "Stationary": result[1] < 0.05,
            }
        except ValueError as e:
            if "sample size is too short" in str(e):
                return {"error": "ADF test requires more data points."}
            else:
                raise

    def detect_outliers(self, method="seasonal", threshold=3.0):
        if self.ts.empty:
            return pd.Series(dtype=float), method
        if len(self.ts) < 2:
            raise ValueError("Time series data is insufficient for outlier detection.")
        ts_clean = self.ts.dropna()
        outliers = pd.Series(dtype=float)
        
        if method == "zscore":
            scores = zscore(ts_clean)
            outliers = ts_clean[np.abs(scores) > threshold]
        elif method == "iqr":
            q1, q3 = np.percentile(ts_clean, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = ts_clean[(ts_clean < lower) | (ts_clean > upper)]
        elif method == "mad":
            median = np.median(ts_clean)
            mad = np.median(np.abs(ts_clean - median))
            if mad != 0:
                outliers = ts_clean[np.abs(ts_clean - median) / mad > threshold]
        elif method == "seasonal":
            if self.periodicity is None or self.periodicity <= 1:
                raise ValueError("Periodicity is required for seasonal outlier detection.")
            stl = STL(ts_clean, period=self.periodicity, robust=True).fit()
            resid = stl.resid
            resid_z = zscore(resid)
            outliers = ts_clean[np.abs(resid_z) > threshold]
        else:
            raise ValueError("Invalid method. Choose 'zscore', 'iqr', 'mad', or 'seasonal'")
        
        return outliers, method

    def decompose(self):
        """Perform STL decomposition to analyze trend and seasonality."""
        if self.periodicity is None:
            raise ValueError("Periodicity is required for decomposition.")
        if len(self.ts.dropna()) < 2 * self.periodicity:
            raise ValueError("Not enough data points for STL decomposition. Need at least twice the periodicity.")
        
        stl = STL(self.ts.dropna(), period=self.periodicity, robust=True)
        return stl.fit()

    def plot_time_series(self, title="Time Series Data"):
        if self.ts.empty:
            print("Time series data is empty. Cannot plot.")
            return
        plt.figure(figsize=(14, 6))
        plt.plot(self.ts, color="blue", label="Observed Data", linewidth=2)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    def plot_rolling_statistics(self):
        """Plot rolling mean and standard deviation along with detected outliers."""
        window = self.periodicity or 7  # Default to 7 if periodicity is not detected
        try:
            rolling_mean, rolling_std = self.rolling_statistics(window)
        except ValueError as e:
            print(f"Error computing rolling statistics: {e}")
            return
        
        try:
            outliers, method = self.detect_outliers()
        except ValueError as e:
            print(f"Error detecting outliers: {e}")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Plot original data and outliers
        axes[0].plot(self.ts, color="#1f77b4", label="Original Data", linewidth=2)
        if not outliers.empty:
            axes[0].scatter(outliers.index, outliers, color="#d62728", label="Outliers", zorder=3)
        axes[0].set_title("Original Data & Outliers", fontsize=14, fontweight="bold")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        
        # Plot rolling mean
        axes[1].plot(rolling_mean, color="#ff7f0e", label="Rolling Mean", linewidth=2)
        axes[1].set_title("Rolling Mean", fontsize=14, fontweight="bold")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        
        # Plot rolling standard deviation
        axes[2].plot(rolling_std, color="#2ca02c", label="Rolling Std Dev", linewidth=2)
        axes[2].set_title("Rolling Std Dev", fontsize=14, fontweight="bold")
        axes[2].legend(loc="upper left")
        axes[2].grid(True, linestyle="--", alpha=0.6)
        
        plt.suptitle(f"Rolling Statistics with Outliers Detected ({method})", fontsize=16, fontweight="bold")
        plt.xlabel("Time", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def plot_decomposition(self):
        """Plot the STL decomposition components."""
        try:
            decomposition = self.decompose()
        except ValueError as e:
            print(f"Could not perform decomposition: {e}")
            return
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        components = ["observed", "trend", "seasonal", "resid"]
        colors = ["blue", "red", "green", "gray"]
        
        for ax, component, color in zip(axes, components, colors):
            ax.plot(getattr(decomposition, component), color=color, linewidth=2)
            ax.set_title(component.capitalize(), fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)
        
        plt.suptitle("STL Decomposition", fontsize=14, fontweight="bold")
        plt.xlabel("Time", fontsize=12)
        plt.tight_layout()
        plt.show()

    def run_full_analysis(self):
        """Run full EDA and display results."""
        print("### Summary Statistics ###")
        print(self.summary_statistics())

        print("\n### Stationarity Test (ADF) ###")
        print(self.test_stationarity())

        print("\n### Outliers Detected (Z-score Method) ###")
        print(self.detect_outliers(method="zscore"))

        print("\n### Outliers Detected (Seasonal Method) ###")
        try:
            print(self.detect_outliers(method="seasonal"))
        except ValueError as e:
            print(f"Could not perform seasonal outlier detection: {e}")

        print("\n### Plotting Data ###")
        self.plot_time_series()
        self.plot_rolling_statistics()
        self.plot_decomposition()