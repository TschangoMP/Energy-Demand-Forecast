import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from scipy.stats import zscore
from scipy.signal import periodogram
import os
from matplotlib.ticker import MaxNLocator

class TimeSeriesEDA:
    """
    TimeSeriesEDA class for performing exploratory data analysis on time series data.
    Methods
    -------
    __init__(self, ts, periodicity=None, theme="darkgrid", color_palette="muted")
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
    plot_time_series(self, title="Time Series Data", figsize=(14, 6), save_to=None)
        Plot the time series data.
    plot_rolling_statistics(self, window=None, figsize=(14, 12), save_to=None)
        Plot rolling mean and standard deviation along with detected outliers.
    plot_decomposition(self, figsize=(14, 10), save_to=None)
        Plot the STL decomposition components.
    run_full_analysis(self, save_plots=False, output_dir=None)
        Run full EDA and display results.
    dashboard(self, figsize=(16, 14), save_to=None)
        Create a comprehensive dashboard of visualizations.
    plot_seasonal(self, figsize=(14, 8), save_to=None)
        Create a seasonal plot to visualize patterns.
    plot_outlier_impact(self, method="seasonal", threshold=3.0, figsize=(14, 8), save_to=None)
        Visualize the impact of outliers on the time series.
    save_plot(self, fig, filename, dpi=300, bbox_inches="tight")
        Save a figure to a file.
    set_visual_theme(self, theme, color_palette)
        Set the visual theme for all plots.
    """
    
    def __init__(self, ts, periodicity=None, theme="darkgrid", color_palette="muted"):
        """
        Initialize TimeSeriesEDA with a time series and optional periodicity.
        
        Parameters:
        ts (pd.Series): Time series data.
        periodicity (int, optional): Periodicity for decomposition and rolling statistics.
        theme (str, optional): The seaborn visual theme to use ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks').
        color_palette (str, optional): The seaborn color palette to use.
        """
        if isinstance(ts, pd.DataFrame):
            if ts.shape[1] != 1:
                raise ValueError("DataFrame must have only one column for time series data.")
            ts = ts.squeeze()  # Convert single-column DataFrame to Series
        elif not isinstance(ts, pd.Series):
            raise ValueError("Input time series must be a pandas Series or a single-column DataFrame.")

        # Force explicit conversion of index to datetime
        self.ts = ts
        self._ensure_datetime_index()

        self.periodicity = periodicity or self._detect_frequency()
        if self.periodicity is None:
            raise ValueError("Could not infer periodicity. Please provide a valid periodicity.")
            
        self.set_visual_theme(theme, color_palette)
    
    def set_visual_theme(self, theme="darkgrid", color_palette="muted"):
        """Set the visual theme for all plots."""
        sns.set_style(theme)
        sns.set_palette(color_palette)
        return self
    
    def _ensure_datetime_index(self):
        """Ensure DataFrame has a proper DatetimeIndex using multiple approaches."""
        print("\n🕒 PREPARING DATETIME INDEX")
        
        # First check if already a DatetimeIndex
        if isinstance(self.ts.index, pd.DatetimeIndex):
            print("✓ Index is already a DatetimeIndex")
            return
        
        try:
            # Approach 1: Direct conversion (most efficient)
            print("   Converting index to datetime...")
            self.ts.index = pd.to_datetime(self.ts.index)
            print(f"✓ Successfully converted to DatetimeIndex")
            return
        except Exception as e:
            print(f"   → Direct conversion failed: {str(e)[:80]}...")
        
        try:
            # Approach 2: Reset index, convert column, set as new index
            print("   Trying alternative conversion method...")
            temp_df = self.ts.reset_index()
            temp_df['index'] = pd.to_datetime(temp_df['index'])
            self.df = temp_df.set_index('index')
            print(f"✓ Successfully converted using alternative method")
            return
        except Exception as e:
            print(f"❌ All conversion methods failed: {str(e)[:80]}...")
            raise ValueError("Could not convert index to DatetimeIndex. Please provide data with a proper datetime index.")

    def _detect_frequency(self):
        """Detect the frequency of the time series."""
        print("\n🔍 DETECTING TIME SERIES FREQUENCY")
        
        try:
            # Sort index to ensure proper detection
            self.ts = self.ts.sort_index()
            
            # Use pandas infer_freq to detect frequency
            freq = pd.infer_freq(self.ts.index)
            if freq:
                print(f"   → Pandas inferred frequency: {freq}")
            
            if freq is None and len(self.ts) > 100:
                # Try with a subset if we have a large dataset
                freq = pd.infer_freq(self.ts.index[:100])
                if freq:
                    print(f"   → Subset frequency detection: {freq}")
            
            if freq is None:
                print("   → Standard frequency detection failed, calculating from time differences...")
                # Calculate most common time difference
                time_diffs = self.ts.index[1:] - self.ts.index[:-1]
                if len(time_diffs) > 0:
                    most_common = time_diffs.mode()[0]
                    seconds = most_common.total_seconds()
                    
                    if seconds <= 1:
                        return 'secondly'
                    elif seconds <= 60:
                        return 'minutely'
                    elif seconds <= 3600:
                        return 'hourly'
                    elif seconds <= 86400:
                        return 'daily'
                    elif seconds <= 604800:
                        return 'weekly'
                    elif seconds <= 2678400:  # ~31 days
                        return 'monthly'
                    else:
                        return 'yearly'
            
            # Map pandas freq string to more readable format
            if freq:
                # Daily frequencies
                if freq == 'D':
                    return 'daily'
                elif freq.startswith('B'):
                    return 'business_daily'
                    
                # Hourly and sub-hourly
                elif freq in ['H', 'h']:
                    return 'hourly'
                elif freq in ['15T', 'T15', '15min']:
                    return 'quarter_hourly'
                elif freq in ['30T', 'T30', '30min']:
                    return 'half_hourly'
                elif freq.startswith('T') or freq.endswith('T') or 'min' in freq:
                    return 'minutely'
                    
                # Seconds
                elif freq.startswith('S'):
                    return 'secondly'
                    
                # Weekly, Monthly, Quarterly, Yearly
                elif freq.startswith('W'):
                    return 'weekly'
                elif freq.startswith('M'):
                    return 'monthly'
                elif freq.startswith('Q'):
                    return 'quarterly'
                elif freq.startswith('Y') or freq.startswith('A'):
                    return 'yearly'
                else:
                    print(f"   → Unknown frequency pattern: {freq}, defaulting to 'unknown'")
                    return 'unknown'
        
        except Exception as e:
            print(f"❌ Error detecting frequency: {str(e)[:80]}...")
        return 'unknown'

    def _get_numeric_periodicity(self):
        """Convert string frequency to numeric periodicity for statistical methods."""
        # Default mappings from string frequencies to numeric periods
        periodicity_map = {
            'hourly': 24,        # 24 hours in a day
            'quarter_hourly': 96, # 96 quarter hours in a day
            'half_hourly': 48,   # 48 half hours in a day
            'daily': 7,          # 7 days in a week
            'business_daily': 5, # 5 days in a business week
            'weekly': 52,        # 52 weeks in a year
            'monthly': 12,       # 12 months in a year
            'quarterly': 4,      # 4 quarters in a year
            'yearly': 10,        # Default for yearly (10-year cycle)
            'minutely': 60,      # 60 minutes in an hour
            'secondly': 60,      # 60 seconds in a minute
            'unknown': 7         # Default fallback to weekly
        }
        
        # Return the numeric periodicity or a default
        if isinstance(self.periodicity, (int, float)) and self.periodicity > 0:
            return self.periodicity
        elif self.periodicity in periodicity_map:
            return periodicity_map[self.periodicity]
        else:
            return 7  # Default fallback

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
            numeric_period = self._get_numeric_periodicity()
            if numeric_period <= 1:
                raise ValueError("Periodicity must be greater than 1 for seasonal outlier detection.")
            stl = STL(ts_clean, period=numeric_period, robust=True).fit()
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
        
        numeric_period = self._get_numeric_periodicity()
        
        if len(self.ts.dropna()) < 2 * numeric_period:
            raise ValueError(f"Not enough data points ({len(self.ts.dropna())}) for STL decomposition. Need at least {2 * numeric_period}.")
        
        stl = STL(self.ts.dropna(), period=numeric_period, robust=True)
        return stl.fit()

    def save_plot(self, fig, filename, dpi=300, bbox_inches="tight"):
        """Save a figure to a file."""
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Plot saved to {filename}")
        return self

    def plot_time_series(self, title="Time Series Data", figsize=(14, 6), save_to=None):
        """
        Plot the time series data with enhanced formatting.
        
        Parameters:
        title (str): Title of the plot
        figsize (tuple): Figure size (width, height)
        save_to (str, optional): Path to save the figure
        """
        if self.ts.empty:
            print("Time series data is empty. Cannot plot.")
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.ts, color="#1f77b4", label="Time Series", linewidth=2)
        
        if len(self.ts) < 50:
            ax.plot(self.ts.index, self.ts.values, 'o', color="#1f77b4", markersize=4)
        
        min_idx = self.ts.idxmin()
        max_idx = self.ts.idxmax()
        ax.scatter([min_idx, max_idx], [self.ts.min(), self.ts.max()], 
                  color=['#d62728', '#2ca02c'], s=80, zorder=5,
                  label=f"Min/Max Values")
        
        ax.annotate(f'Min: {self.ts.min():.2f}', xy=(min_idx, self.ts.min()), 
                   xytext=(10, -20), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        ax.annotate(f'Max: {self.ts.max():.2f}', xy=(max_idx, self.ts.max()), 
                   xytext=(10, 20), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        mean_value = self.ts.mean()
        ax.axhline(y=mean_value, color='#ff7f0e', linestyle='--', 
                  linewidth=1.5, label=f'Mean: {mean_value:.2f}')
        
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        
        ax.grid(True, linestyle="--", alpha=0.6)
        
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        stats_text = (f"N: {len(self.ts)}\n"
                     f"Mean: {self.ts.mean():.2f}\n"
                     f"Std: {self.ts.std():.2f}")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_to:
            self.save_plot(fig, save_to)
            
        plt.show()
        return fig

    def plot_rolling_statistics(self, window=None, figsize=(14, 12), save_to=None):
        """Plot rolling mean and standard deviation with enhanced formatting."""
        # Get numeric version of periodicity if window parameter isn't provided
        if window is None:
            window = self._get_numeric_periodicity()
            
        try:
            rolling_mean, rolling_std = self.rolling_statistics(window)
        except ValueError as e:
            print(f"Error computing rolling statistics: {e}")
            return None
        
        try:
            outliers, method = self.detect_outliers()
        except ValueError as e:
            print(f"Error detecting outliers: {e}")
            return None
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        
        axes[0].plot(self.ts, color=colors[0], label="Original Data", linewidth=2)
        if not outliers.empty:
            axes[0].scatter(outliers.index, outliers, color=colors[3], s=80, 
                           label=f"Outliers ({len(outliers)})", zorder=3, alpha=0.7)
        
        axes[0].fill_between(self.ts.index, 
                            self.ts.mean() - 2*self.ts.std(), 
                            self.ts.mean() + 2*self.ts.std(), 
                            color=colors[0], alpha=0.1, label="±2σ Range")
        
        axes[0].set_title("Original Data & Outliers", fontsize=14, fontweight="bold")
        axes[0].legend(loc="upper left", frameon=True, fancybox=True)
        axes[0].grid(True, linestyle="--", alpha=0.6)
        
        axes[1].plot(rolling_mean, color=colors[1], label=f"{window}-period Rolling Mean", linewidth=2)
        axes[1].fill_between(rolling_mean.index, 
                            rolling_mean - rolling_mean.std(), 
                            rolling_mean + rolling_mean.std(), 
                            color=colors[1], alpha=0.1)
        axes[1].set_title("Rolling Mean", fontsize=14, fontweight="bold")
        axes[1].legend(loc="upper left", frameon=True, fancybox=True)
        axes[1].grid(True, linestyle="--", alpha=0.6)
        
        axes[2].plot(rolling_std, color=colors[2], label=f"{window}-period Rolling Std Dev", linewidth=2)
        axes[2].set_title("Rolling Std Dev", fontsize=14, fontweight="bold")
        axes[2].legend(loc="upper left", frameon=True, fancybox=True)
        axes[2].grid(True, linestyle="--", alpha=0.6)
        
        for ax in axes:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        
        plt.suptitle(f"Rolling Statistics (Window Size: {window})", fontsize=16, fontweight="bold")
        plt.xlabel("Time", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_to:
            self.save_plot(fig, save_to)
            
        plt.show()
        return fig

    def plot_decomposition(self, figsize=(14, 10), save_to=None):
        """Plot the STL decomposition components with enhanced styling."""
        try:
            decomposition = self.decompose()
        except ValueError as e:
            print(f"Could not perform decomposition: {e}")
            return None
        
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        components = ["observed", "trend", "seasonal", "resid"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
        titles = ["Observed", "Trend", "Seasonality", "Residuals"]
        
        for i, (ax, component, color, title) in enumerate(zip(axes, components, colors, titles)):
            data = getattr(decomposition, component)
            
            ax.plot(data, color=color, linewidth=2)
            
            if component == "resid":
                std_resid = data.std()
                ax.fill_between(data.index, -2*std_resid, 2*std_resid, 
                               color=color, alpha=0.2, label="±2σ Range")
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            if component == "trend":
                ax.fill_between(data.index, 
                               data - data.rolling(window=20, min_periods=1).std() * 1.96,
                               data + data.rolling(window=20, min_periods=1).std() * 1.96,
                               color=color, alpha=0.1, label="95% CI")
            
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)
            
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            
            if component != "observed":
                stats_text = f"Min: {data.min():.2f}, Max: {data.max():.2f}, Std: {data.std():.2f}"
                props = dict(boxstyle='round', facecolor='white', alpha=0.7)
                ax.text(0.02, 0.92, stats_text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=props)
        
        plt.suptitle(f"STL Decomposition (Period: {self.periodicity})", fontsize=16, fontweight="bold")
        plt.xlabel("Time", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_to:
            self.save_plot(fig, save_to)
            
        plt.show()
        return fig
        
    def dashboard(self, figsize=(16, 14), save_to=None):
        """Create a comprehensive dashboard of time series visualizations."""
        if self.ts.empty:
            print("Time series data is empty. Cannot create dashboard.")
            return None
            
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(4, 4)
        
        ax_title = fig.add_subplot(gs[0, 0:4])
        ax_title.axis('off')
        ax_title.text(0.5, 0.5, f"Time Series Analysis Dashboard", 
                     ha='center', va='center', fontsize=20, fontweight='bold')
        
        ax1 = fig.add_subplot(gs[1, 0:4])
        ax1.plot(self.ts, color="#1f77b4", linewidth=2)
        ax1.set_title("Original Time Series", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.6)
        
        ax2 = fig.add_subplot(gs[2, 0:2])
        sns.histplot(self.ts, kde=True, ax=ax2, color="#2ca02c")
        ax2.set_title("Distribution", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.6)
        
        ax3 = fig.add_subplot(gs[2, 2:4])
        sns.boxplot(y=self.ts, ax=ax3, color="#ff7f0e")
        ax3.set_title("Box Plot", fontsize=12, fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.6)
        
        ax4 = fig.add_subplot(gs[3, 0:2])
        pd.plotting.autocorrelation_plot(self.ts.dropna(), ax=ax4)
        ax4.set_title("Autocorrelation", fontsize=12, fontweight="bold")
        ax4.grid(True, linestyle="--", alpha=0.6)
        
        ax5 = fig.add_subplot(gs[3, 2:4])
        try:
            outliers, _ = self.detect_outliers()
            ax5.plot(self.ts, color="#1f77b4", label="Time Series", alpha=0.7)
            if not outliers.empty:
                ax5.scatter(outliers.index, outliers, color='red', s=50, 
                           label=f'Outliers ({len(outliers)} points)')
            ax5.set_title("Outlier Detection", fontsize=12, fontweight="bold")
            ax5.legend(loc="best")
            ax5.grid(True, linestyle="--", alpha=0.6)
        except:
            ax5.text(0.5, 0.5, "Outlier detection not available", 
                    ha='center', va='center', fontsize=12)
            ax5.set_title("Outlier Detection", fontsize=12, fontweight="bold")
        
        stats = self.summary_statistics()
        stat_text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax_title.text(0.02, 0.5, "Summary Statistics:\n" + stat_text, 
                     transform=ax_title.transAxes, fontsize=10,
                     verticalalignment='center', bbox=props)
        
        try:
            stat_results = self.test_stationarity()
            if 'Stationary' in stat_results:
                is_stationary = "✓ Stationary" if stat_results['Stationary'] else "✗ Non-stationary"
                p_value = f"p-value: {stat_results['p-value']:.4f}"
                stat_text = f"{is_stationary}\n{p_value}"
            else:
                stat_text = "Stationarity test not available"
        except:
            stat_text = "Stationarity test not available"
        
        ax_title.text(0.98, 0.5, "Stationarity:\n" + stat_text, 
                     transform=ax_title.transAxes, fontsize=10,
                     horizontalalignment='right', verticalalignment='center', 
                     bbox=props)
        
        if save_to:
            self.save_plot(fig, save_to)
            
        plt.show()
        return fig
        
    def plot_seasonal(self, figsize=(14, 8), save_to=None):
        """Create a seasonal plot to visualize patterns."""
        if self.ts.empty or self.periodicity is None:
            print("Cannot create seasonal plot. Empty data or no periodicity detected.")
            return None
            
        freq = None
        if self._get_numeric_periodicity() == 7:
            freq = 'W'
            freq_name = "Weekly"
        elif self._get_numeric_periodicity() in [28, 29, 30, 31]:
            freq = 'M'
            freq_name = "Monthly"
        elif self._get_numeric_periodicity() in [365, 366]:
            freq = 'Y'
            freq_name = "Yearly"
        elif self._get_numeric_periodicity() == 24:
            freq = 'D'
            freq_name = "Daily"
        elif self._get_numeric_periodicity() == 4:
            freq = 'Q'
            freq_name = "Quarterly"
        else:
            freq = 'D'
            freq_name = f"Period-{self.periodicity}"
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if pd.infer_freq(self.ts.index) is not None:
            ax = self.ts.plot(ax=ax, marker='o', linestyle='-', alpha=0.5)
            
            try:
                if freq in ['W', 'M', 'Q', 'Y']:
                    seasonal_means = self.ts.groupby([getattr(self.ts.index, freq_method) 
                                                    for freq_method in {'W': 'weekday', 'M': 'day', 
                                                                       'Q': 'quarter', 'Y': 'month'}[freq]]).mean()
                    seasonal_means.plot(ax=ax, linewidth=3, color='red', 
                                      label=f"Mean by {freq_name}")
            except:
                pass
        else:
            ax.plot(self.ts, marker='o', linestyle='-', alpha=0.5)
        
        ax.set_title(f"Seasonal Pattern ({freq_name}, Period={self.periodicity})", 
                    fontsize=14, fontweight="bold")
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        
        plt.tight_layout()
        
        if save_to:
            self.save_plot(fig, save_to)
            
        plt.show()
        return fig
        
    def plot_outlier_impact(self, method="seasonal", threshold=3.0, figsize=(14, 8), save_to=None):
        """Plot the time series with and without outliers."""
        try:
            outliers, _ = self.detect_outliers(method=method, threshold=threshold)
        except ValueError as e:
            print(f"Error detecting outliers: {e}")
            return None
        
        if outliers.empty:
            print(f"No outliers detected using {method} method.")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.ts, label="Original Data", alpha=0.7, linewidth=2)
        
        ax.scatter(outliers.index, outliers, color='red', s=80, 
                  label=f'Outliers ({len(outliers)} points)', zorder=5)
        
        ts_no_outliers = self.ts.copy()
        ts_no_outliers[outliers.index] = np.nan
        ts_no_outliers = ts_no_outliers.interpolate()
        ax.plot(ts_no_outliers, color='green', linewidth=2, 
               label='Data with Outliers Removed')
        
        mae = np.mean(np.abs(self.ts - ts_no_outliers))
        max_diff = np.max(np.abs(self.ts - ts_no_outliers))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        impact_text = (f"Impact Metrics:\n"
                      f"Total Outliers: {len(outliers)}\n"
                      f"Mean Abs. Change: {mae:.4f}\n"
                      f"Max Change: {max_diff:.4f}")
        
        ax.text(0.02, 0.97, impact_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        ax.set_title(f"Impact of Outliers ({method.capitalize()} Method)", 
                    fontsize=14, fontweight="bold")
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend(loc='best', frameon=True, fancybox=True)
        ax.grid(True, linestyle="--", alpha=0.6)
        
        if save_to:
            self.save_plot(fig, save_to)
            
        plt.show()
        return fig

    def run_full_analysis(self, save_plots=False, output_dir=None):
        """Run full EDA and display results with better formatting."""
        print("╔═════════════════════════════════════════════════╗")
        print("║              TIME SERIES ANALYSIS               ║")
        print("╚═════════════════════════════════════════════════╝")
        
        print("\n📊 SUMMARY STATISTICS")
        print("───────────────────────────────")
        stats = self.summary_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\n📈 STATIONARITY TEST")
        print("───────────────────────────────")
        stat_results = self.test_stationarity()
        for key, value in stat_results.items():
            print(f"  {key}: {value}")

        print("\n⚠️ OUTLIERS DETECTED")
        print("───────────────────────────────")
        z_outliers, _ = self.detect_outliers(method="zscore")
        print(f"  Z-score method: {len(z_outliers)} outliers")
        
        try:
            s_outliers, _ = self.detect_outliers(method="seasonal")
            print(f"  Seasonal method: {len(s_outliers)} outliers")
        except ValueError as e:
            print(f"  Could not perform seasonal outlier detection: {e}")

        print("\n📉 VISUALIZATIONS")
        print("───────────────────────────────")
        
        if save_plots and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        plots = {}
        
        plots["time_series"] = self.plot_time_series(
            save_to=f"{output_dir}/time_series.png" if save_plots and output_dir else None
        )
        
        plots["rolling_stats"] = self.plot_rolling_statistics(
            save_to=f"{output_dir}/rolling_stats.png" if save_plots and output_dir else None
        )
        
        plots["decomposition"] = self.plot_decomposition(
            save_to=f"{output_dir}/decomposition.png" if save_plots and output_dir else None
        )
        
        plots["dashboard"] = self.dashboard(
            save_to=f"{output_dir}/dashboard.png" if save_plots and output_dir else None
        )
        
        plots["seasonal"] = self.plot_seasonal(
            save_to=f"{output_dir}/seasonal.png" if save_plots and output_dir else None
        )
        
        plots["outlier_impact"] = self.plot_outlier_impact(
            save_to=f"{output_dir}/outlier_impact.png" if save_plots and output_dir else None
        )
        
        return {
            "statistics": stats,
            "stationarity": stat_results,
            "outliers": {"zscore": z_outliers, "seasonal": s_outliers if 's_outliers' in locals() else None},
            "plots": plots
        }