import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import holidays
from src.utils.datetime_utils import ensure_datetime_index

class TimeSeriesFeatureEngineering:
    """
    A class for generating engineered features from time series data with frequency detection.
    """
    
    def __init__(self, data, target_col=None, country="US"):
        """
        Initialize with time series data.
        
        Parameters:
        data (pd.DataFrame or pd.Series): DataFrame with a datetime index or Series.
        target_col (str, optional): Column name of the target variable. If None and data is a Series, 
                                   the Series name will be used.
        country (str): Country code for holiday feature extraction.
        """
        print("\n📊 TIME SERIES FEATURE ENGINEERING 📊")
        print("=" * 50)
        
        # Handle Series input by converting to DataFrame
        if isinstance(data, pd.Series):
            print("🔄 Input: pandas Series")
            self.original_name = data.name or "value"
            self.df = data.to_frame(name=self.original_name)
            self.target_col = self.original_name
            print(f"   → Converting to DataFrame with column name: '{self.original_name}'")
        else:
            print("🔄 Input: pandas DataFrame")
            # Make a deep copy to avoid modifying the original DataFrame
            self.df = data.copy()
            
            # If target_col wasn't provided, use the first numeric column as a default
            if target_col is None:
                numeric_cols = self.df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    self.target_col = numeric_cols[0]
                    print(f"   → No target column specified. Using first numeric column: '{self.target_col}'")
                else:
                    raise ValueError("❌ No target column specified and no numeric columns found in DataFrame")
            else:
                self.target_col = target_col
        
        # Check if target column exists
        if self.target_col not in self.df.columns:
            raise KeyError(f"❌ Target column '{self.target_col}' not found in DataFrame.\n   Available columns: {list(self.df.columns)}")
            
        self.country = country
        
        print("\n📋 DATA SUMMARY")
        print("-" * 50)
        print(f"• Rows: {len(self.df):,}")
        print(f"• Columns: {len(self.df.columns)}")
        print(f"• Target: '{self.target_col}'")
        print(f"• Country: {self.country}")
        
        # Force explicit conversion of index to datetime
        self._ensure_datetime_index()
        
        # Detect frequency of the time series
        self.freq = self._detect_frequency()
        print(f"• Frequency: {self.freq}")
        print("-" * 50)
    
    def _ensure_datetime_index(self):
        """Ensure DataFrame has a proper DatetimeIndex using the shared utility function."""
        print("\n🕒 PREPARING DATETIME INDEX")
        try:
            self.df = ensure_datetime_index(self.df)
            print("✓ DatetimeIndex ensured successfully")
        except ValueError as e:
            print(f"❌ Error ensuring DatetimeIndex: {e}")
            raise

    def _detect_frequency(self):
        """Detect the frequency of the time series."""
        print("\n🔍 DETECTING TIME SERIES FREQUENCY")
        
        try:
            # Sort index to ensure proper detection
            self.df = self.df.sort_index()
            
            # Use pandas infer_freq to detect frequency
            freq = pd.infer_freq(self.df.index)
            if freq:
                print(f"   → Pandas inferred frequency: {freq}")
            
            if freq is None and len(self.df) > 100:
                # Try with a subset if we have a large dataset
                freq = pd.infer_freq(self.df.index[:100])
                if freq:
                    print(f"   → Subset frequency detection: {freq}")
            
            if freq is None:
                print("   → Standard frequency detection failed, calculating from time differences...")
                # Calculate most common time difference
                time_diffs = self.df.index[1:] - self.df.index[:-1]
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

    def add_time_features(self):
        """Extracts time-based features depending on the frequency."""
        print("\n⏰ Adding time features")
        
        count = 0
        try:
            # Always add year for any frequency
            self.df["year"] = self.df.index.year
            count += 1
            
            # Add features based on frequency
            if self.freq in ['hourly', 'minutely', 'secondly', 'quarter_hourly', 'half_hourly']:
                self.df["hour"] = self.df.index.hour
                self.df["is_business_hour"] = ((self.df.index.hour >= 9) & 
                                              (self.df.index.hour < 17)).astype(int)
                count += 2
                
                if self.freq in ['minutely', 'secondly', 'quarter_hourly', 'half_hourly']:
                    self.df["minute"] = self.df.index.minute
                    count += 1
                    
                    if self.freq == 'secondly':
                        self.df["second"] = self.df.index.second
                        count += 1
                
            if self.freq in ['daily', 'hourly', 'minutely', 'secondly', 'business_daily', 'weekly', 
                            'quarter_hourly', 'half_hourly']:
                self.df["month"] = self.df.index.month
                self.df["day_of_week"] = self.df.index.weekday  # 0=Monday, 6=Sunday
                self.df["is_weekend"] = (self.df.index.weekday >= 5).astype(int)
                count += 3
                
            print(f"   ✓ Added {count} time features")
            
        except Exception as e:
            print(f"   ❌ Error adding time features: {str(e)[:80]}...")
            
        return self.df

    def add_cyclical_features(self):
        """Add cyclical time features using sine and cosine transformations."""
        print("Adding cyclical time features...")
        
        try:
            # Add features based on frequency
            if self.freq in ['hourly', 'minutely', 'secondly']:
                # Hour of day (0-23) -> cyclical feature
                hours_in_day = 24
                self.df['hour_sin'] = np.sin(2 * np.pi * self.df.index.hour / hours_in_day)
                self.df['hour_cos'] = np.cos(2 * np.pi * self.df.index.hour / hours_in_day)
                
                if self.freq in ['minutely', 'secondly']:
                    # Minute of hour (0-59) -> cyclical feature
                    minutes_in_hour = 60
                    self.df['minute_sin'] = np.sin(2 * np.pi * self.df.index.minute / minutes_in_hour)
                    self.df['minute_cos'] = np.cos(2 * np.pi * self.df.index.minute / minutes_in_hour)
            
            if self.freq in ['daily', 'hourly', 'minutely', 'secondly', 'business_daily', 'weekly']:
                # Day of week (0-6) -> cyclical feature
                days_in_week = 7
                self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df.index.weekday / days_in_week)
                self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df.index.weekday / days_in_week)
            
            # Month features make sense for most frequencies except maybe very high frequency (seconds/minutes)
            if self.freq not in ['secondly', 'minutely'] or len(self.df) > 1000:
                # Month of year (1-12) -> cyclical feature
                months_in_year = 12
                self.df['month_sin'] = np.sin(2 * np.pi * self.df.index.month / months_in_year)
                self.df['month_cos'] = np.cos(2 * np.pi * self.df.index.month / months_in_year)
            
            print(f"Cyclical features added successfully for frequency: {self.freq}")
        except Exception as e:
            print(f"Failed to add cyclical features: {e}")
            
        return self.df

    def add_calendar_features(self):
        """Add additional calendar-based features based on frequency."""
        print("Adding calendar features...")
        
        try:
            # Features for daily or sub-daily frequencies
            if self.freq in ['daily', 'hourly', 'minutely', 'secondly', 'business_daily', 'weekly']:
                # Day of year (1-365)
                self.df['day_of_year'] = self.df.index.dayofyear
                
                # Week of year (1-52)
                try:
                    # For newer pandas versions
                    self.df['week_of_year'] = self.df.index.isocalendar().week
                except:
                    # For older pandas versions
                    self.df['week_of_year'] = self.df.index.week
            
            # Features for almost all frequencies except maybe secondly/minutely
            if self.freq != 'secondly' and self.freq != 'minutely':
                # Quarter (1-4)
                self.df['quarter'] = self.df.index.quarter
                
                # Is month start/end - makes sense for daily or near-daily frequencies
                if self.freq in ['daily', 'business_daily', 'weekly', 'hourly']:
                    self.df['is_month_start'] = self.df.index.is_month_start.astype(int)
                    self.df['is_month_end'] = self.df.index.is_month_end.astype(int)
            
            print(f"Calendar features added successfully for frequency: {self.freq}")
        except Exception as e:
            print(f"Failed to add calendar features: {e}")
            
        return self.df

    def add_time_of_day_features(self):
        """Add time-of-day category features if appropriate for the frequency."""
        print("Adding time-of-day features...")
        
        # Only makes sense for hourly or higher frequency data
        if self.freq not in ['hourly', 'minutely', 'secondly']:
            print(f"Time-of-day features not applicable for {self.freq} frequency.")
            return self.df
        
        try:
            # Morning, afternoon, evening, night categories
            hour = self.df.index.hour
            conditions = [
                (hour >= 5) & (hour < 12),     # Morning: 5 AM - 11:59 AM
                (hour >= 12) & (hour < 17),    # Afternoon: 12 PM - 4:59 PM
                (hour >= 17) & (hour < 22),    # Evening: 5 PM - 9:59 PM
                (hour >= 22) | (hour < 5)      # Night: 10 PM - 4:59 AM
            ]
            categories = ['morning', 'afternoon', 'evening', 'night']
            self.df['time_of_day'] = np.select(conditions, categories)
            
            # One-hot encode time of day
            for category in categories:
                self.df[f'is_{category}'] = (self.df['time_of_day'] == category).astype(int)
                
            print("Time-of-day features added successfully.")
        except Exception as e:
            print(f"Failed to add time-of-day features: {e}")
            
        return self.df

    def add_seasonal_features(self):
        """Add features for seasons if appropriate for the frequency."""
        print("Adding seasonal features...")
        
        # Only makes sense for daily or lower frequency data
        if self.freq not in ['daily', 'business_daily', 'weekly', 'monthly', 'quarterly', 'yearly']:
            print(f"Seasonal features not applicable for {self.freq} frequency.")
            return self.df
        
        try:
            # Seasons (for northern hemisphere)
            month = self.df.index.month
            conditions = [
                (month >= 3) & (month <= 5),    # Spring: March to May
                (month >= 6) & (month <= 8),    # Summer: June to August
                (month >= 9) & (month <= 11),   # Fall: September to November
                (month == 12) | (month <= 2)    # Winter: December to February
            ]
            seasons = ['spring', 'summer', 'fall', 'winter']
            self.df['season'] = np.select(conditions, seasons)
            
            # One-hot encode seasons
            for season in seasons:
                self.df[f'is_{season}'] = (self.df['season'] == season).astype(int)
                
            print("Seasonal features added successfully.")
        except Exception as e:
            print(f"Failed to add seasonal features: {e}")
            
        return self.df

    def add_holiday_feature(self):
        """Adds a binary feature for holidays if appropriate for the frequency."""
        print("Adding holiday features...")
        
        # Only makes sense for daily or subdaily frequency
        if self.freq not in ['daily', 'business_daily', 'hourly', 'minutely', 'secondly']:
            print(f"Holiday features not applicable for {self.freq} frequency.")
            return self.df
        
        try:
            holiday_calendar = holidays.country_holidays(self.country)
            
            # Use Series.dt accessor for safe date extraction
            dt_series = pd.Series(self.df.index)
            dates = dt_series.dt.date.tolist()
            
            # Check each date against the holiday calendar
            self.df["is_holiday"] = [1 if d in holiday_calendar else 0 for d in dates]
            print(f"Holiday features added for {self.country}")
        except Exception as e:
            print(f"Warning: Could not add holiday features: {e}")
            self.df["is_holiday"] = 0
            
        return self.df

    def add_lag_features(self, lags=None):
        """Adds lagged versions of the target variable with frequency-appropriate defaults."""
        if lags is None:
            # Default lags based on frequency
            if self.freq == 'quarter_hourly':
                lags = [1, 4, 12, 96, 672]  # 15min, 1hr, 3hrs, 1day, 1week
            elif self.freq == 'half_hourly':
                lags = [1, 2, 48, 336]  # 30min, 1hr, 1day, 1week
            elif self.freq == 'hourly':
                lags = [1, 2, 3, 24, 48, 168]  # 1hr, 2hr, 3hr, 1day, 2days, 1week
            elif self.freq == 'daily':
                lags = [1, 2, 3, 7, 14, 30]  # 1day, 2days, 3days, 1week, 2weeks, 1month
            elif self.freq == 'weekly':
                lags = [1, 2, 4, 8, 12, 26]  # 1week, 2weeks, 1month, 2months, 3months, 6months
            elif self.freq == 'monthly':
                lags = [1, 2, 3, 6, 12]  # 1month, 2months, 3months, 6months, 1year
            elif self.freq == 'quarterly':
                lags = [1, 2, 4]  # 1quarter, 2quarters, 1year
            elif self.freq == 'yearly':
                lags = [1, 2, 3]  # 1year, 2years, 3years
            else:
                lags = [1, 2, 3]  # Default fallback
        
        print(f"Adding lag features with lags: {lags}")
        
        for lag in lags:
            self.df[f"lag_{lag}"] = self.df[self.target_col].shift(lag)
            
        return self.df

    def add_lag_difference_features(self, lags=None):
        """Add differences between current values and lagged values."""
        # Use the same lags as add_lag_features if not specified
        if lags is None:
            # Default lags based on frequency as in add_lag_features
            if self.freq == 'hourly':
                lags = [1, 24, 168]  # 1hr, 1day, 1week
            elif self.freq == 'daily':
                lags = [1, 7, 30]  # 1day, 1week, 1month
            elif self.freq == 'weekly':
                lags = [1, 4, 12]  # 1week, 1month, 3months
            elif self.freq == 'monthly':
                lags = [1, 3, 12]  # 1month, 3months, 1year
            elif self.freq == 'quarterly':
                lags = [1, 4]  # 1quarter, 1year
            elif self.freq == 'yearly':
                lags = [1]  # 1year
            else:
                lags = [1]  # Default fallback
                
        print(f"Adding lag difference features with lags: {lags}")
        
        for lag in lags:
            lag_col = f'lag_{lag}'
            
            # Make sure the lag column exists
            if lag_col not in self.df.columns:
                self.df[lag_col] = self.df[self.target_col].shift(lag)
            
            # Calculate difference
            self.df[f'diff_{lag}'] = self.df[self.target_col] - self.df[lag_col]
            
            # Calculate percentage change
            self.df[f'pct_change_{lag}'] = self.df[self.target_col].pct_change(periods=lag)
        
        return self.df

    def add_rolling_statistics(self, windows=None):
        """Adds rolling mean and standard deviation features with frequency-appropriate windows."""
        if windows is None:
            # Default windows based on frequency
            if self.freq == 'hourly':
                windows = [3, 6, 12, 24]  # 3hrs, 6hrs, 12hrs, 24hrs
            elif self.freq == 'daily':
                windows = [3, 7, 14, 30]  # 3days, 1week, 2weeks, 1month
            elif self.freq == 'weekly':
                windows = [2, 4, 8, 12]  # 2weeks, 1month, 2months, 3months
            elif self.freq == 'monthly':
                windows = [2, 3, 6, 12]  # 2months, 3months, 6months, 1year
            elif self.freq == 'quarterly':
                windows = [2, 4]  # 2quarters, 1year
            elif self.freq == 'yearly':
                windows = [2, 3]  # 2years, 3years
            else:
                windows = [3, 7]  # Default fallback
        
        print(f"Adding rolling statistics with windows: {windows}")
        
        for window in windows:
            self.df[f"rolling_mean_{window}"] = self.df[self.target_col].rolling(
                window=window, min_periods=1).mean()
            self.df[f"rolling_std_{window}"] = self.df[self.target_col].rolling(
                window=window, min_periods=1).std()
                
        return self.df

    def scale_features(self, feature_cols=None):
        """Standardizes selected numerical features."""
        if feature_cols is None:
            # Skip scaling if no columns specified
            return self.df
        
        try:
            scaler = StandardScaler()
            self.df[feature_cols] = scaler.fit_transform(self.df[feature_cols])
            print(f"Scaled {len(feature_cols)} features")
        except Exception as e:
            print(f"Error scaling features: {e}")
            
        return self.df

    def generate_all_features(self, lags=None, windows=None):
        """Runs all feature engineering steps appropriate for the detected frequency."""
        print("\n🚀 GENERATING FEATURES")
        print("=" * 50)
        print(f"• Initial shape: {self.df.shape[0]:,} rows × {self.df.shape[1]:,} columns")
        print(f"• Target column: '{self.target_col}'")
        print(f"• Time series frequency: '{self.freq}'")
        print("-" * 50)
        
        # Track number of features added
        initial_cols = len(self.df.columns)
        
        # Dictionary to track features added by category
        feature_counts = {}
        
        # Base time features for all frequencies
        cols_before = len(self.df.columns)
        self.add_time_features()
        feature_counts["Time"] = len(self.df.columns) - cols_before
        
        # Add cyclical features for all frequencies
        cols_before = len(self.df.columns)
        self.add_cyclical_features()
        feature_counts["Cyclical"] = len(self.df.columns) - cols_before
        
        # Add calendar features for appropriate frequencies
        if self.freq in ['daily', 'business_daily', 'hourly', 'quarter_hourly', 'half_hourly', 
                        'minutely', 'secondly', 'weekly', 'monthly']:
            cols_before = len(self.df.columns)
            self.add_calendar_features()
            feature_counts["Calendar"] = len(self.df.columns) - cols_before
        
        # Add time-of-day features for sub-daily frequencies
        if self.freq in ['hourly', 'quarter_hourly', 'half_hourly', 'minutely', 'secondly']:
            cols_before = len(self.df.columns)
            self.add_time_of_day_features()
            feature_counts["Time of Day"] = len(self.df.columns) - cols_before
        
        # Add seasonal features for daily or less granular frequencies
        if self.freq in ['daily', 'business_daily', 'weekly', 'monthly', 'quarterly', 'yearly']:
            cols_before = len(self.df.columns)
            self.add_seasonal_features()
            feature_counts["Seasonal"] = len(self.df.columns) - cols_before
        
        # Add holiday features for daily or sub-daily frequencies
        if self.freq in ['daily', 'business_daily', 'hourly', 'quarter_hourly', 'half_hourly', 
                        'minutely', 'secondly']:
            cols_before = len(self.df.columns)
            self.add_holiday_feature()
            feature_counts["Holiday"] = len(self.df.columns) - cols_before
        
        # Add lag features for all frequencies
        cols_before = len(self.df.columns)
        self.add_lag_features(lags)
        feature_counts["Lag"] = len(self.df.columns) - cols_before
        
        # Add lag difference features
        cols_before = len(self.df.columns)
        self.add_lag_difference_features(lags)
        feature_counts["Lag Difference"] = len(self.df.columns) - cols_before
        
        # Add rolling statistics
        cols_before = len(self.df.columns)
        self.add_rolling_statistics(windows)
        feature_counts["Rolling Statistics"] = len(self.df.columns) - cols_before
        
        # Summary of features added
        features_added = len(self.df.columns) - initial_cols
        print("\n✅ FEATURE GENERATION COMPLETE")
        print("-" * 50)
        print(f"• Added {features_added} new features")
        
        # Print feature category breakdown
        print("\n📋 FEATURES BY CATEGORY")
        print("-" * 30)
        for category, count in feature_counts.items():
            if count > 0:  # Only show categories with features added
                print(f"• {category}: {count}")
        
        print(f"\n• Final shape: {self.df.shape[0]:,} rows × {self.df.shape[1]:,} columns")
        print("=" * 50)
        
        return self.df