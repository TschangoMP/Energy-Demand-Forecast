import pandas as pd
import numpy as np

class DataProcessor:
    """
    A class for cleaning and preprocessing time series data for forecasting models.
    """
    
    @staticmethod
    def clean_data(df, threshold=1e9):
        """
        Clean the input dataframe by handling missing values and outliers.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input dataframe to clean
        threshold : float, optional
            The threshold for extreme values, defaults to 1e9
            
        Returns:
        --------
        pandas.DataFrame
            The cleaned dataframe
        """
        print("Checking for problematic values in the dataset:")
        print(f"NaN values count: {df.isna().sum().sum()}")
        
        # Safely check for infinity values only in numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        inf_count = 0
        if len(numeric_cols) > 0:
            inf_count = np.isinf(df[numeric_cols]).sum().sum()
        print(f"Infinity values count: {inf_count}")
        
        # Get summary statistics to identify potential issues
        print("\nData statistics:")
        print(df.describe())
        
        # Replace infinity values with NaN (only in numeric columns)
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Check for extremely large values
        large_value_count = 0
        if len(numeric_cols) > 0:
            large_value_mask = df[numeric_cols].abs() > threshold
            large_value_count = large_value_mask.sum().sum()
        print(f"\nValues larger than {threshold}: {large_value_count}")
        
        if large_value_count > 0:
            # Replace extremely large values with NaN (only in numeric columns)
            for col in numeric_cols:
                df.loc[df[col].abs() > threshold, col] = np.nan
            print("Replaced extremely large values with NaN")
        
        # Handle remaining NaN values - choose an appropriate strategy
        # Strategy 1: Forward fill followed by backward fill
        df_cleaned = df.fillna(method='ffill').fillna(method='bfill')
        
        # Check if cleaning was successful
        print(f"\nNaN values after cleaning: {df_cleaned.isna().sum().sum()}")
        
        # Safely check for infinity values in cleaned dataframe
        inf_count_after = 0
        numeric_cols_cleaned = df_cleaned.select_dtypes(include=np.number).columns
        if len(numeric_cols_cleaned) > 0:
            inf_count_after = np.isinf(df_cleaned[numeric_cols_cleaned]).sum().sum()
        print(f"Infinity values after cleaning: {inf_count_after}")
        
        # Display updated dataframe info
        print(f"\nCleaned data shape: {df_cleaned.shape}")
        
        return df_cleaned
    
    @staticmethod
    def prepare_time_index(df):
        """
        Ensure the DataFrame has a proper datetime index.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input dataframe to process
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with proper datetime index
        """
        # Ensure the date column is in datetime format and set it as index if needed
        if any(col.lower() == 'date' for col in df.columns):
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif any(col.lower() == 'timestamp' for col in df.columns):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
        return df
    
    @staticmethod
    def impute_missing_values(df, method='ffill_bfill', **kwargs):
        """
        Handle missing values with various imputation strategies.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with missing values
        method : str
            Imputation method: 'ffill_bfill', 'mean', 'median', 'interpolate', 'knn'
        kwargs : additional parameters for specific imputation methods
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with imputed values
        """
        df_result = df.copy()
        
        if method == 'ffill_bfill':
            return df_result.fillna(method='ffill').fillna(method='bfill')
        elif method == 'mean':
            return df_result.fillna(df_result.mean())
        elif method == 'median':
            return df_result.fillna(df_result.median()) 
        elif method == 'interpolate':
            return df_result.interpolate(method=kwargs.get('interp_method', 'linear'))
        elif method == 'knn':
            # Requires scikit-learn
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=kwargs.get('n_neighbors', 5))
            numeric_cols = df_result.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                df_result[numeric_cols] = imputer.fit_transform(df_result[numeric_cols])
            return df_result
        else:
            raise ValueError(f"Unsupported imputation method: {method}")
    
    @staticmethod
    def scale_features(df, method='standard', target_cols=None):
        """
        Scale numerical features using different methods.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        method : str
            Scaling method: 'standard', 'minmax', 'robust'
        target_cols : list, optional
            Columns to scale. If None, all numeric columns are scaled.
            
        Returns:
        --------
        pandas.DataFrame, object
            Scaled dataframe and the scaler object for inverse transformation
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        # Determine which columns to scale
        if target_cols is None:
            target_cols = df.select_dtypes(include=np.number).columns
        
        # Create a copy to avoid modifying original data
        df_scaled = df.copy()
        
        # Initialize the appropriate scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        
        # Apply scaling
        if len(target_cols) > 0:
            df_scaled[target_cols] = scaler.fit_transform(df_scaled[target_cols])
        
        return df_scaled, scaler
    
    @staticmethod
    def validate_dataframe(df):
        """
        Validate that the input is a proper DataFrame suitable for time series analysis.
        
        Parameters:
        -----------
        df : object
            Object to validate
            
        Returns:
        --------
        bool
            True if validation passes
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if not any(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
            raise ValueError("DataFrame must contain at least one numeric column")
        
        return True
