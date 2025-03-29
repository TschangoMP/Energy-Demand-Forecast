import pandas as pd
import warnings

def ensure_datetime_index(data):
    """
    Ensure the Series or DataFrame has a proper DatetimeIndex using multiple approaches.
    
    Parameters:
    ----------- 
    data : pd.Series or pd.DataFrame
        The time series data.
        
    Returns:
    --------
    pd.Series or pd.DataFrame
        The data with a proper DatetimeIndex.
    
    Raises:
    -------
    ValueError:
        If the index cannot be converted to a DatetimeIndex.
    """
    print("\n🕒 PREPARING DATETIME INDEX")
    
    # First check if already a DatetimeIndex
    if isinstance(data.index, pd.DatetimeIndex):
        print("✓ Index is already a DatetimeIndex")
        # Ensure frequency is set if not already
        if data.index.freq is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                freq = pd.infer_freq(data.index)
                if freq is not None:
                    data = data.asfreq(freq)
                    print(f"✓ Set frequency to {freq}")
        return data
    
    try:
        # Attempt direct conversion
        print("   Converting index to datetime...")
        data.index = pd.to_datetime(data.index)
        print("✓ Successfully converted to DatetimeIndex")
        return data
    except Exception as e:
        print(f"   → Direct conversion failed: {e}")
    
    try:
        # Attempt alternative conversion method
        print("   Trying alternative conversion method...")
        temp_df = data.reset_index()
        temp_df['index'] = pd.to_datetime(temp_df['index'])
        data = temp_df.set_index('index')
        print("✓ Successfully converted using alternative method")
        return data
    except Exception as e:
        print(f"❌ All conversion methods failed: {e}")
        raise ValueError(
            "Could not convert index to DatetimeIndex. "
            "Please provide data with a proper datetime index."
        )
