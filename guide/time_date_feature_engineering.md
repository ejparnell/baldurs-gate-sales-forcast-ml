# ⏰ Time and Date Feature Engineering Guide

*A comprehensive guide to standardizing, processing, and engineering features from temporal data for machine learning*

## 📋 Table of Contents

### Core Framework
- [📖 Overview](#-overview)
- [🔍 Understanding Temporal Data](#-understanding-temporal-data)
- [⚠️ Common Temporal Data Issues](#️-common-temporal-data-issues)

### Standardization & Preprocessing
- [🔧 Date/Time Parsing and Standardization](#-datetime-parsing-and-standardization)
- [🌍 Timezone Handling](#-timezone-handling)
- [📅 Date Range Validation](#-date-range-validation)

### Feature Engineering
- [⚡ Basic Temporal Features](#-basic-temporal-features)
- [📊 Cyclical Feature Engineering](#-cyclical-feature-engineering)
- [📈 Time-Based Aggregations](#-time-based-aggregations)
- [🔄 Lag and Window Features](#-lag-and-window-features)

### Advanced Techniques
- [📋 Business Calendar Features](#-business-calendar-features)
- [🎯 Holiday and Event Detection](#-holiday-and-event-detection)
- [⏳ Time Since/Until Features](#-time-sinceuntil-features)
- [📊 Seasonal Decomposition](#-seasonal-decomposition)

### Implementation & Validation
- [💻 Complete Pipeline Implementation](#-complete-pipeline-implementation)
- [✅ Temporal Feature Validation](#-temporal-feature-validation)
- [⭐ Best Practices](#-best-practices)
- [⚠️ Common Pitfalls](#️-common-pitfalls)
- [📝 Summary](#-summary)

---

## 📖 Overview

Temporal data is one of the richest sources of information in machine learning, yet it's often underutilized due to preprocessing complexity. This guide provides a comprehensive framework for extracting maximum value from date and time information through systematic standardization and feature engineering.

### Why Temporal Feature Engineering Matters

**Business Impact:**
- **Seasonal Patterns**: Capture recurring business cycles and seasonal trends
- **Operational Insights**: Understand time-based operational patterns
- **Forecasting Power**: Enable sophisticated time series modeling
- **Customer Behavior**: Model temporal aspects of user interactions

**Technical Benefits:**
- **Rich Feature Space**: Extract dozens of meaningful features from single datetime column
- **Model Performance**: Significantly improve predictive accuracy through temporal insights
- **Pattern Recognition**: Help algorithms detect complex temporal relationships
- **Data Quality**: Standardize and validate temporal data consistency

### What This Guide Covers

1. **Data Standardization**: Parsing, timezone handling, and validation of temporal data
2. **Basic Features**: Standard temporal components (year, month, day, etc.)
3. **Cyclical Encoding**: Handle periodic patterns with mathematical transformations
4. **Aggregation Features**: Time-based statistical summaries and rolling windows
5. **Advanced Features**: Business calendars, holidays, lag features, and seasonal decomposition
6. **Production Pipeline**: Complete, reusable implementation for real-world deployment

### Prerequisites

- Basic understanding of pandas datetime functionality
- Familiarity with feature engineering concepts
- Knowledge of time series analysis principles (helpful but not required)

---

## 🔍 Understanding Temporal Data

### Types of Temporal Data

**1. Timestamps**
```python
# Point-in-time events
"2024-03-15 14:30:00"
"2024-03-15T14:30:00Z"  # ISO format with timezone
```

**2. Date-only Data**
```python
# Calendar dates without time component
"2024-03-15"
"March 15, 2024"
```

**3. Time-only Data**
```python
# Time components without date
"14:30:00"
"2:30 PM"
```

**4. Duration/Intervals**
```python
# Time spans
"2 hours 30 minutes"
pd.Timedelta('2 days 3 hours')
```

**5. Relative Time References**
```python
# Business-relative time
"Q1 2024"
"Week 12"
"2 days ago"
```

### Temporal Data Characteristics

**Granularity Levels:**
- **Second-level**: High-frequency trading, sensor data, web analytics
- **Minute-level**: Call center data, manufacturing processes
- **Hour-level**: Energy consumption, web traffic patterns
- **Day-level**: Sales data, user activity, financial markets
- **Week/Month-level**: Business reporting, seasonal analysis
- **Year-level**: Long-term trends, demographic data

**Temporal Patterns:**
- **Linear Trends**: Consistent growth or decline over time
- **Cyclical Patterns**: Repeating patterns (daily, weekly, monthly, yearly)
- **Seasonal Effects**: Weather-dependent or calendar-driven variations
- **Event-driven Spikes**: Holidays, promotions, external events
- **Regime Changes**: Structural breaks in temporal patterns

**Time Zone Considerations:**
- **UTC Standardization**: Global data requires consistent timezone handling
- **Local Time Relevance**: Business operations often depend on local time
- **Daylight Saving**: Automatic adjustments can create data irregularities
- **Cross-timezone Analysis**: Multi-region datasets require careful synchronization

---

## ⚠️ Common Temporal Data Issues

### 1. **Inconsistent Date Formats**

**Problem**: Mixed date formats within the same column
```python
# Example of problematic data
mixed_dates = [
    '2024-03-15 14:30:00',    # ISO format
    '03/15/2024 2:30 PM',     # US format
    '15/03/2024 14:30',       # EU format
    'March 15, 2024',         # Text format
    '1710505800'              # Unix timestamp
]
```

**Solution**: Standardize all dates to consistent format
```python
# Universal parsing approach
def standardize_dates(date_series):
    # Try pandas automatic parsing first
    parsed_dates = pd.to_datetime(date_series, errors='coerce')
    
    # Handle failed parses with custom logic
    failed_mask = parsed_dates.isna() & date_series.notna()
    if failed_mask.any():
        # Custom parsing for specific formats
        for idx in failed_mask[failed_mask].index:
            try:
                # Handle unix timestamps
                if date_series[idx].isdigit():
                    parsed_dates[idx] = pd.to_datetime(int(date_series[idx]), unit='s')
            except:
                continue
    
    return parsed_dates
```

### 2. **Timezone Inconsistencies**

**Problem**: Mixed or unclear timezones
```python
# Timezone-naive vs timezone-aware data
naive_time = pd.Timestamp('2024-03-15 14:30:00')
utc_time = pd.Timestamp('2024-03-15 14:30:00', tz='UTC')
est_time = pd.Timestamp('2024-03-15 14:30:00', tz='America/New_York')
```

**Solution**: Standardize to UTC for storage, convert to local for analysis
```python
def standardize_timezone(datetime_series, source_tz='UTC'):
    # If timezone-naive, assume source timezone
    if datetime_series.dt.tz is None:
        datetime_series = datetime_series.dt.tz_localize(source_tz)
    
    # Convert all to UTC
    return datetime_series.dt.tz_convert('UTC')
```

### 3. **Missing Temporal Data**

**Problem**: Gaps in time series data
```python
# Detecting temporal gaps
def detect_gaps(datetime_series, expected_freq='D'):
    # Create expected date range
    full_range = pd.date_range(
        start=datetime_series.min(),
        end=datetime_series.max(),
        freq=expected_freq
    )
    
    # Find missing dates
    missing_dates = full_range.difference(datetime_series)
    return missing_dates
```

**Solution**: Handle based on business context
```python
def handle_temporal_gaps(df, date_col, strategy='interpolate'):
    if strategy == 'interpolate':
        # Linear interpolation for missing timestamps
        return df.set_index(date_col).interpolate(method='time')
    elif strategy == 'forward_fill':
        # Use last known value
        return df.set_index(date_col).fillna(method='ffill')
    elif strategy == 'drop':
        # Remove rows with missing dates
        return df.dropna(subset=[date_col])
```

---

## 🔧 Date/Time Parsing and Standardization

### Universal DateTime Parser

```python
class DateTimeStandardizer:
    def __init__(self):
        self.common_formats = [
            '%Y-%m-%d %H:%M:%S',      # ISO-like
            '%Y-%m-%d',               # Date only
            '%d/%m/%Y %H:%M:%S',      # EU format
            '%m/%d/%Y %H:%M:%S',      # US format
            '%Y-%m-%dT%H:%M:%S',      # ISO
            '%Y-%m-%dT%H:%M:%SZ',     # ISO with Z
        ]
    
    def parse_datetime_column(self, series, preferred_format=None):
        """Parse datetime column with automatic format detection"""
        
        # Try preferred format first
        if preferred_format:
            try:
                parsed = pd.to_datetime(series, format=preferred_format, errors='coerce')
                if parsed.notna().sum() / len(series) > 0.8:  # 80% success rate
                    return parsed
            except:
                pass
        
        # Try pandas automatic parsing
        parsed = pd.to_datetime(series, errors='coerce')
        
        # If many failures, try common formats
        if parsed.isna().sum() > len(series) * 0.2:  # More than 20% failures
            for fmt in self.common_formats:
                try:
                    test_parsed = pd.to_datetime(series, format=fmt, errors='coerce')
                    if test_parsed.notna().sum() > parsed.notna().sum():
                        parsed = test_parsed
                except:
                    continue
        
        return parsed

# Example usage
standardizer = DateTimeStandardizer()

# Sample data with mixed formats
dates = ['2024-03-15', '03/15/2024', '15-03-2024', '2024-03-15T14:30:00']
df = pd.DataFrame({'date_column': dates})

# Parse and standardize
df['standardized_date'] = standardizer.parse_datetime_column(df['date_column'])
```

---

## 🌍 Timezone Handling

### Timezone Standardization

```python
def standardize_timezones(df, datetime_cols, target_tz='UTC'):
    """Standardize all datetime columns to target timezone"""
    
    for col in datetime_cols:
        dt_series = pd.to_datetime(df[col])
        
        # If timezone-naive, assume UTC
        if dt_series.dt.tz is None:
            dt_series = dt_series.dt.tz_localize('UTC')
        
        # Convert to target timezone
        df[col] = dt_series.dt.tz_convert(target_tz)
    
    return df

# Business timezone conversion for analysis
def convert_to_business_timezone(df, datetime_col, business_tz='America/New_York'):
    """Convert UTC times to business timezone for analysis"""
    
    # Ensure timezone-aware
    dt_series = pd.to_datetime(df[datetime_col])
    if dt_series.dt.tz is None:
        dt_series = dt_series.dt.tz_localize('UTC')
    
    # Convert to business timezone
    return dt_series.dt.tz_convert(business_tz)

# Example
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
    'sales': np.random.uniform(100, 1000, 100)
})

# Standardize to UTC
df = standardize_timezones(df, ['timestamp'])

# Convert to business timezone for analysis
df['business_time'] = convert_to_business_timezone(df, 'timestamp', 'America/New_York')
```

---

## 📅 Date Range Validation

### Business Rule Validation

```python
def validate_date_ranges(df, date_cols, min_date=None, max_date=None):
    """Validate dates against business rules"""
    
    validation_results = {}
    
    # Set default ranges if not provided
    if min_date is None:
        min_date = pd.Timestamp('1990-01-01')  # Business start
    if max_date is None:
        max_date = pd.Timestamp.now() + pd.DateOffset(years=2)  # Reasonable future
    
    for col in date_cols:
        dt_series = pd.to_datetime(df[col])
        
        # Count violations
        too_early = (dt_series < min_date).sum()
        too_late = (dt_series > max_date).sum()
        
        validation_results[col] = {
            'total_records': len(dt_series),
            'valid_records': len(dt_series) - too_early - too_late,
            'too_early': too_early,
            'too_late': too_late,
            'validation_passed': too_early == 0 and too_late == 0
        }
    
    return validation_results

# Example validation
results = validate_date_ranges(
    df, 
    ['timestamp'], 
    min_date='2020-01-01',
    max_date='2025-12-31'
)
```

---

## ⚡ Basic Temporal Features

### Standard Temporal Components

```python
def extract_basic_temporal_features(df, datetime_col):
    """Extract standard temporal features from datetime column"""
    
    dt = pd.to_datetime(df[datetime_col])
    
    # Time components
    df[f'{datetime_col}_year'] = dt.dt.year
    df[f'{datetime_col}_month'] = dt.dt.month
    df[f'{datetime_col}_day'] = dt.dt.day
    df[f'{datetime_col}_dayofweek'] = dt.dt.dayofweek  # 0=Monday
    df[f'{datetime_col}_hour'] = dt.dt.hour
    df[f'{datetime_col}_quarter'] = dt.dt.quarter
    
    # Boolean indicators
    df[f'{datetime_col}_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    df[f'{datetime_col}_is_month_start'] = dt.dt.is_month_start.astype(int)
    df[f'{datetime_col}_is_month_end'] = dt.dt.is_month_end.astype(int)
    df[f'{datetime_col}_is_quarter_end'] = dt.dt.is_quarter_end.astype(int)
    
    # Time periods
    df[f'{datetime_col}_is_morning'] = ((dt.dt.hour >= 6) & (dt.dt.hour < 12)).astype(int)
    df[f'{datetime_col}_is_afternoon'] = ((dt.dt.hour >= 12) & (dt.dt.hour < 18)).astype(int)
    df[f'{datetime_col}_is_evening'] = ((dt.dt.hour >= 18) & (dt.dt.hour < 22)).astype(int)
    df[f'{datetime_col}_is_night'] = ((dt.dt.hour >= 22) | (dt.dt.hour < 6)).astype(int)
    
    return df

# Example usage
df = extract_basic_temporal_features(df, 'timestamp')
```

---

## 📊 Cyclical Feature Engineering

### Sine/Cosine Encoding for Cyclical Patterns

```python
def encode_cyclical_features(df, datetime_col):
    """Encode cyclical temporal features using sine/cosine transformations"""
    
    dt = pd.to_datetime(df[datetime_col])
    
    # Month cyclical encoding (1-12 → sine/cosine)
    df[f'{datetime_col}_month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
    df[f'{datetime_col}_month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
    
    # Day of week cyclical encoding (0-6 → sine/cosine)
    df[f'{datetime_col}_dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df[f'{datetime_col}_dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    
    # Hour cyclical encoding (0-23 → sine/cosine)
    if dt.dt.hour.nunique() > 1:  # Only if hour data exists
        df[f'{datetime_col}_hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df[f'{datetime_col}_hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    
    # Day of year cyclical encoding (1-366 → sine/cosine)
    df[f'{datetime_col}_doy_sin'] = np.sin(2 * np.pi * dt.dt.dayofyear / 366)
    df[f'{datetime_col}_doy_cos'] = np.cos(2 * np.pi * dt.dt.dayofyear / 366)
    
    return df

# Why cyclical encoding matters
def demonstrate_cyclical_importance():
    """Show why cyclical encoding is important"""
    
    # Linear encoding loses cyclical nature
    december = 12
    january = 1
    linear_distance = abs(december - january)  # = 11 (far apart)
    
    # Cyclical encoding preserves cyclical nature
    dec_sin, dec_cos = np.sin(2 * np.pi * 12 / 12), np.cos(2 * np.pi * 12 / 12)
    jan_sin, jan_cos = np.sin(2 * np.pi * 1 / 12), np.cos(2 * np.pi * 1 / 12)
    cyclical_distance = np.sqrt((dec_sin - jan_sin)**2 + (dec_cos - jan_cos)**2)  # ≈ 0.5 (close)
    
    print(f"Linear distance Dec-Jan: {linear_distance}")
    print(f"Cyclical distance Dec-Jan: {cyclical_distance:.3f}")

# Example usage
df = encode_cyclical_features(df, 'timestamp')
```

---

## 📈 Time-Based Aggregations

### Rolling Window Features

```python
def create_rolling_features(df, datetime_col, target_cols, windows=['7D', '30D']):
    """Create rolling window aggregation features"""
    
    # Ensure sorted by datetime
    df = df.sort_values(datetime_col).copy()
    
    for target_col in target_cols:
        for window in windows:
            # Rolling statistics
            df[f'{target_col}_rolling_{window}_mean'] = (
                df[target_col].rolling(window=window, min_periods=1).mean()
            )
            df[f'{target_col}_rolling_{window}_std'] = (
                df[target_col].rolling(window=window, min_periods=1).std()
            )
            df[f'{target_col}_rolling_{window}_min'] = (
                df[target_col].rolling(window=window, min_periods=1).min()
            )
            df[f'{target_col}_rolling_{window}_max'] = (
                df[target_col].rolling(window=window, min_periods=1).max()
            )
    
    return df

def create_seasonal_features(df, datetime_col, target_cols):
    """Create seasonal aggregation features"""
    
    dt = pd.to_datetime(df[datetime_col])
    
    for target_col in target_cols:
        # Monthly averages
        monthly_avg = df.groupby(dt.dt.month)[target_col].mean()
        df[f'{target_col}_month_avg'] = dt.dt.month.map(monthly_avg)
        
        # Day of week averages
        dow_avg = df.groupby(dt.dt.dayofweek)[target_col].mean()
        df[f'{target_col}_dow_avg'] = dt.dt.dayofweek.map(dow_avg)
        
        # Hour averages (if hour data exists)
        if dt.dt.hour.nunique() > 1:
            hour_avg = df.groupby(dt.dt.hour)[target_col].mean()
            df[f'{target_col}_hour_avg'] = dt.dt.hour.map(hour_avg)
    
    return df

# Example usage
df = create_rolling_features(df, 'timestamp', ['sales'])
df = create_seasonal_features(df, 'timestamp', ['sales'])
```

---

## 🔄 Lag and Window Features

### Historical Dependencies

```python
def create_lag_features(df, datetime_col, target_cols, lags=[1, 7, 30], group_col=None):
    """Create lag features safely (avoiding data leakage)"""
    
    # Ensure sorted by datetime
    df = df.sort_values(datetime_col).copy()
    
    for target_col in target_cols:
        for lag in lags:
            if group_col:
                # Create lags within groups
                df[f'{target_col}_lag_{lag}'] = (
                    df.groupby(group_col)[target_col].shift(lag)
                )
            else:
                # Global lags
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df

def create_difference_features(df, datetime_col, target_cols, periods=[1, 7, 30]):
    """Create difference features (current - previous)"""
    
    df = df.sort_values(datetime_col).copy()
    
    for target_col in target_cols:
        for period in periods:
            # Absolute differences
            df[f'{target_col}_diff_{period}'] = (
                df[target_col] - df[target_col].shift(period)
            )
            
            # Percentage changes
            df[f'{target_col}_pct_change_{period}'] = (
                df[target_col].pct_change(periods=period) * 100
            )
    
    return df

# Example usage
df = create_lag_features(df, 'timestamp', ['sales'], lags=[1, 7, 30])
df = create_difference_features(df, 'timestamp', ['sales'])
```

---

## 📋 Business Calendar Features

### Holiday and Business Day Features

```python
def add_business_calendar_features(df, datetime_col, country='US'):
    """Add business calendar and holiday features"""
    
    dt = pd.to_datetime(df[datetime_col])
    
    # Basic business day features
    df[f'{datetime_col}_is_business_day'] = (
        (dt.dt.dayofweek < 5).astype(int)  # Monday-Friday
    )
    
    # Holiday features (simplified - use holidays library for full implementation)
    # Major US holidays (simplified)
    major_holidays = [
        '2024-01-01',  # New Year
        '2024-07-04',  # Independence Day
        '2024-12-25',  # Christmas
    ]
    
    holiday_dates = pd.to_datetime(major_holidays).date
    df[f'{datetime_col}_is_holiday'] = dt.dt.date.isin(holiday_dates).astype(int)
    
    # Days to/from holiday
    def days_to_next_holiday(date_val):
        future_holidays = [h for h in holiday_dates if h > date_val.date()]
        if future_holidays:
            return (min(future_holidays) - date_val.date()).days
        return 365
    
    df[f'{datetime_col}_days_to_holiday'] = dt.apply(days_to_next_holiday)
    
    # Week of month
    df[f'{datetime_col}_week_of_month'] = ((dt.dt.day - 1) // 7) + 1
    
    # Payroll features (bi-weekly Fridays)
    reference_friday = pd.Timestamp('2024-01-05')  # First Friday of year
    days_since_ref = (dt - reference_friday).dt.days
    df[f'{datetime_col}_is_payday'] = (
        ((days_since_ref % 14 == 0) & (dt.dt.dayofweek == 4)).astype(int)
    )
    
    return df

# Example usage
df = add_business_calendar_features(df, 'timestamp')
```

---

## 🎯 Holiday and Event Detection

### Automatic Holiday Detection

```python
def detect_special_events(df, datetime_col, target_col, threshold=2.0):
    """Detect special events based on unusual activity patterns"""
    
    dt = pd.to_datetime(df[datetime_col])
    
    # Calculate rolling mean and std
    rolling_mean = df[target_col].rolling(window=30, min_periods=1).mean()
    rolling_std = df[target_col].rolling(window=30, min_periods=1).std()
    
    # Detect anomalous days (high activity)
    z_scores = (df[target_col] - rolling_mean) / rolling_std
    df[f'{datetime_col}_is_high_activity'] = (z_scores > threshold).astype(int)
    df[f'{datetime_col}_is_low_activity'] = (z_scores < -threshold).astype(int)
    
    # Activity level categories
    df[f'{datetime_col}_activity_level'] = pd.cut(
        z_scores, 
        bins=[-np.inf, -1, 1, np.inf], 
        labels=['low', 'normal', 'high']
    )
    
    return df

# Example usage
df = detect_special_events(df, 'timestamp', 'sales')
```

---

## ⏳ Time Since/Until Features

### Distance-Based Temporal Features

```python
def create_time_distance_features(df, datetime_col, reference_dates=None):
    """Create time distance features from reference points"""
    
    dt = pd.to_datetime(df[datetime_col])
    
    if reference_dates is None:
        # Use common business reference points
        reference_dates = {
            'year_start': dt.dt.year.astype(str) + '-01-01',
            'quarter_start': dt.dt.to_period('Q').dt.start_time,
            'month_start': dt.dt.to_period('M').dt.start_time
        }
    
    for ref_name, ref_dates in reference_dates.items():
        if isinstance(ref_dates, str):
            ref_dates = pd.to_datetime(ref_dates)
        elif isinstance(ref_dates, (list, pd.Series)):
            ref_dates = pd.to_datetime(ref_dates)
        
        # Days since reference
        df[f'{datetime_col}_days_since_{ref_name}'] = (dt - ref_dates).dt.days
        
        # Business days since reference (simplified)
        df[f'{datetime_col}_bdays_since_{ref_name}'] = (
            pd.bdate_range(ref_dates.min(), dt.max()).get_loc(dt, method='ffill')
        )
    
    return df

# Example usage
df = create_time_distance_features(df, 'timestamp')
```

---

## 📊 Seasonal Decomposition

### Trend and Seasonality Extraction

```python
def extract_seasonal_components(df, datetime_col, target_col, period=365):
    """Extract trend and seasonal components from time series"""
    
    # Simple trend extraction using rolling mean
    df[f'{target_col}_trend'] = (
        df[target_col].rolling(window=period//4, center=True, min_periods=1).mean()
    )
    
    # Seasonal component (simplified)
    dt = pd.to_datetime(df[datetime_col])
    
    # Annual seasonality
    seasonal_annual = df.groupby(dt.dt.dayofyear)[target_col].mean()
    df[f'{target_col}_seasonal_annual'] = dt.dt.dayofyear.map(seasonal_annual)
    
    # Weekly seasonality
    seasonal_weekly = df.groupby(dt.dt.dayofweek)[target_col].mean()
    df[f'{target_col}_seasonal_weekly'] = dt.dt.dayofweek.map(seasonal_weekly)
    
    # Residual (actual - trend - seasonal)
    df[f'{target_col}_residual'] = (
        df[target_col] - df[f'{target_col}_trend'] - 
        df[f'{target_col}_seasonal_annual'] - df[f'{target_col}_seasonal_weekly']
    )
    
    return df

# Example usage
df = extract_seasonal_components(df, 'timestamp', 'sales')
```

---

## 💻 Complete Pipeline Implementation

### Comprehensive Temporal Feature Pipeline

```python
class TemporalFeaturePipeline:
    def __init__(self, country='US'):
        self.country = country
        self.processing_log = {}
    
    def process_temporal_data(self, df, datetime_cols, target_cols=None, **kwargs):
        """Complete temporal feature engineering pipeline"""
        
        processed_df = df.copy()
        
        print("=== TEMPORAL FEATURE ENGINEERING PIPELINE ===")
        
        for datetime_col in datetime_cols:
            print(f"\nProcessing column: {datetime_col}")
            
            # 1. Parse and standardize
            parsed_col = self._parse_datetime_column(processed_df[datetime_col])
            processed_df[datetime_col] = parsed_col
            
            # 2. Standardize timezone
            processed_df = self._standardize_timezone(processed_df, datetime_col)
            
            # 3. Extract basic features
            processed_df = extract_basic_temporal_features(processed_df, datetime_col)
            
            # 4. Encode cyclical features
            processed_df = encode_cyclical_features(processed_df, datetime_col)
            
            # 5. Add business calendar features
            processed_df = add_business_calendar_features(processed_df, datetime_col)
            
            # 6. Create aggregation features (if target columns provided)
            if target_cols:
                processed_df = create_rolling_features(
                    processed_df, datetime_col, target_cols,
                    windows=kwargs.get('rolling_windows', ['7D', '30D'])
                )
                processed_df = create_seasonal_features(processed_df, datetime_col, target_cols)
                
                # 7. Create lag features
                processed_df = create_lag_features(
                    processed_df, datetime_col, target_cols,
                    lags=kwargs.get('lag_periods', [1, 7, 30]),
                    group_col=kwargs.get('group_col')
                )
                
                # 8. Create difference features
                processed_df = create_difference_features(
                    processed_df, datetime_col, target_cols
                )
        
        # Summary
        original_features = len(df.columns)
        final_features = len(processed_df.columns)
        features_created = final_features - original_features
        
        print(f"\n=== PIPELINE COMPLETED ===")
        print(f"Features created: {features_created}")
        print(f"Final shape: {processed_df.shape}")
        
        return processed_df
    
    def _parse_datetime_column(self, series):
        """Parse datetime column with error handling"""
        try:
            return pd.to_datetime(series, errors='coerce')
        except Exception as e:
            print(f"Warning: DateTime parsing failed: {e}")
            return series
    
    def _standardize_timezone(self, df, datetime_col):
        """Standardize timezone to UTC"""
        try:
            dt_series = pd.to_datetime(df[datetime_col])
            if dt_series.dt.tz is None:
                dt_series = dt_series.dt.tz_localize('UTC')
            else:
                dt_series = dt_series.dt.tz_convert('UTC')
            df[datetime_col] = dt_series
        except Exception as e:
            print(f"Warning: Timezone standardization failed: {e}")
        
        return df

# Example usage
pipeline = TemporalFeaturePipeline()

# Sample data
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
sample_df = pd.DataFrame({
    'date': dates,
    'sales': np.random.uniform(100, 1000, len(dates)) + 
             50 * np.sin(2 * np.pi * np.arange(len(dates)) / 30),  # Monthly pattern
    'category': np.random.choice(['A', 'B', 'C'], len(dates))
})

# Process through pipeline
enhanced_df = pipeline.process_temporal_data(
    sample_df,
    datetime_cols=['date'],
    target_cols=['sales'],
    rolling_windows=['7D', '30D'],
    lag_periods=[1, 7, 14, 30]
)

print(f"\nSample of created features:")
new_features = [col for col in enhanced_df.columns if col not in sample_df.columns]
for i, feature in enumerate(new_features[:10]):  # Show first 10 features
    print(f"  {i+1:2d}. {feature}")
```

---

## ✅ Temporal Feature Validation

### Quality Assurance for Temporal Features

```python
def validate_temporal_features(df, datetime_cols):
    """Validate quality of temporal features"""
    
    validation_results = {}
    
    for col in datetime_cols:
        dt_series = pd.to_datetime(df[col])
        
        # Basic validation
        validation_results[col] = {
            'data_type': str(dt_series.dtype),
            'null_count': dt_series.isnull().sum(),
            'unique_count': dt_series.nunique(),
            'is_sorted': dt_series.is_monotonic_increasing,
            'date_range': (dt_series.min(), dt_series.max()),
            'has_timezone': dt_series.dt.tz is not None
        }
        
        # Check for duplicates
        duplicates = dt_series.duplicated().sum()
        validation_results[col]['duplicates'] = duplicates
        
        # Check for irregular intervals
        if len(dt_series.dropna()) > 1:
            intervals = dt_series.dropna().diff().dropna()
            validation_results[col]['interval_consistency'] = {
                'median_interval': intervals.median(),
                'std_interval': intervals.std(),
                'irregular_intervals': len(intervals.value_counts()) > 10
            }
    
    return validation_results

def validate_cyclical_encoding(df):
    """Validate cyclical encoding correctness"""
    
    cyclical_features = {}
    sin_features = [col for col in df.columns if '_sin' in col]
    
    for sin_col in sin_features:
        cos_col = sin_col.replace('_sin', '_cos')
        if cos_col in df.columns:
            sin_vals = df[sin_col]
            cos_vals = df[cos_col]
            
            # Check sin² + cos² ≈ 1
            magnitude_check = sin_vals**2 + cos_vals**2
            magnitude_error = abs(magnitude_check - 1).mean()
            
            feature_name = sin_col.replace('_sin', '')
            cyclical_features[feature_name] = {
                'magnitude_error': magnitude_error,
                'encoding_valid': magnitude_error < 0.01
            }
    
    return cyclical_features

# Example validation
validation_report = validate_temporal_features(enhanced_df, ['date'])
cyclical_report = validate_cyclical_encoding(enhanced_df)

print("Temporal Validation Report:")
for col, results in validation_report.items():
    print(f"\n{col}:")
    for key, value in results.items():
        print(f"  {key}: {value}")
```

---

## ⭐ Best Practices

### 1. **Timezone-Aware Processing**

```python
# ✅ Always store in UTC, convert for analysis
def best_practice_timezone():
    # Store all timestamps in UTC
    utc_data = df['timestamp'].dt.tz_convert('UTC')
    
    # Convert to business timezone for analysis
    business_time = utc_data.dt.tz_convert('America/New_York')
    
    # Create business hour features using local time
    df['is_business_hours'] = business_time.dt.hour.between(9, 17)
```

### 2. **Avoid Data Leakage in Lag Features**

```python
# ✅ Use expanding windows, not full dataset statistics
def safe_lag_features(df, target_col):
    # Wrong: uses future information
    # df['target_mean'] = df[target_col].mean()
    
    # Correct: expanding window excludes future
    df['target_expanding_mean'] = df[target_col].expanding().mean().shift(1)
    
    return df
```

### 3. **Memory-Efficient Processing**

```python
# ✅ Optimize data types for temporal features
def optimize_temporal_dtypes(df):
    # Boolean features to int8
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype('int8')
    
    # Downcast numerical features
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
    
    return df
```

---

## ⚠️ Common Pitfalls

### 1. **Data Leakage in Time Series**

```python
# ❌ Wrong: Using future information
def bad_feature_creation(df):
    # This uses ALL data to calculate mean - includes future!
    df['sales_vs_mean'] = df['sales'] - df['sales'].mean()
    return df

# ✅ Correct: Only use past information
def safe_feature_creation(df):
    # Use expanding mean that only includes past values
    df['sales_vs_expanding_mean'] = (
        df['sales'] - df['sales'].expanding().mean().shift(1)
    )
    return df
```

### 2. **Ignoring Business Context for Timezone**

```python
# ❌ Wrong: Using UTC for business hour calculation
def bad_business_hours(df):
    df['is_business_hours'] = df['utc_timestamp'].dt.hour.between(9, 17)
    return df

# ✅ Correct: Convert to business timezone first
def correct_business_hours(df):
    local_time = df['utc_timestamp'].dt.tz_convert('America/New_York')
    df['is_business_hours'] = local_time.dt.hour.between(9, 17)
    return df
```

### 3. **Over-Engineering Cyclical Features**

```python
# ❌ Wrong: Creating unnecessary cyclical features
def excessive_cyclical(df):
    dt = df['timestamp']
    # Creating cyclical features for every possible time unit
    df['second_sin'] = np.sin(2 * np.pi * dt.dt.second / 60)
    df['second_cos'] = np.cos(2 * np.pi * dt.dt.second / 60)
    # ... many more that may not be useful

# ✅ Better: Focus on meaningful cycles for your domain
def focused_cyclical(df):
    dt = df['timestamp']
    # Only create cyclical features that match your data and domain
    if dt.dt.hour.nunique() > 1:  # Only if hour-level data
        df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    
    # Monthly cycles are often meaningful
    df['month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
    
    return df
```

---

## 📝 Summary

Temporal feature engineering is a powerful technique that can significantly enhance machine learning model performance by extracting meaningful patterns from date and time information. This guide covered all essential aspects of temporal data processing and feature creation.

### Key Takeaways

1. **Data Quality First**: Always start with proper parsing, timezone standardization, and validation before feature engineering

2. **Domain-Driven Engineering**: Create features that align with your business domain and known temporal patterns

3. **Avoid Data Leakage**: Be extremely careful with lag features and time-based aggregations to prevent future information leakage

4. **Cyclical Encoding**: Use sine/cosine transformations for truly cyclical features like time of day, day of week, and month

5. **Business Calendar Integration**: Incorporate holidays, business days, and industry-specific events for more relevant features

### Feature Engineering Strategy

- **Basic Features**: Start with standard datetime components (year, month, day, etc.)
- **Cyclical Features**: Encode periodic patterns mathematically
- **Lag Features**: Capture historical dependencies safely
- **Aggregation Features**: Create rolling and seasonal statistics
- **Business Features**: Add domain-specific temporal knowledge

### Production Considerations

- **Performance**: Use efficient data types and chunked processing for large datasets
- **Validation**: Implement comprehensive validation to catch feature engineering errors
- **Monitoring**: Track temporal feature drift in production environments
- **Documentation**: Maintain clear documentation of temporal assumptions and business rules

### Advanced Techniques Covered

- **Seasonal Decomposition**: Separate trend, seasonal, and residual components
- **Holiday Detection**: Automatic identification of holidays and special events
- **Business Calendar Features**: Integration with payroll schedules and fiscal calendars
- **Advanced Lag Features**: Momentum, volatility, and trend strength calculations

### Next Steps

With comprehensive temporal features, you're ready for:

- **Feature Selection**: Identifying the most predictive temporal features
- **Time Series Modeling**: Building sophisticated forecasting models
- **Anomaly Detection**: Using temporal patterns to identify unusual events
- **Real-time Systems**: Implementing temporal features in streaming environments

### Final Recommendations

1. **Start Simple**: Begin with basic temporal features and add complexity incrementally
2. **Validate Impact**: Always measure the performance impact of new temporal features
3. **Think Business Context**: Consider how time affects your specific business problem
4. **Handle Edge Cases**: Plan for timezone changes, data gaps, and irregular patterns
5. **Monitor Continuously**: Temporal patterns can change over time - monitor and adapt

Remember: Effective temporal feature engineering requires understanding both the technical aspects of time series data and the business context in which temporal patterns occur. The most successful implementations combine statistical rigor with domain expertise.

---

**Related Guides in This Series:**

- [Exploratory Data Analysis](./exploratory_data_analysis.md)
- [Missing Data Imputation](./missing_data_imputation.md)
- [Duplication and Outlier Handling](./duplication_outlier_handling.md)
- [Categorical Variable Cleaning](./categorical_variable_cleaning.md)
- [Numerical Variable Cleaning](./numerical_variable_cleaning.md)
- Advanced Feature Engineering and Selection (coming soon)
- Model Selection and Validation (coming soon)

---

## 📖 Overview

Temporal data is one of the richest sources of information in machine learning, yet it's often underutilized due to preprocessing complexity. This guide provides a comprehensive framework for extracting maximum value from date and time information through systematic standardization and feature engineering.

### Why Temporal Feature Engineering Matters

**Business Impact:**
- **Seasonal Patterns**: Capture recurring business cycles and seasonal trends
- **Operational Insights**: Understand time-based operational patterns
- **Forecasting Power**: Enable sophisticated time series modeling
- **Customer Behavior**: Model temporal aspects of user interactions

**Technical Benefits:**
- **Rich Feature Space**: Extract dozens of meaningful features from single datetime column
- **Model Performance**: Significantly improve predictive accuracy through temporal insights
- **Pattern Recognition**: Help algorithms detect complex temporal relationships
- **Data Quality**: Standardize and validate temporal data consistency

### What This Guide Covers

1. **Data Standardization**: Parsing, timezone handling, and validation of temporal data
2. **Basic Features**: Standard temporal components (year, month, day, etc.)
3. **Cyclical Encoding**: Handle periodic patterns with mathematical transformations
4. **Aggregation Features**: Time-based statistical summaries and rolling windows
5. **Advanced Features**: Business calendars, holidays, lag features, and seasonal decomposition
6. **Production Pipeline**: Complete, reusable implementation for real-world deployment

### Prerequisites

- Basic understanding of pandas datetime functionality
- Familiarity with feature engineering concepts
- Knowledge of time series analysis principles (helpful but not required)

---

## 🔍 Understanding Temporal Data

### Types of Temporal Data

**1. Timestamps**
```python
# Point-in-time events
"2024-03-15 14:30:00"
"2024-03-15T14:30:00Z"  # ISO format with timezone
```

**2. Date-only Data**
```python
# Calendar dates without time component
"2024-03-15"
"March 15, 2024"
```

**3. Time-only Data**
```python
# Time components without date
"14:30:00"
"2:30 PM"
```

**4. Duration/Intervals**
```python
# Time spans
"2 hours 30 minutes"
pd.Timedelta('2 days 3 hours')
```

**5. Relative Time References**
```python
# Business-relative time
"Q1 2024"
"Week 12"
"2 days ago"
```

### Temporal Data Characteristics

**Granularity Levels:**
- **Second-level**: High-frequency trading, sensor data, web analytics
- **Minute-level**: Call center data, manufacturing processes
- **Hour-level**: Energy consumption, web traffic patterns
- **Day-level**: Sales data, user activity, financial markets
- **Week/Month-level**: Business reporting, seasonal analysis
- **Year-level**: Long-term trends, demographic data

**Temporal Patterns:**
- **Linear Trends**: Consistent growth or decline over time
- **Cyclical Patterns**: Repeating patterns (daily, weekly, monthly, yearly)
- **Seasonal Effects**: Weather-dependent or calendar-driven variations
- **Event-driven Spikes**: Holidays, promotions, external events
- **Regime Changes**: Structural breaks in temporal patterns

**Time Zone Considerations:**
- **UTC Standardization**: Global data requires consistent timezone handling
- **Local Time Relevance**: Business operations often depend on local time
- **Daylight Saving**: Automatic adjustments can create data irregularities
- **Cross-timezone Analysis**: Multi-region datasets require careful synchronization

---

## ⚠️ Common Temporal Data Issues

### 1. **Inconsistent Formats and Standards**

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

class TemporalDataProfiler:
    def __init__(self, df):
        self.df = df
        self.temporal_columns = self._identify_temporal_columns()
        self.analysis_results = {}
    
    def _identify_temporal_columns(self):
        """Automatically identify potential temporal columns"""
        temporal_cols = []
        
        for col in self.df.columns:
            # Check dtype
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                temporal_cols.append({
                    'column': col,
                    'type': 'datetime',
                    'confidence': 1.0
                })
                continue
            
            # Check object columns for date-like strings
            if self.df[col].dtype == 'object':
                sample_values = self.df[col].dropna().head(100)
                date_like_count = 0
                
                for value in sample_values:
                    if isinstance(value, str):
                        # Common date patterns
                        if any(pattern in str(value).lower() for pattern in 
                              ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                            date_like_count += 1
                        elif any(char in str(value) for char in ['-', '/', ':']):
                            try:
                                pd.to_datetime(value)
                                date_like_count += 1
                            except:
                                pass
                
                confidence = date_like_count / len(sample_values) if sample_values.any() else 0
                if confidence > 0.5:
                    temporal_cols.append({
                        'column': col,
                        'type': 'potential_datetime',
                        'confidence': confidence
                    })
        
        return temporal_cols
    
    def detect_format_inconsistencies(self):
        """Detect inconsistent datetime formats within columns"""
        
        format_issues = {}
        
        for col_info in self.temporal_columns:
            col = col_info['column']
            
            if col_info['type'] == 'potential_datetime':
                # Sample values to detect format patterns
                sample_values = self.df[col].dropna().astype(str).head(1000)
                
                format_patterns = {
                    'ISO_8601': 0,
                    'US_format': 0,  # MM/DD/YYYY
                    'EU_format': 0,  # DD/MM/YYYY
                    'timestamp': 0,
                    'other': 0
                }
                
                for value in sample_values:
                    if 'T' in value and ('Z' in value or '+' in value):
                        format_patterns['ISO_8601'] += 1
                    elif '/' in value:
                        parts = value.split('/')
                        if len(parts) == 3:
                            # Heuristic: if first part > 12, likely DD/MM/YYYY
                            try:
                                first_part = int(parts[0])
                                if first_part > 12:
                                    format_patterns['EU_format'] += 1
                                else:
                                    format_patterns['US_format'] += 1
                            except:
                                format_patterns['other'] += 1
                    elif value.isdigit() and len(value) == 10:
                        format_patterns['timestamp'] += 1
                    else:
                        format_patterns['other'] += 1
                
                # Check for mixed formats (inconsistency indicator)
                non_zero_formats = sum(1 for count in format_patterns.values() if count > 0)
                if non_zero_formats > 1:
                    format_issues[col] = {
                        'mixed_formats': True,
                        'format_distribution': format_patterns,
                        'dominant_format': max(format_patterns, key=format_patterns.get)
                    }
        
        return format_issues
    
    def detect_timezone_issues(self):
        """Detect timezone-related problems"""
        
        timezone_issues = {}
        
        for col_info in self.temporal_columns:
            col = col_info['column']
            
            if col_info['type'] == 'datetime':
                dt_series = self.df[col]
                
                # Check if timezone-aware
                if hasattr(dt_series.dtype, 'tz') and dt_series.dtype.tz is not None:
                    timezone_issues[col] = {
                        'timezone_aware': True,
                        'timezone': str(dt_series.dtype.tz),
                        'mixed_timezones': False  # pandas doesn't allow mixed tz in series
                    }
                else:
                    # Check for timezone indicators in string data
                    sample_strings = self.df[col].astype(str).head(100)
                    tz_indicators = sum(1 for s in sample_strings 
                                      if any(tz in s for tz in ['UTC', 'GMT', '+', 'Z']))
                    
                    timezone_issues[col] = {
                        'timezone_aware': False,
                        'potential_tz_info': tz_indicators > 0,
                        'tz_indicator_ratio': tz_indicators / len(sample_strings)
                    }
        
        return timezone_issues
    
    def detect_temporal_gaps_and_irregularities(self):
        """Detect missing time periods and irregular intervals"""
        
        gap_analysis = {}
        
        for col_info in self.temporal_columns:
            col = col_info['column']
            
            try:
                # Convert to datetime if not already
                if col_info['type'] == 'potential_datetime':
                    dt_series = pd.to_datetime(self.df[col], errors='coerce')
                else:
                    dt_series = self.df[col]
                
                # Remove NaT values
                clean_series = dt_series.dropna().sort_values()
                
                if len(clean_series) < 2:
                    continue
                
                # Calculate intervals
                intervals = clean_series.diff().dropna()
                
                # Basic statistics
                median_interval = intervals.median()
                interval_std = intervals.std()
                
                # Detect likely frequency
                common_intervals = intervals.value_counts().head(3)
                most_common_interval = common_intervals.index[0] if len(common_intervals) > 0 else None
                
                # Detect gaps (intervals much larger than normal)
                if median_interval.total_seconds() > 0:
                    gap_threshold = median_interval * 3  # 3x normal interval
                    gaps = intervals[intervals > gap_threshold]
                    
                    gap_analysis[col] = {
                        'total_records': len(clean_series),
                        'date_range': (clean_series.min(), clean_series.max()),
                        'median_interval': median_interval,
                        'most_common_interval': most_common_interval,
                        'irregular_intervals': len(intervals.value_counts()) > len(intervals) * 0.1,
                        'gaps_detected': len(gaps),
                        'largest_gap': gaps.max() if len(gaps) > 0 else None,
                        'interval_consistency': interval_std / median_interval if median_interval.total_seconds() > 0 else np.inf
                    }
            
            except Exception as e:
                gap_analysis[col] = {'error': str(e)}
        
        return gap_analysis
    
    def comprehensive_temporal_analysis(self):
        """Run complete temporal data analysis"""
        
        print("=== TEMPORAL DATA ANALYSIS ===")
        print(f"Identified {len(self.temporal_columns)} temporal columns:")
        
        for col_info in self.temporal_columns:
            print(f"  - {col_info['column']}: {col_info['type']} (confidence: {col_info['confidence']:.2f})")
        
        # Run all analyses
        format_issues = self.detect_format_inconsistencies()
        timezone_issues = self.detect_timezone_issues()
        gap_analysis = self.detect_temporal_gaps_and_irregularities()
        
        self.analysis_results = {
            'temporal_columns': self.temporal_columns,
            'format_issues': format_issues,
            'timezone_issues': timezone_issues,
            'gap_analysis': gap_analysis
        }
        
        # Print summary
        if format_issues:
            print(f"\n⚠️  Format inconsistencies detected in {len(format_issues)} columns")
            for col, issues in format_issues.items():
                print(f"    {col}: {issues['dominant_format']} (mixed formats)")
        
        if timezone_issues:
            print(f"\n🌍 Timezone considerations for {len(timezone_issues)} columns")
            for col, issues in timezone_issues.items():
                tz_status = "aware" if issues['timezone_aware'] else "naive"
                print(f"    {col}: {tz_status}")
        
        if gap_analysis:
            print(f"\n📊 Temporal patterns analysis:")
            for col, analysis in gap_analysis.items():
                if 'error' not in analysis:
                    print(f"    {col}: {analysis['total_records']} records, "
                          f"{analysis['gaps_detected']} gaps detected")
        
        return self.analysis_results
```

### 2. **Missing and Invalid Temporal Data**

```python
class TemporalDataCleaner:
    def __init__(self):
        self.cleaning_log = []
    
    def detect_invalid_dates(self, df, date_columns):
        """Detect various types of invalid dates"""
        
        invalid_patterns = {}
        
        for col in date_columns:
            invalid_data = {
                'null_values': df[col].isnull().sum(),
                'empty_strings': (df[col] == '').sum() if df[col].dtype == 'object' else 0,
                'future_dates': 0,
                'ancient_dates': 0,
                'invalid_formats': 0,
                'outlier_dates': []
            }
            
            # Check for unrealistic dates
            try:
                dt_series = pd.to_datetime(df[col], errors='coerce')
                
                # Future dates (beyond reasonable business range)
                future_threshold = pd.Timestamp.now() + pd.Timedelta(days=365*5)  # 5 years
                invalid_data['future_dates'] = (dt_series > future_threshold).sum()
                
                # Ancient dates (before reasonable business range)
                ancient_threshold = pd.Timestamp('1900-01-01')
                invalid_data['ancient_dates'] = (dt_series < ancient_threshold).sum()
                
                # Invalid formats (couldn't parse)
                invalid_data['invalid_formats'] = dt_series.isnull().sum() - df[col].isnull().sum()
                
                # Statistical outliers in date distribution
                if len(dt_series.dropna()) > 0:
                    numeric_dates = dt_series.dropna().astype('int64')  # nanoseconds since epoch
                    Q1 = numeric_dates.quantile(0.25)
                    Q3 = numeric_dates.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    outlier_mask = (numeric_dates < Q1 - 3*IQR) | (numeric_dates > Q3 + 3*IQR)
                    outlier_dates = dt_series.dropna()[outlier_mask]
                    invalid_data['outlier_dates'] = outlier_dates.tolist()
            
            except Exception as e:
                invalid_data['processing_error'] = str(e)
            
            invalid_patterns[col] = invalid_data
        
        return invalid_patterns
    
    def standardize_date_formats(self, df, date_columns, target_format='%Y-%m-%d %H:%M:%S'):
        """Standardize date formats across columns"""
        
        standardized_df = df.copy()
        conversion_log = {}
        
        for col in date_columns:
            original_series = df[col]
            
            try:
                # Attempt automatic parsing with pandas
                standardized_series = pd.to_datetime(original_series, errors='coerce')
                
                # Count successful conversions
                successful_conversions = standardized_series.notna().sum()
                total_non_null = original_series.notna().sum()
                
                conversion_log[col] = {
                    'success_rate': successful_conversions / total_non_null if total_non_null > 0 else 0,
                    'converted_count': successful_conversions,
                    'failed_count': total_non_null - successful_conversions,
                    'original_dtype': str(original_series.dtype),
                    'final_dtype': str(standardized_series.dtype)
                }
                
                # Store standardized series
                standardized_df[col] = standardized_series
                
                self.cleaning_log.append({
                    'column': col,
                    'action': 'standardize_format',
                    'success_rate': conversion_log[col]['success_rate'],
                    'timestamp': pd.Timestamp.now()
                })
            
            except Exception as e:
                conversion_log[col] = {'error': str(e)}
                self.cleaning_log.append({
                    'column': col,
                    'action': 'standardize_format',
                    'error': str(e),
                    'timestamp': pd.Timestamp.now()
                })
        
        return standardized_df, conversion_log
    
    def handle_missing_temporal_data(self, df, date_columns, strategy='drop'):
        """Handle missing temporal data with various strategies"""
        
        handled_df = df.copy()
        handling_log = {}
        
        for col in date_columns:
            missing_count = df[col].isnull().sum()
            
            if missing_count == 0:
                handling_log[col] = {'missing_count': 0, 'action': 'no_action_needed'}
                continue
            
            if strategy == 'drop':
                # Drop rows with missing dates
                before_count = len(handled_df)
                handled_df = handled_df.dropna(subset=[col])
                after_count = len(handled_df)
                
                handling_log[col] = {
                    'missing_count': missing_count,
                    'action': 'drop_rows',
                    'rows_dropped': before_count - after_count
                }
            
            elif strategy == 'interpolate':
                # Interpolate missing dates (for time series)
                if handled_df[col].dtype.kind == 'M':  # datetime
                    handled_df[col] = handled_df[col].interpolate(method='time')
                    
                    handling_log[col] = {
                        'missing_count': missing_count,
                        'action': 'interpolate',
                        'remaining_missing': handled_df[col].isnull().sum()
                    }
            
            elif strategy == 'forward_fill':
                # Forward fill missing dates
                handled_df[col] = handled_df[col].fillna(method='ffill')
                
                handling_log[col] = {
                    'missing_count': missing_count,
                    'action': 'forward_fill',
                    'remaining_missing': handled_df[col].isnull().sum()
                }
            
            elif strategy == 'median_impute':
                # Use median date for imputation
                if handled_df[col].dtype.kind == 'M':
                    median_date = handled_df[col].median()
                    handled_df[col] = handled_df[col].fillna(median_date)
                    
                    handling_log[col] = {
                        'missing_count': missing_count,
                        'action': 'median_impute',
                        'impute_value': median_date,
                        'remaining_missing': handled_df[col].isnull().sum()
                    }
            
            self.cleaning_log.append({
                'column': col,
                'action': f'handle_missing_{strategy}',
                'missing_count': missing_count,
                'timestamp': pd.Timestamp.now()
            })
        
        return handled_df, handling_log

# Example usage
def demonstrate_temporal_data_issues():
    """Demonstrate common temporal data quality issues"""
    
    # Create sample data with various issues
    problematic_data = {
        'mixed_formats': [
            '2024-03-15 14:30:00',    # ISO format
            '03/15/2024 2:30 PM',     # US format
            '15/03/2024 14:30',       # EU format
            '1710505800',             # Unix timestamp
            '2024-03-15T14:30:00Z',   # ISO with timezone
            'March 15, 2024',         # Text format
            '',                       # Empty string
            np.nan,                   # Missing value
            '2050-12-31',            # Future date
            '1800-01-01'             # Ancient date
        ],
        'customer_id': range(10),
        'sales_amount': np.random.uniform(10, 1000, 10)
    }
    
    df = pd.DataFrame(problematic_data)
    
    # Analyze the data
    profiler = TemporalDataProfiler(df)
    analysis = profiler.comprehensive_temporal_analysis()
    
    # Clean the data
    cleaner = TemporalDataCleaner()
    invalid_patterns = cleaner.detect_invalid_dates(df, ['mixed_formats'])
    
    print("\nInvalid Date Patterns:")
    for col, patterns in invalid_patterns.items():
        print(f"  {col}:")
        for pattern_type, count in patterns.items():
            if isinstance(count, (int, float)) and count > 0:
                print(f"    {pattern_type}: {count}")
    
    # Standardize formats
    clean_df, conversion_log = cleaner.standardize_date_formats(df, ['mixed_formats'])
    
    print("\nFormat Standardization Results:")
    for col, log in conversion_log.items():
        if 'success_rate' in log:
            print(f"  {col}: {log['success_rate']:.2%} success rate")
            print(f"    Converted: {log['converted_count']}, Failed: {log['failed_count']}")
    
    return df, clean_df, analysis

# Run demonstration
# original_df, cleaned_df, analysis_results = demonstrate_temporal_data_issues()
```

---

## 🔧 Date/Time Parsing and Standardization

### Universal DateTime Parser

```python
import pytz
from dateutil import parser as date_parser
import re
from typing import List, Dict, Optional, Union

class UniversalDateTimeParser:
    def __init__(self):
        self.common_formats = [
            '%Y-%m-%d %H:%M:%S',      # ISO-like
            '%Y-%m-%d',               # Date only
            '%d/%m/%Y %H:%M:%S',      # EU format with time
            '%d/%m/%Y',               # EU date only
            '%m/%d/%Y %H:%M:%S',      # US format with time
            '%m/%d/%Y',               # US date only
            '%Y/%m/%d %H:%M:%S',      # Alternative with slashes
            '%Y/%m/%d',               # Alternative date only
            '%d-%m-%Y %H:%M:%S',      # EU with dashes
            '%d-%m-%Y',               # EU date with dashes
            '%B %d, %Y',              # Month name format
            '%d %B %Y',               # Day month year
            '%Y-%m-%dT%H:%M:%S',      # ISO without timezone
            '%Y-%m-%dT%H:%M:%SZ',     # ISO with Z
            '%Y-%m-%dT%H:%M:%S%z',    # ISO with timezone offset
        ]
        self.parsing_stats = {}
    
    def detect_format_pattern(self, date_strings: List[str]) -> Dict:
        """Detect the most likely format pattern in a list of date strings"""
        
        format_scores = {}
        sample_size = min(100, len(date_strings))
        sample_strings = [s for s in date_strings[:sample_size] if pd.notna(s) and str(s).strip()]
        
        for fmt in self.common_formats:
            successful_parses = 0
            
            for date_str in sample_strings:
                try:
                    datetime.strptime(str(date_str).strip(), fmt)
                    successful_parses += 1
                except ValueError:
                    continue
            
            if sample_strings:
                success_rate = successful_parses / len(sample_strings)
                if success_rate > 0:
                    format_scores[fmt] = success_rate
        
        # Also try dateutil parser for flexibility
        dateutil_success = 0
        for date_str in sample_strings:
            try:
                date_parser.parse(str(date_str).strip())
                dateutil_success += 1
            except:
                continue
        
        if sample_strings:
            dateutil_rate = dateutil_success / len(sample_strings)
            if dateutil_rate > 0:
                format_scores['dateutil_parser'] = dateutil_rate
        
        return {
            'format_scores': format_scores,
            'best_format': max(format_scores, key=format_scores.get) if format_scores else None,
            'confidence': max(format_scores.values()) if format_scores else 0,
            'sample_size': len(sample_strings)
        }
    
    def parse_datetime_column(self, series: pd.Series, 
                            preferred_format: Optional[str] = None,
                            timezone: Optional[str] = None) -> Dict:
        """Parse a datetime column with comprehensive error handling"""
        
        original_series = series.copy()
        parsed_series = pd.Series(index=series.index, dtype='datetime64[ns]')
        parsing_log = {
            'total_values': len(series),
            'null_values': series.isnull().sum(),
            'successful_parses': 0,
            'failed_parses': 0,
            'format_used': None,
            'timezone_applied': timezone,
            'parsing_errors': []
        }
        
        # Clean the series first
        clean_series = series.astype(str).str.strip()
        clean_series = clean_series.replace(['', 'nan', 'None', 'null'], np.nan)
        non_null_series = clean_series.dropna()
        
        if len(non_null_series) == 0:
            parsing_log['format_used'] = 'no_valid_data'
            return {'parsed_series': parsed_series, 'log': parsing_log}
        
        # Try preferred format first
        if preferred_format:
            try:
                temp_series = pd.to_datetime(non_null_series, format=preferred_format, errors='coerce')
                success_rate = temp_series.notna().sum() / len(temp_series)
                
                if success_rate > 0.8:  # 80% success threshold
                    parsed_series.loc[non_null_series.index] = temp_series
                    parsing_log['successful_parses'] = temp_series.notna().sum()
                    parsing_log['failed_parses'] = temp_series.isna().sum()
                    parsing_log['format_used'] = preferred_format
                    
                    # Apply timezone if specified
                    if timezone and parsed_series.notna().any():
                        parsed_series = parsed_series.dt.tz_localize(timezone, errors='coerce')
                    
                    return {'parsed_series': parsed_series, 'log': parsing_log}
            except Exception as e:
                parsing_log['parsing_errors'].append(f"Preferred format failed: {str(e)}")
        
        # Auto-detect format
        format_detection = self.detect_format_pattern(non_null_series.tolist())
        best_format = format_detection['best_format']
        
        if best_format and best_format != 'dateutil_parser':
            try:
                temp_series = pd.to_datetime(non_null_series, format=best_format, errors='coerce')
                parsed_series.loc[non_null_series.index] = temp_series
                parsing_log['successful_parses'] = temp_series.notna().sum()
                parsing_log['failed_parses'] = temp_series.isna().sum()
                parsing_log['format_used'] = best_format
            except Exception as e:
                parsing_log['parsing_errors'].append(f"Auto-detected format failed: {str(e)}")
                # Fall back to pandas auto-parsing
                temp_series = pd.to_datetime(non_null_series, errors='coerce')
                parsed_series.loc[non_null_series.index] = temp_series
                parsing_log['successful_parses'] = temp_series.notna().sum()
                parsing_log['failed_parses'] = temp_series.isna().sum()
                parsing_log['format_used'] = 'pandas_auto'
        else:
            # Use pandas auto-parsing or dateutil
            temp_series = pd.to_datetime(non_null_series, errors='coerce')
            parsed_series.loc[non_null_series.index] = temp_series
            parsing_log['successful_parses'] = temp_series.notna().sum()
            parsing_log['failed_parses'] = temp_series.isna().sum()
            parsing_log['format_used'] = 'pandas_auto'
        
        # Apply timezone if specified and not already timezone-aware
        if timezone and parsed_series.notna().any():
            try:
                if parsed_series.dt.tz is None:
                    parsed_series = parsed_series.dt.tz_localize(timezone, errors='coerce')
                else:
                    parsed_series = parsed_series.dt.tz_convert(timezone)
            except Exception as e:
                parsing_log['parsing_errors'].append(f"Timezone handling failed: {str(e)}")
        
        return {'parsed_series': parsed_series, 'log': parsing_log}
    
    def batch_parse_datetime_columns(self, df: pd.DataFrame, 
                                   datetime_columns: List[str],
                                   column_formats: Dict[str, str] = None,
                                   default_timezone: str = None) -> Dict:
        """Parse multiple datetime columns in batch"""
        
        parsed_df = df.copy()
        batch_log = {}
        
        column_formats = column_formats or {}
        
        for col in datetime_columns:
            if col not in df.columns:
                batch_log[col] = {'error': 'Column not found in DataFrame'}
                continue
            
            preferred_format = column_formats.get(col)
            
            result = self.parse_datetime_column(
                df[col], 
                preferred_format=preferred_format,
                timezone=default_timezone
            )
            
            parsed_df[col] = result['parsed_series']
            batch_log[col] = result['log']
        
        return {'parsed_df': parsed_df, 'batch_log': batch_log}
    
    def generate_parsing_report(self, batch_log: Dict) -> str:
        """Generate a comprehensive parsing report"""
        
        report = "DateTime Parsing Report\n"
        report += "=" * 50 + "\n\n"
        
        total_columns = len(batch_log)
        successful_columns = sum(1 for log in batch_log.values() 
                               if 'error' not in log and log.get('successful_parses', 0) > 0)
        
        report += f"Columns Processed: {successful_columns}/{total_columns}\n\n"
        
        for col, log in batch_log.items():
            report += f"Column: {col}\n"
            
            if 'error' in log:
                report += f"  ❌ Error: {log['error']}\n"
            else:
                success_rate = (log['successful_parses'] / 
                              (log['successful_parses'] + log['failed_parses']) 
                              if (log['successful_parses'] + log['failed_parses']) > 0 else 0)
                
                report += f"  ✅ Success Rate: {success_rate:.1%}\n"
                report += f"  📊 Parsed: {log['successful_parses']}, Failed: {log['failed_parses']}\n"
                report += f"  🔧 Format Used: {log['format_used']}\n"
                
                if log.get('timezone_applied'):
                    report += f"  🌍 Timezone: {log['timezone_applied']}\n"
                
                if log.get('parsing_errors'):
                    report += f"  ⚠️  Errors: {'; '.join(log['parsing_errors'])}\n"
            
            report += "\n"
        
        return report

# Example usage and testing
def demonstrate_datetime_parsing():
    """Demonstrate comprehensive datetime parsing capabilities"""
    
    # Create sample data with various datetime formats
    sample_data = {
        'iso_format': [
            '2024-03-15T14:30:00',
            '2024-03-16T09:15:30',
            '2024-03-17T18:45:00',
            None,
            '2024-03-18T12:00:00'
        ],
        'us_format': [
            '03/15/2024 2:30 PM',
            '03/16/2024 9:15 AM',
            '03/17/2024 6:45 PM',
            '',
            '03/18/2024 12:00 PM'
        ],
        'eu_format': [
            '15/03/2024 14:30',
            '16/03/2024 09:15',
            '17/03/2024 18:45',
            'invalid_date',
            '18/03/2024 12:00'
        ],
        'mixed_format': [
            '2024-03-15 14:30:00',
            'March 16, 2024',
            '17/03/2024',
            '1710505800',  # Unix timestamp
            '2024-03-18T12:00:00Z'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original Data:")
    print(df)
    print()
    
    # Initialize parser
    parser = UniversalDateTimeParser()
    
    # Parse all datetime columns
    datetime_cols = ['iso_format', 'us_format', 'eu_format', 'mixed_format']
    
    # Specify known formats for better performance
    column_formats = {
        'iso_format': '%Y-%m-%dT%H:%M:%S',
        'us_format': '%m/%d/%Y %I:%M %p',
        'eu_format': '%d/%m/%Y %H:%M'
        # mixed_format will use auto-detection
    }
    
    result = parser.batch_parse_datetime_columns(
        df, 
        datetime_cols,
        column_formats=column_formats,
        default_timezone='UTC'
    )
    
    parsed_df = result['parsed_df']
    batch_log = result['batch_log']
    
    print("Parsed Data:")
    print(parsed_df)
    print()
    
    # Generate report
    report = parser.generate_parsing_report(batch_log)
    print(report)
    
    return df, parsed_df, batch_log

# Run demonstration
# original_data, parsed_data, parsing_results = demonstrate_datetime_parsing()
```

---

## 🌍 Timezone Handling

### Comprehensive Timezone Management

```python
import pytz
from zoneinfo import ZoneInfo  # Python 3.9+

class TimezoneManager:
    def __init__(self):
        self.common_timezones = {
            'UTC': 'UTC',
            'EST': 'America/New_York',
            'PST': 'America/Los_Angeles',
            'CST': 'America/Chicago',
            'MST': 'America/Denver',
            'GMT': 'GMT',
            'CET': 'Europe/Paris',
            'JST': 'Asia/Tokyo',
            'AEST': 'Australia/Sydney'
        }
        self.conversion_log = []
    
    def detect_timezone_from_data(self, datetime_series: pd.Series) -> Dict:
        """Attempt to detect timezone from datetime data patterns"""
        
        if len(datetime_series.dropna()) == 0:
            return {'detected_timezone': None, 'confidence': 0, 'method': 'no_data'}
        
        # Check if already timezone-aware
        if hasattr(datetime_series.dtype, 'tz') and datetime_series.dtype.tz is not None:
            return {
                'detected_timezone': str(datetime_series.dtype.tz),
                'confidence': 1.0,
                'method': 'existing_timezone_info'
            }
        
        # Analyze patterns that might indicate timezone
        clean_series = datetime_series.dropna()
        
        # Check for business hours patterns (rough heuristic)
        hours = clean_series.dt.hour
        business_hours_count = ((hours >= 9) & (hours <= 17)).sum()
        total_count = len(hours)
        business_hours_ratio = business_hours_count / total_count if total_count > 0 else 0
        
        # Check for weekend patterns
        weekdays = clean_series.dt.dayofweek
        weekend_count = (weekdays >= 5).sum()
        weekend_ratio = weekend_count / total_count if total_count > 0 else 0
        
        # Heuristic timezone detection (very basic)
        if business_hours_ratio > 0.7:  # Strong business hours pattern
            # Could indicate local business timezone
            confidence = 0.3  # Low confidence without more context
            method = 'business_hours_heuristic'
        else:
            confidence = 0.1
            method = 'insufficient_pattern_data'
        
        return {
            'detected_timezone': 'UTC',  # Safe default
            'confidence': confidence,
            'method': method,
            'business_hours_ratio': business_hours_ratio,
            'weekend_ratio': weekend_ratio
        }
    
    def standardize_to_utc(self, datetime_series: pd.Series, 
                          source_timezone: str = None) -> Dict:
        """Convert datetime series to UTC"""
        
        if len(datetime_series.dropna()) == 0:
            return {
                'converted_series': datetime_series,
                'conversion_success': False,
                'error': 'No valid datetime data'
            }
        
        try:
            # If already timezone-aware
            if hasattr(datetime_series.dtype, 'tz') and datetime_series.dtype.tz is not None:
                if str(datetime_series.dtype.tz) == 'UTC':
                    return {
                        'converted_series': datetime_series,
                        'conversion_success': True,
                        'note': 'Already in UTC'
                    }
                else:
                    # Convert to UTC
                    utc_series = datetime_series.dt.tz_convert('UTC')
                    return {
                        'converted_series': utc_series,
                        'conversion_success': True,
                        'source_timezone': str(datetime_series.dtype.tz),
                        'target_timezone': 'UTC'
                    }
            
            # If timezone-naive, need to localize first
            if source_timezone:
                # Resolve timezone string to proper timezone
                if source_timezone in self.common_timezones:
                    tz = self.common_timezones[source_timezone]
                else:
                    tz = source_timezone
                
                # Localize then convert to UTC
                localized_series = datetime_series.dt.tz_localize(tz, errors='coerce')
                utc_series = localized_series.dt.tz_convert('UTC')
                
                return {
                    'converted_series': utc_series,
                    'conversion_success': True,
                    'source_timezone': source_timezone,
                    'target_timezone': 'UTC',
                    'localization_errors': localized_series.isna().sum() - datetime_series.isna().sum()
                }
            else:
                # Assume UTC and localize
                utc_series = datetime_series.dt.tz_localize('UTC', errors='coerce')
                return {
                    'converted_series': utc_series,
                    'conversion_success': True,
                    'assumption': 'Assumed UTC timezone',
                    'localization_errors': utc_series.isna().sum() - datetime_series.isna().sum()
                }
        
        except Exception as e:
            return {
                'converted_series': datetime_series,
                'conversion_success': False,
                'error': str(e)
            }
    
    def convert_to_local_timezone(self, datetime_series: pd.Series, 
                                target_timezone: str) -> Dict:
        """Convert UTC datetime series to specified local timezone"""
        
        try:
            # Resolve timezone
            if target_timezone in self.common_timezones:
                tz = self.common_timezones[target_timezone]
            else:
                tz = target_timezone
            
            # Ensure series is timezone-aware (assume UTC if not)
            if hasattr(datetime_series.dtype, 'tz') and datetime_series.dtype.tz is not None:
                source_tz = str(datetime_series.dtype.tz)
                local_series = datetime_series.dt.tz_convert(tz)
            else:
                # Assume UTC and convert
                utc_series = datetime_series.dt.tz_localize('UTC', errors='coerce')
                local_series = utc_series.dt.tz_convert(tz)
                source_tz = 'UTC (assumed)'
            
            return {
                'converted_series': local_series,
                'conversion_success': True,
                'source_timezone': source_tz,
                'target_timezone': tz
            }
        
        except Exception as e:
            return {
                'converted_series': datetime_series,
                'conversion_success': False,
                'error': str(e)
            }
    
    def handle_dst_transitions(self, datetime_series: pd.Series) -> Dict:
        """Detect and handle Daylight Saving Time transitions"""
        
        if not hasattr(datetime_series.dtype, 'tz') or datetime_series.dtype.tz is None:
            return {
                'dst_analysis': 'Series is timezone-naive',
                'dst_transitions_detected': 0,
                'gaps_detected': 0,
                'overlaps_detected': 0
            }
        
        try:
            clean_series = datetime_series.dropna().sort_values()
            
            if len(clean_series) < 2:
                return {
                    'dst_analysis': 'Insufficient data for DST analysis',
                    'dst_transitions_detected': 0
                }
            
            # Calculate time differences
            time_diffs = clean_series.diff()
            
            # Look for unusual gaps (DST spring forward) or overlaps (DST fall back)
            # This is a simplified heuristic
            median_diff = time_diffs.median()
            
            # Detect gaps much larger than normal (potential spring forward)
            gap_threshold = median_diff * 2
            gaps = time_diffs[time_diffs > gap_threshold]
            
            # Detect very small or negative differences (potential fall back)
            overlap_threshold = median_diff * 0.5
            overlaps = time_diffs[time_diffs < overlap_threshold]
            
            return {
                'dst_analysis': f'Analyzed {len(clean_series)} records',
                'dst_transitions_detected': len(gaps) + len(overlaps),
                'gaps_detected': len(gaps),
                'overlaps_detected': len(overlaps),
                'largest_gap': gaps.max() if len(gaps) > 0 else None,
                'smallest_diff': overlaps.min() if len(overlaps) > 0 else None,
                'timezone': str(datetime_series.dtype.tz)
            }
        
        except Exception as e:
            return {
                'dst_analysis': 'Error during DST analysis',
                'error': str(e)
            }
    
    def batch_timezone_standardization(self, df: pd.DataFrame, 
                                     datetime_columns: List[str],
                                     source_timezones: Dict[str, str] = None,
                                     target_timezone: str = 'UTC') -> Dict:
        """Standardize multiple datetime columns to target timezone"""
        
        standardized_df = df.copy()
        batch_log = {}
        source_timezones = source_timezones or {}
        
        for col in datetime_columns:
            if col not in df.columns:
                batch_log[col] = {'error': 'Column not found'}
                continue
            
            source_tz = source_timezones.get(col)
            
            if target_timezone == 'UTC':
                result = self.standardize_to_utc(df[col], source_tz)
            else:
                # First convert to UTC, then to target timezone
                utc_result = self.standardize_to_utc(df[col], source_tz)
                if utc_result['conversion_success']:
                    result = self.convert_to_local_timezone(
                        utc_result['converted_series'], target_timezone
                    )
                else:
                    result = utc_result
            
            standardized_df[col] = result['converted_series']
            batch_log[col] = result
            
            # Add DST analysis
            dst_analysis = self.handle_dst_transitions(result['converted_series'])
            batch_log[col]['dst_analysis'] = dst_analysis
        
        return {
            'standardized_df': standardized_df,
            'batch_log': batch_log
        }

# Example usage
def demonstrate_timezone_handling():
    """Demonstrate comprehensive timezone handling"""
    
    # Create sample data with different timezone scenarios
    dates = pd.date_range('2024-03-10', '2024-03-15', freq='6H')  # Includes DST transition
    
    sample_data = {
        'utc_times': dates.tz_localize('UTC'),
        'naive_times': dates.tz_localize(None),  # Timezone-naive
        'est_times': dates.tz_localize('UTC').tz_convert('America/New_York'),
        'pst_times': dates.tz_localize('UTC').tz_convert('America/Los_Angeles'),
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original Data with Mixed Timezones:")
    print(df)
    print()
    
    # Initialize timezone manager
    tz_manager = TimezoneManager()
    
    # Standardize all to UTC
    datetime_cols = ['utc_times', 'naive_times', 'est_times', 'pst_times']
    source_timezones = {
        'naive_times': 'America/New_York'  # Assume EST for naive times
    }
    
    result = tz_manager.batch_timezone_standardization(
        df,
        datetime_cols,
        source_timezones=source_timezones,
        target_timezone='UTC'
    )
    
    standardized_df = result['standardized_df']
    batch_log = result['batch_log']
    
    print("Standardized Data (All UTC):")
    print(standardized_df)
    print()
    
    # Print conversion log
    print("Timezone Conversion Log:")
    for col, log in batch_log.items():
        print(f"\n{col}:")
        for key, value in log.items():
            if key != 'converted_series':
                print(f"  {key}: {value}")
    
    return df, standardized_df, batch_log

# Run demonstration
# original_tz_data, standardized_tz_data, tz_log = demonstrate_timezone_handling()
```

---

## 📅 Date Range Validation

### Date Range Validator

```python
class DateRangeValidator:
    def __init__(self, business_start_year: int = 1990, 
                 max_future_years: int = 5):
        self.business_start = pd.Timestamp(f'{business_start_year}-01-01')
        self.max_future = pd.Timestamp.now() + pd.DateOffset(years=max_future_years)
        self.validation_rules = {}
    
    def validate_date_ranges(self, df: pd.DataFrame, 
                           datetime_columns: List[str],
                           custom_ranges: Dict[str, tuple] = None) -> Dict:
        """Validate datetime columns against business rules"""
        
        validation_results = {}
        custom_ranges = custom_ranges or {}
        
        for col in datetime_columns:
            if col not in df.columns:
                validation_results[col] = {'error': 'Column not found'}
                continue
            
            series = df[col].dropna()
            if len(series) == 0:
                validation_results[col] = {'error': 'No valid dates to validate'}
                continue
            
            # Get range for this column
            if col in custom_ranges:
                min_date, max_date = custom_ranges[col]
                min_date = pd.Timestamp(min_date)
                max_date = pd.Timestamp(max_date)
            else:
                min_date, max_date = self.business_start, self.max_future
            
            # Perform validation
            too_early = (series < min_date).sum()
            too_late = (series > max_date).sum()
            valid_count = len(series) - too_early - too_late
            
            validation_results[col] = {
                'total_dates': len(series),
                'valid_dates': valid_count,
                'too_early': too_early,
                'too_late': too_late,
                'earliest_date': series.min(),
                'latest_date': series.max(),
                'expected_range': (min_date, max_date),
                'validation_passed': too_early == 0 and too_late == 0
            }
        
        return validation_results

# Example validation
def demonstrate_date_validation():
    """Demonstrate date range validation"""
    
    # Create sample data with some invalid dates
    sample_dates = [
        '2024-03-15',  # Valid
        '2024-03-16',  # Valid
        '1850-01-01',  # Too early
        '2050-12-31',  # Too late
        '2024-03-17',  # Valid
    ]
    
    df = pd.DataFrame({
        'transaction_date': pd.to_datetime(sample_dates),
        'amount': [100, 200, 300, 400, 500]
    })
    
    validator = DateRangeValidator(business_start_year=1990, max_future_years=2)
    results = validator.validate_date_ranges(df, ['transaction_date'])
    
    print("Date Validation Results:")
    for col, result in results.items():
        print(f"\n{col}:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    return df, results

# Run validation demo
# validation_df, validation_results = demonstrate_date_validation()
```

---

## ⚡ Basic Temporal Features

### Core Temporal Feature Extractor

```python
class TemporalFeatureExtractor:
    def __init__(self):
        self.feature_definitions = {
            'year': 'Year component',
            'month': 'Month component (1-12)',
            'day': 'Day of month (1-31)',
            'dayofweek': 'Day of week (0=Monday)',
            'dayofyear': 'Day of year (1-366)',
            'quarter': 'Quarter (1-4)',
            'week': 'Week of year (1-53)',
            'hour': 'Hour component (0-23)',
            'minute': 'Minute component (0-59)',
            'second': 'Second component (0-59)',
            'is_weekend': 'Weekend indicator',
            'is_month_start': 'Month start indicator',
            'is_month_end': 'Month end indicator',
            'is_quarter_start': 'Quarter start indicator',
            'is_quarter_end': 'Quarter end indicator',
            'is_year_start': 'Year start indicator',
            'is_year_end': 'Year end indicator'
        }
        self.extraction_log = {}
    
    def extract_basic_features(self, df: pd.DataFrame, 
                             datetime_column: str,
                             features_to_extract: List[str] = None) -> pd.DataFrame:
        """Extract basic temporal features from datetime column"""
        
        if datetime_column not in df.columns:
            raise ValueError(f"Column '{datetime_column}' not found in DataFrame")
        
        # Default to all features if none specified
        if features_to_extract is None:
            features_to_extract = list(self.feature_definitions.keys())
        
        feature_df = df.copy()
        dt_series = pd.to_datetime(feature_df[datetime_column])
        
        # Extract standard datetime components
        if 'year' in features_to_extract:
            feature_df[f'{datetime_column}_year'] = dt_series.dt.year
        
        if 'month' in features_to_extract:
            feature_df[f'{datetime_column}_month'] = dt_series.dt.month
        
        if 'day' in features_to_extract:
            feature_df[f'{datetime_column}_day'] = dt_series.dt.day
        
        if 'dayofweek' in features_to_extract:
            feature_df[f'{datetime_column}_dayofweek'] = dt_series.dt.dayofweek
        
        if 'dayofyear' in features_to_extract:
            feature_df[f'{datetime_column}_dayofyear'] = dt_series.dt.dayofyear
        
        if 'quarter' in features_to_extract:
            feature_df[f'{datetime_column}_quarter'] = dt_series.dt.quarter
        
        if 'week' in features_to_extract:
            feature_df[f'{datetime_column}_week'] = dt_series.dt.isocalendar().week
        
        if 'hour' in features_to_extract:
            feature_df[f'{datetime_column}_hour'] = dt_series.dt.hour
        
        if 'minute' in features_to_extract:
            feature_df[f'{datetime_column}_minute'] = dt_series.dt.minute
        
        if 'second' in features_to_extract:
            feature_df[f'{datetime_column}_second'] = dt_series.dt.second
        
        # Extract boolean indicators
        if 'is_weekend' in features_to_extract:
            feature_df[f'{datetime_column}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
        
        if 'is_month_start' in features_to_extract:
            feature_df[f'{datetime_column}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
        
        if 'is_month_end' in features_to_extract:
            feature_df[f'{datetime_column}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
        
        if 'is_quarter_start' in features_to_extract:
            feature_df[f'{datetime_column}_is_quarter_start'] = dt_series.dt.is_quarter_start.astype(int)
        
        if 'is_quarter_end' in features_to_extract:
            feature_df[f'{datetime_column}_is_quarter_end'] = dt_series.dt.is_quarter_end.astype(int)
        
        if 'is_year_start' in features_to_extract:
            feature_df[f'{datetime_column}_is_year_start'] = dt_series.dt.is_year_start.astype(int)
        
        if 'is_year_end' in features_to_extract:
            feature_df[f'{datetime_column}_is_year_end'] = dt_series.dt.is_year_end.astype(int)
        
        # Log extraction results
        new_features = [col for col in feature_df.columns if col not in df.columns]
        self.extraction_log[datetime_column] = {
            'features_extracted': new_features,
            'feature_count': len(new_features),
            'extraction_timestamp': pd.Timestamp.now()
        }
        
        return feature_df
    
    def extract_time_parts(self, df: pd.DataFrame, 
                          datetime_column: str,
                          time_granularity: str = 'hour') -> pd.DataFrame:
        """Extract time-of-day features with specified granularity"""
        
        feature_df = df.copy()
        dt_series = pd.to_datetime(feature_df[datetime_column])
        
        if time_granularity == 'hour':
            feature_df[f'{datetime_column}_hour_of_day'] = dt_series.dt.hour
            
            # Time periods
            feature_df[f'{datetime_column}_is_morning'] = ((dt_series.dt.hour >= 6) & 
                                                         (dt_series.dt.hour < 12)).astype(int)
            feature_df[f'{datetime_column}_is_afternoon'] = ((dt_series.dt.hour >= 12) & 
                                                           (dt_series.dt.hour < 18)).astype(int)
            feature_df[f'{datetime_column}_is_evening'] = ((dt_series.dt.hour >= 18) & 
                                                         (dt_series.dt.hour < 22)).astype(int)
            feature_df[f'{datetime_column}_is_night'] = ((dt_series.dt.hour >= 22) | 
                                                       (dt_series.dt.hour < 6)).astype(int)
            
        elif time_granularity == 'minute':
            feature_df[f'{datetime_column}_hour'] = dt_series.dt.hour
            feature_df[f'{datetime_column}_minute'] = dt_series.dt.minute
            feature_df[f'{datetime_column}_minute_of_day'] = (dt_series.dt.hour * 60 + 
                                                            dt_series.dt.minute)
            
        elif time_granularity == 'second':
            feature_df[f'{datetime_column}_hour'] = dt_series.dt.hour
            feature_df[f'{datetime_column}_minute'] = dt_series.dt.minute
            feature_df[f'{datetime_column}_second'] = dt_series.dt.second
            feature_df[f'{datetime_column}_second_of_day'] = (dt_series.dt.hour * 3600 + 
                                                            dt_series.dt.minute * 60 + 
                                                            dt_series.dt.second)
        
        return feature_df

# Example usage
def demonstrate_basic_feature_extraction():
    """Demonstrate basic temporal feature extraction"""
    
    # Create sample datetime data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'transaction_date': dates,
        'amount': np.random.uniform(10, 1000, len(dates))
    })
    
    # Add some timestamps with time components
    sample_data['timestamp'] = pd.date_range('2024-01-01 08:30:00', 
                                           periods=len(dates), 
                                           freq='D')
    
    print("Original Data Sample:")
    print(sample_data.head())
    print()
    
    # Extract features
    extractor = TemporalFeatureExtractor()
    
    # Extract basic date features
    featured_df = extractor.extract_basic_features(
        sample_data, 
        'transaction_date',
        features_to_extract=['year', 'month', 'quarter', 'dayofweek', 
                           'is_weekend', 'is_month_start', 'is_quarter_end']
    )
    
    # Extract time features
    featured_df = extractor.extract_time_parts(
        featured_df, 
        'timestamp',
        time_granularity='hour'
    )
    
    print("Data with Extracted Features:")
    print(featured_df[['transaction_date', 'transaction_date_year', 
                      'transaction_date_month', 'transaction_date_quarter',
                      'transaction_date_dayofweek', 'transaction_date_is_weekend',
                      'timestamp_hour_of_day', 'timestamp_is_morning']].head(10))
    
    return sample_data, featured_df

# Run demonstration
# original_basic_data, featured_basic_data = demonstrate_basic_feature_extraction()
```

---

## 📊 Cyclical Feature Engineering

### Cyclical Encoder for Temporal Patterns

```python
import math

class CyclicalFeatureEncoder:
    def __init__(self):
        self.encoding_parameters = {}
        self.encoding_log = {}
    
    def encode_cyclical_feature(self, values: pd.Series, 
                              max_value: int,
                              feature_name: str = None) -> pd.DataFrame:
        """Encode cyclical features using sine and cosine transformations"""
        
        if feature_name is None:
            feature_name = values.name or 'feature'
        
        # Handle missing values
        clean_values = values.fillna(0)  # or handle differently based on context
        
        # Normalize to 0-1 range
        normalized_values = clean_values / max_value
        
        # Apply sine and cosine transformations
        sin_values = np.sin(2 * np.pi * normalized_values)
        cos_values = np.cos(2 * np.pi * normalized_values)
        
        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame({
            f'{feature_name}_sin': sin_values,
            f'{feature_name}_cos': cos_values
        }, index=values.index)
        
        # Store encoding parameters for future use
        self.encoding_parameters[feature_name] = {
            'max_value': max_value,
            'encoding_type': 'sine_cosine'
        }
        
        return encoded_df
    
    def encode_temporal_cycles(self, df: pd.DataFrame, 
                             datetime_column: str) -> pd.DataFrame:
        """Encode multiple temporal cycles from datetime column"""
        
        dt_series = pd.to_datetime(df[datetime_column])
        encoded_df = df.copy()
        
        # Month cycle (1-12)
        month_encoded = self.encode_cyclical_feature(
            dt_series.dt.month, 12, f'{datetime_column}_month'
        )
        encoded_df = pd.concat([encoded_df, month_encoded], axis=1)
        
        # Day of week cycle (0-6)
        dow_encoded = self.encode_cyclical_feature(
            dt_series.dt.dayofweek, 7, f'{datetime_column}_dayofweek'
        )
        encoded_df = pd.concat([encoded_df, dow_encoded], axis=1)
        
        # Hour cycle (0-23) - if time information is available
        if dt_series.dt.hour.nunique() > 1:  # Check if hour info exists
            hour_encoded = self.encode_cyclical_feature(
                dt_series.dt.hour, 24, f'{datetime_column}_hour'
            )
            encoded_df = pd.concat([encoded_df, hour_encoded], axis=1)
        
        # Day of year cycle (1-366)
        doy_encoded = self.encode_cyclical_feature(
            dt_series.dt.dayofyear, 366, f'{datetime_column}_dayofyear'
        )
        encoded_df = pd.concat([encoded_df, doy_encoded], axis=1)
        
        # Quarter cycle (1-4)
        quarter_encoded = self.encode_cyclical_feature(
            dt_series.dt.quarter, 4, f'{datetime_column}_quarter'
        )
        encoded_df = pd.concat([encoded_df, quarter_encoded], axis=1)
        
        # Log the encoding
        new_features = [col for col in encoded_df.columns if col not in df.columns]
        self.encoding_log[datetime_column] = {
            'cyclical_features_created': new_features,
            'feature_count': len(new_features),
            'encoding_timestamp': pd.Timestamp.now()
        }
        
        return encoded_df
    
    def create_custom_cyclical_features(self, df: pd.DataFrame,
                                      datetime_column: str,
                                      custom_cycles: Dict[str, int]) -> pd.DataFrame:
        """Create custom cyclical features based on user-defined cycles"""
        
        dt_series = pd.to_datetime(df[datetime_column])
        encoded_df = df.copy()
        
        for cycle_name, cycle_length in custom_cycles.items():
            if cycle_name == 'week_of_month':
                # Week of month (1-5)
                week_of_month = ((dt_series.dt.day - 1) // 7) + 1
                week_encoded = self.encode_cyclical_feature(
                    week_of_month, 5, f'{datetime_column}_week_of_month'
                )
                encoded_df = pd.concat([encoded_df, week_encoded], axis=1)
            
            elif cycle_name == 'business_quarter':
                # Custom business quarter (if different from calendar quarter)
                # This is just an example - adjust based on business needs
                business_quarter = ((dt_series.dt.month - 1) // 3) + 1
                bq_encoded = self.encode_cyclical_feature(
                    business_quarter, 4, f'{datetime_column}_business_quarter'
                )
                encoded_df = pd.concat([encoded_df, bq_encoded], axis=1)
            
            elif cycle_name == 'minute_of_hour':
                # Minute of hour cycle
                minute_encoded = self.encode_cyclical_feature(
                    dt_series.dt.minute, 60, f'{datetime_column}_minute'
                )
                encoded_df = pd.concat([encoded_df, minute_encoded], axis=1)
        
        return encoded_df
    
    def validate_cyclical_encoding(self, original_values: pd.Series,
                                 sin_values: pd.Series,
                                 cos_values: pd.Series,
                                 max_value: int) -> Dict:
        """Validate that cyclical encoding preserves the cyclical nature"""
        
        # Reconstruct approximate original values
        angles = np.arctan2(sin_values, cos_values)
        # Convert from [-π, π] to [0, 2π]
        angles = np.where(angles < 0, angles + 2*np.pi, angles)
        # Convert back to original scale
        reconstructed = (angles / (2 * np.pi)) * max_value
        
        # Calculate reconstruction error
        original_clean = original_values.fillna(0)
        reconstruction_error = np.mean(np.abs(original_clean - reconstructed))
        
        # Check cyclical properties
        # Values at boundaries should be close
        boundary_check = {}
        if max_value in [12, 24, 7]:  # Common cyclical features
            if max_value == 12:  # Month
                jan_sin, jan_cos = sin_values[original_values == 1].mean(), cos_values[original_values == 1].mean()
                dec_sin, dec_cos = sin_values[original_values == 12].mean(), cos_values[original_values == 12].mean()
                boundary_distance = np.sqrt((jan_sin - dec_sin)**2 + (jan_cos - dec_cos)**2)
                boundary_check['month_boundary_distance'] = boundary_distance
        
        return {
            'reconstruction_error': reconstruction_error,
            'max_reconstruction_error': np.max(np.abs(original_clean - reconstructed)),
            'boundary_checks': boundary_check,
            'encoding_preserves_cyclical_nature': reconstruction_error < 0.1
        }

# Example usage and visualization
def demonstrate_cyclical_encoding():
    """Demonstrate cyclical feature encoding with visualization"""
    
    # Create sample data with strong cyclical patterns
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    
    # Create synthetic data with cyclical patterns
    np.random.seed(42)
    monthly_pattern = np.sin(2 * np.pi * pd.to_datetime(dates).month / 12)
    weekly_pattern = np.sin(2 * np.pi * pd.to_datetime(dates).dayofweek / 7)
    daily_trend = np.linspace(0, 1, len(dates))
    noise = np.random.normal(0, 0.1, len(dates))
    
    sample_data = pd.DataFrame({
        'date': dates,
        'sales': 100 + 50 * monthly_pattern + 20 * weekly_pattern + 30 * daily_trend + noise
    })
    
    print("Original Data Sample:")
    print(sample_data.head())
    print()
    
    # Apply cyclical encoding
    encoder = CyclicalFeatureEncoder()
    
    # Encode standard temporal cycles
    encoded_df = encoder.encode_temporal_cycles(sample_data, 'date')
    
    # Add custom cycles
    custom_cycles = {
        'week_of_month': 5,
        'minute_of_hour': 60  # This won't be used since we have date-only data
    }
    encoded_df = encoder.create_custom_cyclical_features(
        encoded_df, 'date', custom_cycles
    )
    
    print("Sample of Cyclically Encoded Features:")
    cyclical_cols = [col for col in encoded_df.columns if '_sin' in col or '_cos' in col]
    print(encoded_df[['date'] + cyclical_cols[:6]].head())
    print()
    
    # Validate encoding
    month_validation = encoder.validate_cyclical_encoding(
        pd.to_datetime(sample_data['date']).dt.month,
        encoded_df['date_month_sin'],
        encoded_df['date_month_cos'],
        12
    )
    
    print("Cyclical Encoding Validation (Month):")
    for key, value in month_validation.items():
        print(f"  {key}: {value}")
    
    return sample_data, encoded_df, encoder

# Run demonstration
# original_cyclical_data, encoded_cyclical_data, cyclical_encoder = demonstrate_cyclical_encoding()
```

---

## 📈 Time-Based Aggregations

### Rolling and Temporal Aggregation Engine

```python
from typing import Callable, Any

class TemporalAggregationEngine:
    def __init__(self):
        self.aggregation_functions = {
            'mean': np.mean,
            'median': np.median,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'sum': np.sum,
            'count': len,
            'q25': lambda x: np.percentile(x, 25),
            'q75': lambda x: np.percentile(x, 75),
            'skew': lambda x: stats.skew(x) if len(x) > 2 else 0,
            'kurtosis': lambda x: stats.kurtosis(x) if len(x) > 2 else 0
        }
        self.aggregation_log = {}
    
    def create_rolling_features(self, df: pd.DataFrame,
                               datetime_column: str,
                               target_columns: List[str],
                               window_sizes: List[str],
                               aggregation_funcs: List[str] = None) -> pd.DataFrame:
        """Create rolling window features for specified columns"""
        
        if aggregation_funcs is None:
            aggregation_funcs = ['mean', 'std', 'min', 'max']
        
        # Ensure dataframe is sorted by datetime
        df_sorted = df.sort_values(datetime_column).copy()
        df_sorted = df_sorted.set_index(datetime_column)
        
        feature_df = df_sorted.copy()
        created_features = []
        
        for target_col in target_columns:
            if target_col not in df_sorted.columns:
                continue
            
            for window_size in window_sizes:
                for agg_func in aggregation_funcs:
                    if agg_func not in self.aggregation_functions:
                        continue
                    
                    feature_name = f'{target_col}_rolling_{window_size}_{agg_func}'
                    
                    try:
                        # Create rolling window
                        rolling_window = df_sorted[target_col].rolling(window=window_size, min_periods=1)
                        
                        # Apply aggregation function
                        if agg_func in ['mean', 'std', 'min', 'max', 'sum', 'count']:
                            # Use pandas built-in methods
                            feature_values = getattr(rolling_window, agg_func)()
                        else:
                            # Use custom functions
                            feature_values = rolling_window.apply(self.aggregation_functions[agg_func])
                        
                        feature_df[feature_name] = feature_values
                        created_features.append(feature_name)
                    
                    except Exception as e:
                        print(f"Error creating feature {feature_name}: {str(e)}")
        
        # Reset index to return to original structure
        feature_df = feature_df.reset_index()
        
        # Log creation
        self.aggregation_log['rolling_features'] = {
            'features_created': created_features,
            'feature_count': len(created_features),
            'target_columns': target_columns,
            'window_sizes': window_sizes,
            'aggregation_functions': aggregation_funcs,
            'creation_timestamp': pd.Timestamp.now()
        }
        
        return feature_df
    
    def create_expanding_features(self, df: pd.DataFrame,
                                datetime_column: str,
                                target_columns: List[str],
                                aggregation_funcs: List[str] = None) -> pd.DataFrame:
        """Create expanding window features (cumulative statistics)"""
        
        if aggregation_funcs is None:
            aggregation_funcs = ['mean', 'std', 'min', 'max', 'sum', 'count']
        
        df_sorted = df.sort_values(datetime_column).copy()
        df_sorted = df_sorted.set_index(datetime_column)
        
        feature_df = df_sorted.copy()
        created_features = []
        
        for target_col in target_columns:
            if target_col not in df_sorted.columns:
                continue
            
            for agg_func in aggregation_funcs:
                if agg_func not in self.aggregation_functions:
                    continue
                
                feature_name = f'{target_col}_expanding_{agg_func}'
                
                try:
                    expanding_window = df_sorted[target_col].expanding(min_periods=1)
                    
                    if agg_func in ['mean', 'std', 'min', 'max', 'sum', 'count']:
                        feature_values = getattr(expanding_window, agg_func)()
                    else:
                        feature_values = expanding_window.apply(self.aggregation_functions[agg_func])
                    
                    feature_df[feature_name] = feature_values
                    created_features.append(feature_name)
                
                except Exception as e:
                    print(f"Error creating expanding feature {feature_name}: {str(e)}")
        
        feature_df = feature_df.reset_index()
        
        # Log creation
        self.aggregation_log['expanding_features'] = {
            'features_created': created_features,
            'feature_count': len(created_features),
            'target_columns': target_columns,
            'aggregation_functions': aggregation_funcs,
            'creation_timestamp': pd.Timestamp.now()
        }
        
        return feature_df
    
    def create_grouped_time_features(self, df: pd.DataFrame,
                                   datetime_column: str,
                                   target_columns: List[str],
                                   groupby_columns: List[str],
                                   time_periods: List[str] = None,
                                   aggregation_funcs: List[str] = None) -> pd.DataFrame:
        """Create time-based aggregations grouped by categorical variables"""
        
        if time_periods is None:
            time_periods = ['D', 'W', 'M']  # Daily, Weekly, Monthly
        
        if aggregation_funcs is None:
            aggregation_funcs = ['mean', 'sum', 'count', 'std']
        
        df_sorted = df.sort_values(datetime_column).copy()
        feature_df = df_sorted.copy()
        created_features = []
        
        for period in time_periods:
            for target_col in target_columns:
                if target_col not in df_sorted.columns:
                    continue
                
                for group_col in groupby_columns:
                    if group_col not in df_sorted.columns:
                        continue
                    
                    for agg_func in aggregation_funcs:
                        feature_name = f'{target_col}_{group_col}_{period}_{agg_func}'
                        
                        try:
                            # Create time-based grouping
                            df_grouped = df_sorted.set_index(datetime_column)
                            
                            # Group by time period and categorical variable
                            if agg_func == 'count':
                                grouped_agg = (df_grouped.groupby([group_col, pd.Grouper(freq=period)])[target_col]
                                             .count().reset_index())
                            elif agg_func in ['mean', 'sum', 'std', 'min', 'max']:
                                grouped_agg = (df_grouped.groupby([group_col, pd.Grouper(freq=period)])[target_col]
                                             .agg(agg_func).reset_index())
                            else:
                                grouped_agg = (df_grouped.groupby([group_col, pd.Grouper(freq=period)])[target_col]
                                             .apply(self.aggregation_functions[agg_func]).reset_index())
                            
                            # Merge back to original dataframe
                            # This is a simplified approach - in practice, you'd want more sophisticated merging
                            grouped_agg.columns = [group_col, datetime_column, feature_name]
                            
                            feature_df = feature_df.merge(grouped_agg, 
                                                        on=[group_col, datetime_column], 
                                                        how='left')
                            created_features.append(feature_name)
                        
                        except Exception as e:
                            print(f"Error creating grouped feature {feature_name}: {str(e)}")
        
        return feature_df
    
    def create_seasonal_aggregations(self, df: pd.DataFrame,
                                   datetime_column: str,
                                   target_columns: List[str],
                                   seasonal_periods: Dict[str, str] = None) -> pd.DataFrame:
        """Create seasonal aggregation features"""
        
        if seasonal_periods is None:
            seasonal_periods = {
                'month': 'M',
                'quarter': 'Q',
                'day_of_week': 'D',
                'hour': 'H'
            }
        
        df_sorted = df.sort_values(datetime_column).copy()
        feature_df = df_sorted.copy()
        dt_series = pd.to_datetime(df_sorted[datetime_column])
        
        for target_col in target_columns:
            if target_col not in df_sorted.columns:
                continue
            
            # Monthly averages
            if 'month' in seasonal_periods:
                monthly_avg = df_sorted.groupby(dt_series.dt.month)[target_col].mean()
                feature_df[f'{target_col}_month_avg'] = dt_series.dt.month.map(monthly_avg)
            
            # Day of week averages
            if 'day_of_week' in seasonal_periods:
                dow_avg = df_sorted.groupby(dt_series.dt.dayofweek)[target_col].mean()
                feature_df[f'{target_col}_dayofweek_avg'] = dt_series.dt.dayofweek.map(dow_avg)
            
            # Hourly averages (if time info available)
            if 'hour' in seasonal_periods and dt_series.dt.hour.nunique() > 1:
                hourly_avg = df_sorted.groupby(dt_series.dt.hour)[target_col].mean()
                feature_df[f'{target_col}_hour_avg'] = dt_series.dt.hour.map(hourly_avg)
            
            # Quarterly averages
            if 'quarter' in seasonal_periods:
                quarterly_avg = df_sorted.groupby(dt_series.dt.quarter)[target_col].mean()
                feature_df[f'{target_col}_quarter_avg'] = dt_series.dt.quarter.map(quarterly_avg)
        
        return feature_df

# Example usage
def demonstrate_temporal_aggregations():
    """Demonstrate various temporal aggregation techniques"""
    
    # Create sample time series data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    
    # Create sample data with trends and seasonality
    trend = np.linspace(100, 200, len(dates))
    seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)  # Monthly pattern
    noise = np.random.normal(0, 10, len(dates))
    
    sample_data = pd.DataFrame({
        'date': dates,
        'sales': trend + seasonal + noise,
        'category': np.random.choice(['A', 'B', 'C'], len(dates)),
        'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates))
    })
    
    print("Original Data Sample:")
    print(sample_data.head())
    print()
    
    # Initialize aggregation engine
    agg_engine = TemporalAggregationEngine()
    
    # Create rolling features
    rolling_df = agg_engine.create_rolling_features(
        sample_data,
        datetime_column='date',
        target_columns=['sales'],
        window_sizes=['7D', '30D'],
        aggregation_funcs=['mean', 'std', 'min', 'max']
    )
    
    print("Sample Rolling Features:")
    rolling_cols = [col for col in rolling_df.columns if 'rolling' in col]
    print(rolling_df[['date', 'sales'] + rolling_cols[:4]].head())
    print()
    
    # Create expanding features
    expanding_df = agg_engine.create_expanding_features(
        rolling_df,
        datetime_column='date',
        target_columns=['sales'],
        aggregation_funcs=['mean', 'std', 'count']
    )
    
    print("Sample Expanding Features:")
    expanding_cols = [col for col in expanding_df.columns if 'expanding' in col]
    print(expanding_df[['date', 'sales'] + expanding_cols].tail())
    print()
    
    # Create seasonal aggregations
    seasonal_df = agg_engine.create_seasonal_aggregations(
        expanding_df,
        datetime_column='date',
        target_columns=['sales']
    )
    
    print("Sample Seasonal Features:")
    seasonal_cols = [col for col in seasonal_df.columns if any(x in col for x in ['month_avg', 'dayofweek_avg'])]
    print(seasonal_df[['date', 'sales'] + seasonal_cols].head())
    
    return sample_data, seasonal_df, agg_engine

# Run demonstration
# original_agg_data, aggregated_data, aggregation_engine = demonstrate_temporal_aggregations()
```

---

## 🔄 Lag and Window Features

### Advanced Lag Feature Generator

```python
class LagFeatureGenerator:
    def __init__(self):
        self.lag_parameters = {}
        self.generation_log = {}
    
    def create_lag_features(self, df: pd.DataFrame,
                          datetime_column: str,
                          target_columns: List[str],
                          lag_periods: List[int],
                          group_columns: List[str] = None) -> pd.DataFrame:
        """Create lag features for specified columns"""
        
        df_sorted = df.sort_values(datetime_column).copy()
        feature_df = df_sorted.copy()
        created_features = []
        
        for target_col in target_columns:
            if target_col not in df_sorted.columns:
                continue
            
            if group_columns:
                # Create lags within groups
                for lag in lag_periods:
                    feature_name = f'{target_col}_lag_{lag}'
                    feature_df[feature_name] = (df_sorted.groupby(group_columns)[target_col]
                                              .shift(lag))
                    created_features.append(feature_name)
            else:
                # Create global lags
                for lag in lag_periods:
                    feature_name = f'{target_col}_lag_{lag}'
                    feature_df[feature_name] = df_sorted[target_col].shift(lag)
                    created_features.append(feature_name)
        
        # Log creation
        self.generation_log['lag_features'] = {
            'features_created': created_features,
            'feature_count': len(created_features),
            'target_columns': target_columns,
            'lag_periods': lag_periods,
            'group_columns': group_columns,
            'creation_timestamp': pd.Timestamp.now()
        }
        
        return feature_df
    
    def create_lead_features(self, df: pd.DataFrame,
                           datetime_column: str,
                           target_columns: List[str],
                           lead_periods: List[int],
                           group_columns: List[str] = None) -> pd.DataFrame:
        """Create lead (future) features for specified columns"""
        
        df_sorted = df.sort_values(datetime_column).copy()
        feature_df = df_sorted.copy()
        created_features = []
        
        for target_col in target_columns:
            if target_col not in df_sorted.columns:
                continue
            
            if group_columns:
                # Create leads within groups
                for lead in lead_periods:
                    feature_name = f'{target_col}_lead_{lead}'
                    feature_df[feature_name] = (df_sorted.groupby(group_columns)[target_col]
                                              .shift(-lead))
                    created_features.append(feature_name)
            else:
                # Create global leads
                for lead in lead_periods:
                    feature_name = f'{target_col}_lead_{lead}'
                    feature_df[feature_name] = df_sorted[target_col].shift(-lead)
                    created_features.append(feature_name)
        
        return feature_df
    
    def create_diff_features(self, df: pd.DataFrame,
                           datetime_column: str,
                           target_columns: List[str],
                           diff_periods: List[int] = None,
                           group_columns: List[str] = None) -> pd.DataFrame:
        """Create difference features (current - previous values)"""
        
        if diff_periods is None:
            diff_periods = [1, 7, 30]  # 1-day, 1-week, 1-month differences
        
        df_sorted = df.sort_values(datetime_column).copy()
        feature_df = df_sorted.copy()
        created_features = []
        
        for target_col in target_columns:
            if target_col not in df_sorted.columns:
                continue
            
            if group_columns:
                # Create differences within groups
                for diff_period in diff_periods:
                    feature_name = f'{target_col}_diff_{diff_period}'
                    lagged_values = (df_sorted.groupby(group_columns)[target_col]
                                   .shift(diff_period))
                    feature_df[feature_name] = df_sorted[target_col] - lagged_values
                    created_features.append(feature_name)
            else:
                # Create global differences
                for diff_period in diff_periods:
                    feature_name = f'{target_col}_diff_{diff_period}'
                    feature_df[feature_name] = (df_sorted[target_col] - 
                                              df_sorted[target_col].shift(diff_period))
                    created_features.append(feature_name)
        
        return feature_df
    
    def create_pct_change_features(self, df: pd.DataFrame,
                                 datetime_column: str,
                                 target_columns: List[str],
                                 pct_periods: List[int] = None,
                                 group_columns: List[str] = None) -> pd.DataFrame:
        """Create percentage change features"""
        
        if pct_periods is None:
            pct_periods = [1, 7, 30]
        
        df_sorted = df.sort_values(datetime_column).copy()
        feature_df = df_sorted.copy()
        created_features = []
        
        for target_col in target_columns:
            if target_col not in df_sorted.columns:
                continue
            
            if group_columns:
                # Create percentage changes within groups
                for pct_period in pct_periods:
                    feature_name = f'{target_col}_pct_change_{pct_period}'
                    feature_df[feature_name] = (df_sorted.groupby(group_columns)[target_col]
                                              .pct_change(periods=pct_period))
                    created_features.append(feature_name)
            else:
                # Create global percentage changes
                for pct_period in pct_periods:
                    feature_name = f'{target_col}_pct_change_{pct_period}'
                    feature_df[feature_name] = df_sorted[target_col].pct_change(periods=pct_period)
                    created_features.append(feature_name)
        
        return feature_df
    
    def create_moving_statistics(self, df: pd.DataFrame,
                               datetime_column: str,
                               target_columns: List[str],
                               window_sizes: List[int],
                               statistics: List[str] = None) -> pd.DataFrame:
        """Create moving statistical features beyond simple rolling means"""
        
        if statistics is None:
            statistics = ['momentum', 'volatility', 'trend_strength']
        
        df_sorted = df.sort_values(datetime_column).copy()
        feature_df = df_sorted.copy()
        created_features = []
        
        for target_col in target_columns:
            if target_col not in df_sorted.columns:
                continue
            
            for window_size in window_sizes:
                # Momentum (rate of change over window)
                if 'momentum' in statistics:
                    feature_name = f'{target_col}_momentum_{window_size}'
                    feature_df[feature_name] = (df_sorted[target_col] - 
                                              df_sorted[target_col].shift(window_size))
                    created_features.append(feature_name)
                
                # Volatility (rolling standard deviation)
                if 'volatility' in statistics:
                    feature_name = f'{target_col}_volatility_{window_size}'
                    feature_df[feature_name] = (df_sorted[target_col]
                                              .rolling(window=window_size, min_periods=1)
                                              .std())
                    created_features.append(feature_name)
                
                # Trend strength (correlation with time index)
                if 'trend_strength' in statistics:
                    feature_name = f'{target_col}_trend_{window_size}'
                    
                    def calculate_trend_strength(series):
                        if len(series) < 3:
                            return 0
                        x = np.arange(len(series))
                        try:
                            correlation = np.corrcoef(x, series)[0, 1]
                            return correlation if not np.isnan(correlation) else 0
                        except:
                            return 0
                    
                    feature_df[feature_name] = (df_sorted[target_col]
                                              .rolling(window=window_size, min_periods=3)
                                              .apply(calculate_trend_strength))
                    created_features.append(feature_name)
        
        return feature_df

# Example usage
def demonstrate_lag_features():
    """Demonstrate lag and window feature creation"""
    
    # Create sample time series data with multiple groups
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    groups = ['A', 'B', 'C']
    
    sample_data = []
    for group in groups:
        for date in dates:
            # Create different patterns for each group
            if group == 'A':
                base_value = 100 + np.sin(2 * np.pi * (date.dayofyear / 365)) * 20
            elif group == 'B':
                base_value = 150 + np.cos(2 * np.pi * (date.dayofyear / 365)) * 30
            else:
                base_value = 80 + np.random.normal(0, 10)
            
            noise = np.random.normal(0, 5)
            sample_data.append({
                'date': date,
                'group': group,
                'value': base_value + noise
            })
    
    df = pd.DataFrame(sample_data)
    
    print("Original Data Sample:")
    print(df.head(10))
    print()
    
    # Initialize lag feature generator
    lag_generator = LagFeatureGenerator()
    
    # Create lag features
    lag_df = lag_generator.create_lag_features(
        df,
        datetime_column='date',
        target_columns=['value'],
        lag_periods=[1, 7, 14],
        group_columns=['group']
    )
    
    print("Sample Lag Features:")
    lag_cols = [col for col in lag_df.columns if 'lag' in col]
    print(lag_df[['date', 'group', 'value'] + lag_cols].head(10))
    print()
    
    # Create difference features
    diff_df = lag_generator.create_diff_features(
        lag_df,
        datetime_column='date',
        target_columns=['value'],
        diff_periods=[1, 7],
        group_columns=['group']
    )
    
    print("Sample Difference Features:")
    diff_cols = [col for col in diff_df.columns if 'diff' in col]
    print(diff_df[['date', 'group', 'value'] + diff_cols].head(10))
    print()
    
    # Create moving statistics
    stats_df = lag_generator.create_moving_statistics(
        diff_df,
        datetime_column='date',
        target_columns=['value'],
        window_sizes=[7, 14],
        statistics=['momentum', 'volatility', 'trend_strength']
    )
    
    print("Sample Moving Statistics:")
    stats_cols = [col for col in stats_df.columns if any(x in col for x in ['momentum', 'volatility', 'trend'])]
    print(stats_df[['date', 'group', 'value'] + stats_cols[:3]].head(10))
    
    return df, stats_df, lag_generator

# Run demonstration
# original_lag_data, enhanced_lag_data, lag_feature_generator = demonstrate_lag_features()
```

---

## 📋 Business Calendar Features

### Business Calendar and Holiday Feature Generator

```python
import holidays
from datetime import date

class BusinessCalendarFeatureGenerator:
    def __init__(self, country: str = 'US', custom_holidays: Dict = None):
        """Initialize with country-specific holidays and custom business calendar"""
        
        self.country = country
        self.holidays_lib = holidays.country_holidays(country)
        self.custom_holidays = custom_holidays or {}
        self.business_calendar_features = {}
        
        # Define standard business calendar features
        self.standard_features = {
            'is_holiday': 'National/regional holiday indicator',
            'is_business_day': 'Business day indicator (Mon-Fri, non-holiday)',
            'days_to_holiday': 'Days until next holiday',
            'days_from_holiday': 'Days since last holiday',
            'is_month_end_business': 'Last business day of month',
            'is_quarter_end_business': 'Last business day of quarter',
            'business_days_in_month': 'Number of business days in month',
            'week_of_month': 'Week number within month',
            'is_payday': 'Typical payday indicator'
        }
    
    def add_custom_business_rules(self, rules: Dict):
        """Add custom business rules and events"""
        self.custom_holidays.update(rules)
    
    def extract_holiday_features(self, df: pd.DataFrame, 
                                datetime_column: str) -> pd.DataFrame:
        """Extract holiday-related features"""
        
        feature_df = df.copy()
        dt_series = pd.to_datetime(feature_df[datetime_column])
        
        # Basic holiday indicator
        feature_df[f'{datetime_column}_is_holiday'] = dt_series.dt.date.isin(self.holidays_lib.keys()).astype(int)
        
        # Holiday names (for specific holiday analysis)
        feature_df[f'{datetime_column}_holiday_name'] = dt_series.dt.date.map(
            lambda x: self.holidays_lib.get(x, 'None')
        )
        
        # Days to next holiday
        def days_to_next_holiday(date_val):
            current_date = date_val.date()
            future_holidays = [h for h in self.holidays_lib.keys() if h > current_date]
            if future_holidays:
                return (min(future_holidays) - current_date).days
            return 365  # Default if no holidays found in next year
        
        feature_df[f'{datetime_column}_days_to_holiday'] = dt_series.apply(days_to_next_holiday)
        
        # Days from last holiday
        def days_from_last_holiday(date_val):
            current_date = date_val.date()
            past_holidays = [h for h in self.holidays_lib.keys() if h < current_date]
            if past_holidays:
                return (current_date - max(past_holidays)).days
            return 365  # Default if no past holidays found
        
        feature_df[f'{datetime_column}_days_from_holiday'] = dt_series.apply(days_from_last_holiday)
        
        return feature_df
    
    def extract_business_day_features(self, df: pd.DataFrame, 
                                    datetime_column: str) -> pd.DataFrame:
        """Extract business day related features"""
        
        feature_df = df.copy()
        dt_series = pd.to_datetime(feature_df[datetime_column])
        
        # Basic business day indicator (Monday-Friday, non-holiday)
        is_weekday = dt_series.dt.dayofweek < 5
        is_not_holiday = ~dt_series.dt.date.isin(self.holidays_lib.keys())
        feature_df[f'{datetime_column}_is_business_day'] = (is_weekday & is_not_holiday).astype(int)
        
        # Business days in month
        def business_days_in_month(date_val):
            start_of_month = date_val.replace(day=1)
            next_month = (start_of_month + pd.DateOffset(months=1))
            month_range = pd.date_range(start_of_month, next_month - pd.Timedelta(days=1), freq='B')
            # Remove holidays
            business_days = [d for d in month_range if d.date() not in self.holidays_lib]
            return len(business_days)
        
        feature_df[f'{datetime_column}_business_days_in_month'] = dt_series.apply(business_days_in_month)
        
        # Last business day of month/quarter
        def is_last_business_day_of_month(date_val):
            next_day = date_val + pd.Timedelta(days=1)
            if next_day.month != date_val.month:  # Next day is in next month
                return 1 if feature_df.loc[feature_df[datetime_column] == date_val, f'{datetime_column}_is_business_day'].iloc[0] else 0
            return 0
        
        feature_df[f'{datetime_column}_is_month_end_business'] = dt_series.apply(is_last_business_day_of_month)
        
        # Week of month
        feature_df[f'{datetime_column}_week_of_month'] = ((dt_series.dt.day - 1) // 7) + 1
        
        return feature_df
    
    def extract_payroll_features(self, df: pd.DataFrame, 
                                datetime_column: str,
                                payroll_schedule: str = 'biweekly') -> pd.DataFrame:
        """Extract payroll-related features"""
        
        feature_df = df.copy()
        dt_series = pd.to_datetime(feature_df[datetime_column])
        
        if payroll_schedule == 'biweekly':
            # Assuming biweekly payroll on Fridays
            # Use a reference date (first Friday of year) and calculate biweekly intervals
            reference_date = pd.Timestamp('2024-01-05')  # First Friday of 2024
            days_since_ref = (dt_series - reference_date).dt.days
            is_payday = ((days_since_ref % 14 == 0) & (dt_series.dt.dayofweek == 4)).astype(int)
            feature_df[f'{datetime_column}_is_payday'] = is_payday
        
        elif payroll_schedule == 'monthly':
            # Last business day of month
            feature_df[f'{datetime_column}_is_payday'] = feature_df[f'{datetime_column}_is_month_end_business']
        
        elif payroll_schedule == 'semi_monthly':
            # 15th and last day of month (adjusted for weekends)
            is_15th = dt_series.dt.day == 15
            is_month_end = dt_series.dt.is_month_end
            feature_df[f'{datetime_column}_is_payday'] = (is_15th | is_month_end).astype(int)
        
        return feature_df

# Example usage
def demonstrate_business_calendar_features():
    """Demonstrate business calendar feature extraction"""
    
    # Create sample data spanning multiple months including holidays
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.uniform(100, 1000, len(dates))
    })
    
    print("Original Data Sample (Around Holiday Season):")
    holiday_sample = sample_data[(sample_data['date'] >= '2024-12-20') & 
                                (sample_data['date'] <= '2024-01-05')]
    print(holiday_sample.head())
    print()
    
    # Initialize business calendar generator
    business_calendar = BusinessCalendarFeatureGenerator(country='US')
    
    # Add custom holidays/events
    custom_events = {
        date(2024, 3, 15): 'Company Anniversary',
        date(2024, 8, 15): 'Summer Sale Event'
    }
    business_calendar.add_custom_business_rules(custom_events)
    
    # Extract holiday features
    holiday_df = business_calendar.extract_holiday_features(sample_data, 'date')
    
    print("Sample Holiday Features:")
    holiday_cols = [col for col in holiday_df.columns if 'holiday' in col]
    print(holiday_df[['date'] + holiday_cols].head(10))
    print()
    
    # Extract business day features
    business_df = business_calendar.extract_business_day_features(holiday_df, 'date')
    
    print("Sample Business Day Features:")
    business_cols = [col for col in business_df.columns if 'business' in col or 'week_of_month' in col]
    print(business_df[['date'] + business_cols[:3]].head(10))
    print()
    
    # Extract payroll features
    payroll_df = business_calendar.extract_payroll_features(business_df, 'date', 'biweekly')
    
    print("Sample Payroll Features:")
    payroll_cols = [col for col in payroll_df.columns if 'payday' in col]
    print(payroll_df[['date'] + payroll_cols].head(20))
    
    return sample_data, payroll_df, business_calendar

# Run demonstration
# original_calendar_data, business_calendar_data, calendar_generator = demonstrate_business_calendar_features()
```

---

## 💻 Complete Pipeline Implementation

### Comprehensive Temporal Feature Engineering Pipeline

```python
class ComprehensiveTemporalPipeline:
    def __init__(self, country: str = 'US'):
        """Initialize the complete temporal feature engineering pipeline"""
        
        # Initialize all component classes
        self.parser = UniversalDateTimeParser()
        self.timezone_manager = TimezoneManager()
        self.validator = DateRangeValidator()
        self.feature_extractor = TemporalFeatureExtractor()
        self.cyclical_encoder = CyclicalFeatureEncoder()
        self.aggregation_engine = TemporalAggregationEngine()
        self.lag_generator = LagFeatureGenerator()
        self.business_calendar = BusinessCalendarFeatureGenerator(country=country)
        
        # Pipeline configuration
        self.pipeline_config = {
            'parse_datetimes': True,
            'standardize_timezones': True,
            'validate_ranges': True,
            'extract_basic_features': True,
            'encode_cyclical': True,
            'create_aggregations': True,
            'create_lag_features': True,
            'add_business_calendar': True
        }
        
        # Pipeline execution log
        self.execution_log = {}
    
    def configure_pipeline(self, config: Dict):
        """Configure which pipeline steps to execute"""
        self.pipeline_config.update(config)
    
    def process_temporal_data(self, df: pd.DataFrame,
                            datetime_columns: List[str],
                            target_columns: List[str] = None,
                            group_columns: List[str] = None,
                            **kwargs) -> Dict:
        """Execute the complete temporal feature engineering pipeline"""
        
        print("=== COMPREHENSIVE TEMPORAL FEATURE ENGINEERING PIPELINE ===")
        start_time = pd.Timestamp.now()
        
        processed_df = df.copy()
        pipeline_results = {}
        
        # Step 1: Parse and standardize datetime columns
        if self.pipeline_config['parse_datetimes']:
            print("\n1. Parsing datetime columns...")
            parse_result = self.parser.batch_parse_datetime_columns(
                processed_df, 
                datetime_columns,
                default_timezone=kwargs.get('default_timezone', 'UTC')
            )
            processed_df = parse_result['parsed_df']
            pipeline_results['datetime_parsing'] = parse_result['batch_log']
        
        # Step 2: Timezone standardization
        if self.pipeline_config['standardize_timezones']:
            print("2. Standardizing timezones...")
            tz_result = self.timezone_manager.batch_timezone_standardization(
                processed_df,
                datetime_columns,
                target_timezone=kwargs.get('target_timezone', 'UTC')
            )
            processed_df = tz_result['standardized_df']
            pipeline_results['timezone_standardization'] = tz_result['batch_log']
        
        # Step 3: Date range validation
        if self.pipeline_config['validate_ranges']:
            print("3. Validating date ranges...")
            validation_result = self.validator.validate_date_ranges(
                processed_df,
                datetime_columns,
                custom_ranges=kwargs.get('custom_date_ranges')
            )
            pipeline_results['date_validation'] = validation_result
        
        # Step 4: Extract basic temporal features
        if self.pipeline_config['extract_basic_features']:
            print("4. Extracting basic temporal features...")
            for dt_col in datetime_columns:
                processed_df = self.feature_extractor.extract_basic_features(
                    processed_df,
                    dt_col,
                    features_to_extract=kwargs.get('basic_features')
                )
                processed_df = self.feature_extractor.extract_time_parts(
                    processed_df,
                    dt_col,
                    time_granularity=kwargs.get('time_granularity', 'hour')
                )
            pipeline_results['basic_features'] = self.feature_extractor.extraction_log
        
        # Step 5: Encode cyclical features
        if self.pipeline_config['encode_cyclical']:
            print("5. Encoding cyclical features...")
            for dt_col in datetime_columns:
                processed_df = self.cyclical_encoder.encode_temporal_cycles(
                    processed_df, dt_col
                )
            pipeline_results['cyclical_encoding'] = self.cyclical_encoder.encoding_log
        
        # Step 6: Create aggregation features
        if self.pipeline_config['create_aggregations'] and target_columns:
            print("6. Creating aggregation features...")
            
            # Rolling features
            processed_df = self.aggregation_engine.create_rolling_features(
                processed_df,
                datetime_columns[0],  # Use first datetime column as primary
                target_columns,
                window_sizes=kwargs.get('rolling_windows', ['7D', '30D']),
                aggregation_funcs=kwargs.get('agg_functions', ['mean', 'std'])
            )
            
            # Seasonal aggregations
            processed_df = self.aggregation_engine.create_seasonal_aggregations(
                processed_df,
                datetime_columns[0],
                target_columns
            )
            
            pipeline_results['aggregation_features'] = self.aggregation_engine.aggregation_log
        
        # Step 7: Create lag features
        if self.pipeline_config['create_lag_features'] and target_columns:
            print("7. Creating lag features...")
            
            # Lag features
            processed_df = self.lag_generator.create_lag_features(
                processed_df,
                datetime_columns[0],
                target_columns,
                lag_periods=kwargs.get('lag_periods', [1, 7, 30]),
                group_columns=group_columns
            )
            
            # Difference features
            processed_df = self.lag_generator.create_diff_features(
                processed_df,
                datetime_columns[0],
                target_columns,
                group_columns=group_columns
            )
            
            pipeline_results['lag_features'] = self.lag_generator.generation_log
        
        # Step 8: Add business calendar features
        if self.pipeline_config['add_business_calendar']:
            print("8. Adding business calendar features...")
            for dt_col in datetime_columns:
                processed_df = self.business_calendar.extract_holiday_features(
                    processed_df, dt_col
                )
                processed_df = self.business_calendar.extract_business_day_features(
                    processed_df, dt_col
                )
                processed_df = self.business_calendar.extract_payroll_features(
                    processed_df, dt_col,
                    payroll_schedule=kwargs.get('payroll_schedule', 'biweekly')
                )
        
        # Final summary
        end_time = pd.Timestamp.now()
        execution_time = (end_time - start_time).total_seconds()
        
        original_features = len(df.columns)
        final_features = len(processed_df.columns)
        features_created = final_features - original_features
        
        self.execution_log = {
            'start_time': start_time,
            'end_time': end_time,
            'execution_time_seconds': execution_time,
            'original_shape': df.shape,
            'final_shape': processed_df.shape,
            'features_created': features_created,
            'datetime_columns_processed': datetime_columns,
            'target_columns_processed': target_columns or [],
            'pipeline_config': self.pipeline_config
        }
        
        print(f"\n=== PIPELINE COMPLETED ===")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Features created: {features_created}")
        print(f"Final shape: {processed_df.shape}")
        
        return {
            'processed_df': processed_df,
            'pipeline_results': pipeline_results,
            'execution_log': self.execution_log
        }
    
    def generate_pipeline_report(self) -> str:
        """Generate comprehensive pipeline execution report"""
        
        if not self.execution_log:
            return "No pipeline execution data available."
        
        report = f"""
        Temporal Feature Engineering Pipeline Report
        ==========================================
        
        Execution Summary:
        - Start Time: {self.execution_log['start_time']}
        - End Time: {self.execution_log['end_time']}
        - Duration: {self.execution_log['execution_time_seconds']:.2f} seconds
        
        Data Transformation:
        - Original Shape: {self.execution_log['original_shape']}
        - Final Shape: {self.execution_log['final_shape']}
        - Features Created: {self.execution_log['features_created']}
        
        Columns Processed:
        - DateTime Columns: {', '.join(self.execution_log['datetime_columns_processed'])}
        - Target Columns: {', '.join(self.execution_log['target_columns_processed'])}
        
        Pipeline Steps Executed:
        """
        
        for step, enabled in self.execution_log['pipeline_config'].items():
            status = "✅" if enabled else "⏭️"
            report += f"\n  {status} {step.replace('_', ' ').title()}"
        
        return report

# Complete example with all features
def demonstrate_complete_pipeline():
    """Demonstrate the complete temporal feature engineering pipeline"""
    
    # Create comprehensive sample dataset
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    
    # Create realistic business data with temporal patterns
    sample_data = []
    for i, date in enumerate(dates):
        # Seasonal sales pattern
        seasonal_multiplier = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
        
        # Weekly pattern (higher on weekends)
        weekly_multiplier = 1.2 if date.dayofweek >= 5 else 1.0
        
        # Holiday boost
        if date.month == 12 and date.day >= 20:  # Christmas season
            holiday_multiplier = 1.5
        elif date.month == 11 and 22 <= date.day <= 28:  # Thanksgiving week
            holiday_multiplier = 1.3
        else:
            holiday_multiplier = 1.0
        
        base_sales = 1000
        sales = (base_sales * seasonal_multiplier * weekly_multiplier * 
                holiday_multiplier + np.random.normal(0, 50))
        
        sample_data.append({
            'transaction_date': date,
            'sales_amount': max(sales, 0),  # Ensure non-negative
            'customer_segment': np.random.choice(['Premium', 'Standard', 'Budget']),
            'region': np.random.choice(['North', 'South', 'East', 'West']),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'])
        })
    
    df = pd.DataFrame(sample_data)
    
    print("Original Dataset Sample:")
    print(df.head())
    print(f"Original shape: {df.shape}")
    print()
    
    # Initialize and configure pipeline
    pipeline = ComprehensiveTemporalPipeline(country='US')
    
    # Configure pipeline steps
    pipeline.configure_pipeline({
        'parse_datetimes': True,
        'standardize_timezones': True,
        'validate_ranges': True,
        'extract_basic_features': True,
        'encode_cyclical': True,
        'create_aggregations': True,
        'create_lag_features': True,
        'add_business_calendar': True
    })
    
    # Execute pipeline
    result = pipeline.process_temporal_data(
        df,
        datetime_columns=['transaction_date'],
        target_columns=['sales_amount'],
        group_columns=['customer_segment', 'region'],
        rolling_windows=['7D', '30D', '90D'],
        lag_periods=[1, 7, 14, 30],
        time_granularity='hour'
    )
    
    processed_df = result['processed_df']
    
    print("\nProcessed Dataset Sample:")
    print(f"Final shape: {processed_df.shape}")
    
    # Show sample of created features
    new_features = [col for col in processed_df.columns if col not in df.columns]
    print(f"\nTotal features created: {len(new_features)}")
    print("\nSample of created features:")
    for i, feature in enumerate(new_features[:20]):  # Show first 20 features
        print(f"  {i+1:2d}. {feature}")
    if len(new_features) > 20:
        print(f"  ... and {len(new_features) - 20} more features")
    
    # Generate and print pipeline report
    report = pipeline.generate_pipeline_report()
    print(report)
    
    return df, processed_df, pipeline

# Run complete pipeline demonstration
# original_complete_data, enhanced_complete_data, complete_pipeline = demonstrate_complete_pipeline()
```

---

## ✅ Temporal Feature Validation

### Comprehensive Validation Framework

```python
class TemporalFeatureValidator:
    def __init__(self):
        self.validation_results = {}
        self.validation_thresholds = {
            'missing_ratio_threshold': 0.1,
            'correlation_threshold': 0.95,
            'variance_threshold': 0.01
        }
    
    def validate_temporal_consistency(self, df: pd.DataFrame,
                                    datetime_columns: List[str]) -> Dict:
        """Validate temporal data consistency"""
        
        validation_results = {}
        
        for col in datetime_columns:
            if col not in df.columns:
                validation_results[col] = {'error': 'Column not found'}
                continue
            
            dt_series = pd.to_datetime(df[col], errors='coerce')
            
            # Check for temporal ordering
            is_sorted = dt_series.is_monotonic_increasing
            
            # Check for duplicates
            duplicate_count = dt_series.duplicated().sum()
            
            # Check for gaps
            if len(dt_series.dropna()) > 1:
                time_diffs = dt_series.dropna().diff().dropna()
                median_diff = time_diffs.median()
                large_gaps = (time_diffs > median_diff * 3).sum()
            else:
                large_gaps = 0
            
            validation_results[col] = {
                'is_temporally_sorted': is_sorted,
                'duplicate_timestamps': duplicate_count,
                'large_gaps_detected': large_gaps,
                'missing_values': dt_series.isna().sum(),
                'date_range': (dt_series.min(), dt_series.max()) if dt_series.notna().any() else (None, None)
            }
        
        return validation_results
    
    def validate_feature_quality(self, df: pd.DataFrame,
                                feature_columns: List[str]) -> Dict:
        """Validate quality of created temporal features"""
        
        quality_results = {}
        
        for col in feature_columns:
            if col not in df.columns:
                continue
            
            series = df[col]
            
            # Basic quality metrics
            missing_ratio = series.isna().sum() / len(series)
            variance = series.var() if series.dtype in ['int64', 'float64'] else None
            unique_ratio = series.nunique() / len(series)
            
            # Detect constant features
            is_constant = series.nunique() <= 1
            
            # Detect highly correlated features
            correlations = {}
            if series.dtype in ['int64', 'float64']:
                for other_col in feature_columns:
                    if other_col != col and other_col in df.columns:
                        other_series = df[other_col]
                        if other_series.dtype in ['int64', 'float64']:
                            try:
                                corr = series.corr(other_series)
                                if abs(corr) > self.validation_thresholds['correlation_threshold']:
                                    correlations[other_col] = corr
                            except:
                                pass
            
            quality_results[col] = {
                'missing_ratio': missing_ratio,
                'variance': variance,
                'unique_ratio': unique_ratio,
                'is_constant': is_constant,
                'high_correlations': correlations,
                'quality_score': self._calculate_quality_score(missing_ratio, variance, unique_ratio, is_constant)
            }
        
        return quality_results
    
    def _calculate_quality_score(self, missing_ratio: float, 
                               variance: float, 
                               unique_ratio: float, 
                               is_constant: bool) -> float:
        """Calculate overall quality score for a feature"""
        
        if is_constant:
            return 0.0
        
        score = 1.0
        
        # Penalize high missing ratio
        if missing_ratio > self.validation_thresholds['missing_ratio_threshold']:
            score -= missing_ratio * 0.5
        
        # Penalize low variance (for numerical features)
        if variance is not None and variance < self.validation_thresholds['variance_threshold']:
            score -= 0.3
        
        # Reward reasonable unique ratio
        if 0.01 < unique_ratio < 0.99:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def validate_cyclical_encoding(self, df: pd.DataFrame) -> Dict:
        """Validate cyclical encoding quality"""
        
        cyclical_features = {}
        
        # Find cyclical feature pairs (sin/cos)
        sin_features = [col for col in df.columns if '_sin' in col]
        
        for sin_col in sin_features:
            cos_col = sin_col.replace('_sin', '_cos')
            if cos_col in df.columns:
                sin_series = df[sin_col]
                cos_series = df[cos_col]
                
                # Validate sin^2 + cos^2 ≈ 1
                magnitude_check = (sin_series**2 + cos_series**2)
                magnitude_error = abs(magnitude_check - 1).mean()
                
                # Check range constraints
                sin_range_valid = (sin_series >= -1).all() and (sin_series <= 1).all()
                cos_range_valid = (cos_series >= -1).all() and (cos_series <= 1).all()
                
                cyclical_features[sin_col.replace('_sin', '')] = {
                    'magnitude_error': magnitude_error,
                    'sin_range_valid': sin_range_valid,
                    'cos_range_valid': cos_range_valid,
                    'encoding_valid': magnitude_error < 0.01 and sin_range_valid and cos_range_valid
                }
        
        return cyclical_features
    
    def validate_lag_features(self, df: pd.DataFrame,
                            datetime_column: str,
                            group_columns: List[str] = None) -> Dict:
        """Validate lag feature correctness"""
        
        lag_validation = {}
        
        # Find lag features
        lag_features = [col for col in df.columns if '_lag_' in col]
        
        for lag_col in lag_features:
            # Extract original column and lag period from feature name
            parts = lag_col.split('_lag_')
            if len(parts) == 2:
                original_col = parts[0]
                lag_period = int(parts[1])
                
                if original_col in df.columns:
                    # Sort by datetime for validation
                    df_sorted = df.sort_values(datetime_column)
                    
                    if group_columns:
                        # Validate within groups
                        correct_lags = 0
                        total_checks = 0
                        
                        for group_vals in df_sorted[group_columns].drop_duplicates().values:
                            group_filter = (df_sorted[group_columns] == group_vals).all(axis=1)
                            group_df = df_sorted[group_filter]
                            
                            if len(group_df) > lag_period:
                                # Check a few lag values
                                for i in range(lag_period, min(lag_period + 10, len(group_df))):
                                    expected_val = group_df.iloc[i - lag_period][original_col]
                                    actual_val = group_df.iloc[i][lag_col]
                                    
                                    if pd.notna(expected_val) and pd.notna(actual_val):
                                        if abs(expected_val - actual_val) < 1e-10:
                                            correct_lags += 1
                                        total_checks += 1
                        
                        accuracy = correct_lags / total_checks if total_checks > 0 else 0
                    else:
                        # Global lag validation
                        correct_lags = 0
                        total_checks = 0
                        
                        for i in range(lag_period, min(lag_period + 100, len(df_sorted))):
                            expected_val = df_sorted.iloc[i - lag_period][original_col]
                            actual_val = df_sorted.iloc[i][lag_col]
                            
                            if pd.notna(expected_val) and pd.notna(actual_val):
                                if abs(expected_val - actual_val) < 1e-10:
                                    correct_lags += 1
                                total_checks += 1
                        
                        accuracy = correct_lags / total_checks if total_checks > 0 else 0
                    
                    lag_validation[lag_col] = {
                        'original_column': original_col,
                        'lag_period': lag_period,
                        'validation_accuracy': accuracy,
                        'validation_passed': accuracy > 0.95
                    }
        
        return lag_validation

# Example validation
def demonstrate_temporal_validation():
    """Demonstrate temporal feature validation"""
    
    # Create sample data with some quality issues
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    
    # Create features with some problems
    sample_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.uniform(100, 1000, len(dates)),
        'constant_feature': 42,  # Constant feature
        'high_missing_feature': [np.nan if i % 3 == 0 else i for i in range(len(dates))],
        'normal_feature': np.random.normal(100, 20, len(dates))
    })
    
    # Add some temporal features for validation
    sample_data['sales_lag_1'] = sample_data['sales'].shift(1)
    sample_data['sales_lag_7'] = sample_data['sales'].shift(7)
    
    # Add cyclical features
    sample_data['month_sin'] = np.sin(2 * np.pi * sample_data['date'].dt.month / 12)
    sample_data['month_cos'] = np.cos(2 * np.pi * sample_data['date'].dt.month / 12)
    
    print("Sample Data for Validation:")
    print(sample_data.head())
    print()
    
    # Initialize validator
    validator = TemporalFeatureValidator()
    
    # Validate temporal consistency
    temporal_validation = validator.validate_temporal_consistency(sample_data, ['date'])
    print("Temporal Consistency Validation:")
    for col, results in temporal_validation.items():
        print(f"  {col}:")
        for key, value in results.items():
            print(f"    {key}: {value}")
    print()
    
    # Validate feature quality
    feature_cols = ['sales', 'constant_feature', 'high_missing_feature', 'normal_feature']
    quality_validation = validator.validate_feature_quality(sample_data, feature_cols)
    print("Feature Quality Validation:")
    for col, results in quality_validation.items():
        print(f"  {col}:")
        print(f"    Quality Score: {results['quality_score']:.3f}")
        print(f"    Missing Ratio: {results['missing_ratio']:.3f}")
        print(f"    Is Constant: {results['is_constant']}")
    print()
    
    # Validate cyclical encoding
    cyclical_validation = validator.validate_cyclical_encoding(sample_data)
    print("Cyclical Encoding Validation:")
    for feature, results in cyclical_validation.items():
        print(f"  {feature}:")
        print(f"    Encoding Valid: {results['encoding_valid']}")
        print(f"    Magnitude Error: {results['magnitude_error']:.6f}")
    print()
    
    # Validate lag features
    lag_validation = validator.validate_lag_features(sample_data, 'date')
    print("Lag Feature Validation:")
    for feature, results in lag_validation.items():
        print(f"  {feature}:")
        print(f"    Validation Passed: {results['validation_passed']}")
        print(f"    Accuracy: {results['validation_accuracy']:.3f}")
    
    return sample_data, validator

# Run validation demonstration
# validation_data, temporal_validator = demonstrate_temporal_validation()
```

---

## ⭐ Best Practices

### 1. **Timezone-Aware Data Handling**

```python
def best_practice_timezone_handling():
    """Best practices for timezone handling"""
    
    # ✅ Always store data in UTC
    def store_in_utc(df, datetime_col, source_timezone='America/New_York'):
        # Convert to UTC for storage
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        if df[datetime_col].dt.tz is None:
            df[datetime_col] = df[datetime_col].dt.tz_localize(source_timezone)
        return df[datetime_col].dt.tz_convert('UTC')
    
    # ✅ Convert to local time for analysis when needed
    def convert_for_analysis(utc_series, target_timezone='America/New_York'):
        return utc_series.dt.tz_convert(target_timezone)
    
    # ✅ Handle DST transitions gracefully
    def handle_dst_safely(datetime_series, timezone='America/New_York'):
        try:
            if datetime_series.dt.tz is None:
                return datetime_series.dt.tz_localize(timezone, ambiguous='infer', nonexistent='shift_forward')
            return datetime_series
        except:
            # Fallback to UTC if localization fails
            return datetime_series.dt.tz_localize('UTC')
```

### 2. **Feature Engineering Strategy**

```python
def strategic_feature_engineering():
    """Strategic approach to temporal feature engineering"""
    
    # ✅ Start with domain knowledge
    business_patterns = {
        'retail': ['holiday_effects', 'payday_cycles', 'seasonal_trends'],
        'finance': ['market_hours', 'quarter_end_effects', 'economic_calendar'],
        'healthcare': ['shift_patterns', 'weekend_effects', 'seasonal_illness'],
        'manufacturing': ['production_cycles', 'maintenance_schedules', 'supply_chain_timing']
    }
    
    # ✅ Create features incrementally and validate impact
    def incremental_feature_creation(df, datetime_col, target_col):
        baseline_performance = evaluate_model_performance(df, target_col)
        
        feature_groups = [
            'basic_temporal',
            'cyclical_encoding', 
            'lag_features',
            'rolling_aggregations',
            'business_calendar'
        ]
        
        for group in feature_groups:
            enhanced_df = add_feature_group(df, datetime_col, group)
            new_performance = evaluate_model_performance(enhanced_df, target_col)
            
            if new_performance > baseline_performance:
                df = enhanced_df
                baseline_performance = new_performance
                print(f"✅ {group} improved performance to {new_performance:.4f}")
            else:
                print(f"⏭️ {group} did not improve performance")
        
        return df
```

### 3. **Memory-Efficient Processing**

```python
def memory_efficient_temporal_processing():
    """Memory-efficient approaches for large temporal datasets"""
    
    # ✅ Process in chunks for large datasets
    def process_temporal_chunks(df, datetime_col, chunk_size=10000):
        processed_chunks = []
        
        for start_idx in range(0, len(df), chunk_size):
            end_idx = min(start_idx + chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx].copy()
            
            # Process chunk
            chunk = add_temporal_features(chunk, datetime_col)
            processed_chunks.append(chunk)
        
        return pd.concat(processed_chunks, ignore_index=True)
    
    # ✅ Use appropriate data types
    def optimize_temporal_dtypes(df):
        # Convert boolean features to int8
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype('int8')
        
        # Downcast numerical features
        int_cols = df.select_dtypes(include=['int64']).columns
        df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
        
        float_cols = df.select_dtypes(include=['float64']).columns  
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
        
        return df
```

---

## ⚠️ Common Pitfalls

### 1. **Data Leakage in Temporal Features**

```python
# ❌ Wrong: Using future information in lag features
def bad_lag_features(df, datetime_col, target_col):
    # This creates data leakage!
    df_sorted = df.sort_values(datetime_col)
    df_sorted['target_mean'] = df_sorted[target_col].mean()  # Uses future values!
    return df_sorted

# ✅ Correct: Expanding window to avoid future leakage
def safe_lag_features(df, datetime_col, target_col):
    df_sorted = df.sort_values(datetime_col)
    # Use expanding mean that only includes past values
    df_sorted['target_expanding_mean'] = df_sorted[target_col].expanding().mean().shift(1)
    return df_sorted
```

### 2. **Ignoring Timezone in Business Logic**

```python
# ❌ Wrong: Ignoring business timezone
def bad_business_hours(df, datetime_col):
    # This uses UTC time for business hour calculation
    df['is_business_hours'] = df[datetime_col].dt.hour.between(9, 17)
    return df

# ✅ Correct: Convert to business timezone first
def correct_business_hours(df, datetime_col, business_timezone='America/New_York'):
    # Convert to business timezone for accurate business hour calculation
    local_time = df[datetime_col].dt.tz_convert(business_timezone)
    df['is_business_hours'] = local_time.dt.hour.between(9, 17)
    return df
```

### 3. **Over-Engineering Cyclical Features**

```python
# ❌ Wrong: Creating too many unnecessary cyclical features
def excessive_cyclical_features(df, datetime_col):
    dt = pd.to_datetime(df[datetime_col])
    
    # Too many cyclical features - many won't be useful
    df['second_sin'] = np.sin(2 * np.pi * dt.dt.second / 60)
    df['second_cos'] = np.cos(2 * np.pi * dt.dt.second / 60)
    df['minute_sin'] = np.sin(2 * np.pi * dt.dt.minute / 60)
    df['minute_cos'] = np.cos(2 * np.pi * dt.dt.minute / 60)
    # ... many more
    
    return df

# ✅ Better: Focus on meaningful cyclical patterns
def focused_cyclical_features(df, datetime_col):
    dt = pd.to_datetime(df[datetime_col])
    
    # Only create cyclical features that match your data granularity and business logic
    if dt.dt.hour.nunique() > 1:  # Only if hour-level data exists
        df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    
    # Monthly patterns are often meaningful
    df['month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
    
    return df
```

---

## 📝 Summary

Temporal feature engineering is a powerful technique that can significantly enhance machine learning model performance by extracting meaningful patterns from date and time information. This comprehensive guide covered all essential aspects of temporal data processing and feature creation.

### Key Takeaways

1. **Data Quality First**: Always start with proper parsing, timezone standardization, and validation before feature engineering

2. **Domain-Driven Engineering**: Create features that align with your business domain and known temporal patterns

3. **Avoid Data Leakage**: Be extremely careful with lag features and time-based aggregations to prevent future information leakage

4. **Cyclical Encoding**: Use sine/cosine transformations for truly cyclical features like time of day, day of week, and month

5. **Business Calendar Integration**: Incorporate holidays, business days, and industry-specific events for more relevant features

### Feature Engineering Strategy

- **Basic Features**: Start with standard datetime components (year, month, day, etc.)
- **Cyclical Features**: Encode periodic patterns mathematically
- **Lag Features**: Capture historical dependencies safely
- **Aggregation Features**: Create rolling and seasonal statistics
- **Business Features**: Add domain-specific temporal knowledge

### Production Considerations

- **Performance**: Use efficient data types and chunked processing for large datasets
- **Validation**: Implement comprehensive validation to catch feature engineering errors
- **Monitoring**: Track temporal feature drift in production environments
- **Documentation**: Maintain clear documentation of temporal assumptions and business rules

### Advanced Techniques

The guide covered sophisticated techniques including:
- **Seasonal Decomposition**: Separate trend, seasonal, and residual components
- **Holiday Detection**: Automatic identification of holidays and special events
- **Business Calendar Features**: Integration with payroll schedules and fiscal calendars
- **Advanced Lag Features**: Momentum, volatility, and trend strength calculations

### Next Steps

With comprehensive temporal features, you're ready for:

- **Feature Selection**: Identifying the most predictive temporal features
- **Time Series Modeling**: Building sophisticated forecasting models
- **Anomaly Detection**: Using temporal patterns to identify unusual events
- **Real-time Systems**: Implementing temporal features in streaming environments

### Final Recommendations

1. **Start Simple**: Begin with basic temporal features and add complexity incrementally
2. **Validate Impact**: Always measure the performance impact of new temporal features
3. **Think Business Context**: Consider how time affects your specific business problem
4. **Handle Edge Cases**: Plan for timezone changes, data gaps, and irregular patterns
5. **Monitor Continuously**: Temporal patterns can change over time - monitor and adapt

Remember: Effective temporal feature engineering requires understanding both the technical aspects of time series data and the business context in which temporal patterns occur. The most successful implementations combine statistical rigor with domain expertise.

---

**Related Guides in This Series:**

- [Exploratory Data Analysis](./exploratory_data_analysis.md)
- [Missing Data Imputation](./missing_data_imputation.md)
- [Duplication and Outlier Handling](./duplication_outlier_handling.md)
- [Categorical Variable Cleaning](./categorical_variable_cleaning.md)
- [Numerical Variable Cleaning](./numerical_variable_cleaning.md)
- Advanced Feature Engineering and Selection (coming soon)
- Model Selection and Validation (coming soon)
```