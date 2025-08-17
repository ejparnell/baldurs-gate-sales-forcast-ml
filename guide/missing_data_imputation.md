# 🔧 Complete Missing Data Handling and Imputation Guide

This guide is ideal for data analysts, data scientists, and ML engineers working in Python who want a systematic, repeatable framework for handling missing data and applying imputation techniques.

This guide is a part of a larger series on Machine Learning Pipelines. Each section is designed to be modular, allowing you to adapt the code and techniques to your specific dataset and analysis needs.

**The guide before:** [Exploratory Data Analysis](./exploratory_data_analysis.md)  
**The guide after:** [Duplication and Outlier Handling](./duplication_outlier_handling.md)

## 📋 Table of Contents

1. [📖 Overview](#-overview)
2. [🧠 Understanding Missing Data](#-understanding-missing-data)
3. [🔍 Missing Data Analysis](#-missing-data-analysis)
4. [📝 Imputation Strategy Decision Framework](#-imputation-strategy-decision-framework)
5. [🛠️ Imputation Techniques](#️-imputation-techniques)
6. [💻 Code Examples](#-code-examples)
7. [✅ Validation and Quality Checks](#-validation-and-quality-checks)
8. [⭐ Best Practices](#-best-practices)
9. [⚠️ Common Pitfalls](#️-common-pitfalls)
10. [📝 Summary](#-summary)

## 📖 Overview

After you have explored your data and identified the missing values, the next step is to handle these missing data points effectively. This guide will walk you through various imputation techniques, including mean, median, mode, and more advanced methods like KNN and MICE.

But before you change any data, it's crucial to understand the nature of the missing data. Are they missing at random, or is there a pattern? Does the missingness itself carry information? Sometimes, you might not even want to impute missing values if they are informative.

So moral of the story is:

**DO NOT SKIP EXPLORATORY DATA ANALYSIS!**

### Why Missing Data Handling Matters

Proper missing data handling is crucial because:

- **Algorithm Requirements**: Many ML algorithms cannot handle missing values
- **Bias Prevention**: Poor imputation can introduce systematic bias
- **Model Performance**: Thoughtful imputation often improves model accuracy
- **Data Integrity**: Maintains the underlying patterns in your data
- **Production Readiness**: Ensures your model can handle real-world data gaps

### Key Principles

1. **Understand Before Acting**: Always analyze missingness patterns first
2. **Preserve Information**: Choose methods that maintain data relationships
3. **Consider Domain Knowledge**: Use business logic to guide decisions
4. **Validate Impact**: Always check how imputation affects your data
5. **Document Decisions**: Record your rationale for reproducibility

## 🧠 Understanding Missing Data

### Types of Missing Data

Understanding the mechanism behind missing data is crucial for choosing the right imputation strategy:

#### 1. **Missing Completely at Random (MCAR)**

- Missing values are completely independent of any observed or unobserved variables
  - **Example**: Survey responses lost due to computer malfunction
  - **Implication**: Can be ignored or simple imputation works well
  - **Test**: Little's MCAR test, missing pattern analysis

#### 2. **Missing at Random (MAR)**

- Missing values depend on observed variables but not on the missing values themselves
  - **Example**: Income not reported by older respondents (age is observed)
  - **Implication**: Can be handled with sophisticated imputation using observed variables
  - **Strategy**: Use other variables to predict missing values

#### 3. **Missing Not at Random (MNAR)**

- Missing values depend on the unobserved values themselves
  - **Example**: High earners not reporting income due to privacy concerns
  - **Implication**: Requires domain knowledge and careful modeling
  - **Strategy**: May need to model the missingness mechanism explicitly

### Common Causes of Missing Data

| Cause | Example | Impact | Strategy |
|-------|---------|---------|----------|
| **Data Collection Issues** | Sensor failures, survey non-response | Random patterns | Technical fixes, simple imputation |
| **System Limitations** | Optional fields, legacy system gaps | Systematic patterns | Business rules, domain-specific imputation |
| **Privacy/Confidentiality** | Sensitive information withheld | Non-random patterns | Special handling, missingness indicators |
| **Natural Missingness** | Future events, inapplicable questions | Logical patterns | Domain-specific rules, custom imputation |
| **Data Processing Errors** | ETL failures, transformation issues | Various patterns | Data quality improvements, validation |

## 🔍 Missing Data Analysis

Before any imputation, conduct thorough missing data analysis:

### Missing Data Patterns

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno

# Comprehensive missing data analysis
def analyze_missing_data(df):
    """
    Comprehensive missing data analysis
    """
    print("=== MISSING DATA ANALYSIS ===")
    
    # Basic missing data summary
    missing_summary = df.isnull().sum()
    missing_percentage = (missing_summary / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_summary.index,
        'Missing_Count': missing_summary.values,
        'Missing_Percentage': missing_percentage.values,
        'Data_Type': df.dtypes.values
    }).sort_values('Missing_Percentage', ascending=False)
    
    print("Missing Data Summary:")
    display(missing_df[missing_df['Missing_Count'] > 0])
    
    # Missing data patterns visualization
    if missing_df['Missing_Count'].sum() > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Missing data matrix
        msno.matrix(df, ax=axes[0,0])
        axes[0,0].set_title('Missing Data Pattern Matrix')
        
        # Missing data heatmap
        msno.heatmap(df, ax=axes[0,1])
        axes[0,1].set_title('Missing Data Correlation Heatmap')
        
        # Missing data bar chart
        msno.bar(df, ax=axes[1,0])
        axes[1,0].set_title('Missing Data Count by Column')
        
        # Missing data dendrogram
        msno.dendrogram(df, ax=axes[1,1])
        axes[1,1].set_title('Missing Data Clustering')
        
        plt.tight_layout()
        plt.show()
    
    return missing_df

# Analyze patterns between missing values
def analyze_missing_patterns(df):
    """
    Analyze relationships between missing values
    """
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    if len(missing_cols) > 1:
        # Create missingness indicators
        missing_indicators = df[missing_cols].isnull().astype(int)
        
        # Calculate correlation between missingness patterns
        missing_corr = missing_indicators.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Correlation Between Missing Value Patterns')
        plt.show()
        
        # Identify co-occurring missing patterns
        pattern_counts = missing_indicators.value_counts().head(10)
        print("Top 10 Missing Value Patterns:")
        for pattern, count in pattern_counts.items():
            pattern_desc = [f"{col}={'Missing' if val else 'Present'}" 
                          for col, val in zip(missing_cols, pattern)]
            print(f"  {count:4d} rows: {', '.join(pattern_desc)}")
    
    return missing_indicators if len(missing_cols) > 1 else None
```

### Missingness Tests

```python
# Test for MCAR (Missing Completely at Random)
def test_mcar(df):
    """
    Test if data is Missing Completely at Random using Little's MCAR test
    Note: This is a simplified version - use impyute.diagnostics.mcar_test for full implementation
    """
    from scipy.stats import chi2
    
    # Create missingness indicators
    missing_indicator = df.isnull()
    
    # Count missing patterns
    pattern_counts = missing_indicator.value_counts()
    
    print("Missing Completely at Random (MCAR) Analysis:")
    print(f"Number of unique missing patterns: {len(pattern_counts)}")
    print(f"Most common pattern frequency: {pattern_counts.iloc[0] / len(df):.2%}")
    
    # Simple pattern analysis (not a formal Little's test)
    if len(pattern_counts) == 1:
        print("✅ Only one missing pattern - likely MCAR or complete data")
    elif pattern_counts.iloc[0] / len(df) > 0.8:
        print("⚠️ One pattern dominates - investigate further")
    else:
        print("🔍 Multiple patterns detected - likely MAR or MNAR")

# Analyze missing data by groups
def analyze_missing_by_groups(df, group_columns):
    """
    Analyze missing data patterns across different groups
    """
    for group_col in group_columns:
        if group_col in df.columns and df[group_col].notna().any():
            print(f"\nMissing Data Analysis by {group_col}:")
            
            missing_by_group = df.groupby(group_col).apply(
                lambda x: x.isnull().sum() / len(x) * 100
            ).round(2)
            
            # Show columns with varying missingness across groups
            varying_cols = []
            for col in missing_by_group.columns:
                if missing_by_group[col].std() > 5:  # 5% standard deviation threshold
                    varying_cols.append(col)
            
            if varying_cols:
                print("Columns with varying missingness across groups:")
                display(missing_by_group[varying_cols])
            else:
                print("No significant variation in missingness across groups")
```

## 📝 Imputation Strategy Decision Framework

### Decision Tree for Imputation Strategy

```md
1. Is the missing percentage < 5%?
   ├─ Yes: Consider simple imputation (mean/median/mode)
   └─ No: Go to step 2

2. Is the missing percentage > 50%?
   ├─ Yes: Consider dropping the column or creating missing indicator
   └─ No: Go to step 3

3. Is the data MCAR?
   ├─ Yes: Simple imputation methods work well
   └─ No: Go to step 4

4. Are there correlated variables available?
   ├─ Yes: Use advanced imputation (KNN, MICE, ML-based)
   └─ No: Use domain knowledge or business rules

5. Is computational efficiency critical?
   ├─ Yes: Use simpler methods (mean/median/mode)
   └─ No: Use advanced methods for better accuracy
```

### Imputation Strategy Matrix

| Missing % | Data Type | Pattern | Recommended Strategy |
|-----------|-----------|---------|---------------------|
| < 5% | Numerical | Any | Mean/Median imputation |
| < 5% | Categorical | Any | Mode imputation |
| 5-20% | Numerical | MCAR | Mean/Median + validation |
| 5-20% | Numerical | MAR | KNN or MICE |
| 5-20% | Categorical | Any | Mode or create "Unknown" category |
| 20-50% | Any | MCAR | Consider advanced methods |
| 20-50% | Any | MAR/MNAR | Advanced methods + missing indicators |
| > 50% | Any | Any | Consider dropping or missing indicators |

## 🛠️ Imputation Techniques

### 1. Simple Imputation Methods

#### Numerical Variables

```python
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

def simple_numerical_imputation(df, strategy='median'):
    """
    Simple imputation for numerical variables
    
    Parameters:
    - strategy: 'mean', 'median', 'most_frequent', or 'constant'
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    missing_numerical = [col for col in numerical_cols if df[col].isnull().any()]
    
    if not missing_numerical:
        print("No missing values in numerical columns")
        return df.copy()
    
    df_imputed = df.copy()
    
    # Simple imputation
    imputer = SimpleImputer(strategy=strategy)
    df_imputed[missing_numerical] = imputer.fit_transform(df[missing_numerical])
    
    print(f"Applied {strategy} imputation to columns: {missing_numerical}")
    return df_imputed

# Advanced constant imputation with business logic
def business_rule_imputation(df, imputation_rules):
    """
    Apply business rule-based imputation
    
    Parameters:
    - imputation_rules: dict with column names as keys and imputation logic as values
    
    Example:
    rules = {
        'income': lambda row: row['industry_median_income'] if pd.isna(row['income']) else row['income'],
        'age': lambda row: 25 if pd.isna(row['age']) and row['student'] == True else row['age']
    }
    """
    df_imputed = df.copy()
    
    for column, rule in imputation_rules.items():
        if column in df.columns:
            df_imputed[column] = df_imputed.apply(rule, axis=1)
            print(f"Applied business rule imputation to {column}")
    
    return df_imputed
```

#### Categorical Variables

```python
def categorical_imputation(df, strategy='most_frequent', unknown_category='Unknown'):
    """
    Imputation for categorical variables
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    missing_categorical = [col for col in categorical_cols if df[col].isnull().any()]
    
    if not missing_categorical:
        print("No missing values in categorical columns")
        return df.copy()
    
    df_imputed = df.copy()
    
    for col in missing_categorical:
        if strategy == 'most_frequent':
            most_frequent = df[col].mode().iloc[0] if not df[col].mode().empty else unknown_category
            df_imputed[col].fillna(most_frequent, inplace=True)
        elif strategy == 'unknown':
            df_imputed[col].fillna(unknown_category, inplace=True)
        elif strategy == 'forward_fill':
            df_imputed[col].fillna(method='ffill', inplace=True)
        elif strategy == 'backward_fill':
            df_imputed[col].fillna(method='bfill', inplace=True)
    
    print(f"Applied {strategy} imputation to categorical columns: {missing_categorical}")
    return df_imputed
```

### 2. Advanced Imputation Methods

#### K-Nearest Neighbors (KNN) Imputation

```python
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def knn_imputation(df, n_neighbors=5, weights='uniform'):
    """
    KNN-based imputation for numerical variables
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    missing_numerical = [col for col in numerical_cols if df[col].isnull().any()]
    
    if not missing_numerical:
        print("No missing values in numerical columns")
        return df.copy()
    
    df_imputed = df.copy()
    
    # Standardize data before KNN (important for distance calculation)
    scaler = StandardScaler()
    
    # Apply KNN imputation
    knn_imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    
    # Scale, impute, and inverse scale
    scaled_data = scaler.fit_transform(df[numerical_cols])
    imputed_scaled = knn_imputer.fit_transform(scaled_data)
    imputed_data = scaler.inverse_transform(imputed_scaled)
    
    df_imputed[numerical_cols] = imputed_data
    
    print(f"Applied KNN imputation (k={n_neighbors}) to numerical columns")
    return df_imputed
```

#### Multiple Imputation by Chained Equations (MICE)

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

def mice_imputation(df, max_iter=10, random_state=42):
    """
    MICE (Multiple Imputation by Chained Equations) imputation
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    missing_numerical = [col for col in numerical_cols if df[col].isnull().any()]
    
    if not missing_numerical:
        print("No missing values in numerical columns")
        return df.copy()
    
    df_imputed = df.copy()
    
    # Use Random Forest as the estimator for MICE
    mice_imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=random_state),
        max_iter=max_iter,
        random_state=random_state
    )
    
    # Apply MICE imputation
    imputed_data = mice_imputer.fit_transform(df[numerical_cols])
    df_imputed[numerical_cols] = imputed_data
    
    print(f"Applied MICE imputation (max_iter={max_iter}) to numerical columns")
    return df_imputed

# Advanced MICE with different estimators
def advanced_mice_imputation(df, estimator_map=None, max_iter=10):
    """
    MICE with different estimators for different variable types
    """
    if estimator_map is None:
        estimator_map = {
            'default': RandomForestRegressor(n_estimators=10),
            'binary': RandomForestClassifier(n_estimators=10),
            'categorical': RandomForestClassifier(n_estimators=10)
        }
    
    # Implement column-specific MICE logic here
    # This is a simplified version - full implementation would handle different data types
    return mice_imputation(df, max_iter=max_iter)
```

### 3. Machine Learning-Based Imputation

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score

def ml_based_imputation(df, target_column, method='random_forest'):
    """
    Use machine learning models to predict missing values
    """
    df_imputed = df.copy()
    missing_cols = [col for col in df.columns if df[col].isnull().any() and col != target_column]
    
    for col in missing_cols:
        # Separate data with and without missing values for this column
        train_data = df[df[col].notna()]
        predict_data = df[df[col].isna()]
        
        if len(train_data) == 0 or len(predict_data) == 0:
            continue
        
        # Select features (all other columns except the target)
        feature_cols = [c for c in df.columns if c != col and c != target_column]
        feature_cols = [c for c in feature_cols if train_data[c].notna().all()]
        
        if len(feature_cols) == 0:
            continue
        
        X_train = train_data[feature_cols]
        y_train = train_data[col]
        X_predict = predict_data[feature_cols]
        
        # Choose model based on data type
        if df[col].dtype == 'object':
            model = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Train and predict
        model.fit(X_train, y_train)
        predictions = model.predict(X_predict)
        
        # Fill missing values
        df_imputed.loc[df[col].isna(), col] = predictions
        
        print(f"Applied {method} imputation to {col}")
    
    return df_imputed
```

### 4. Domain-Specific Imputation

```python
def time_series_imputation(df, time_column, method='interpolation'):
    """
    Time series specific imputation methods
    """
    df_imputed = df.copy()
    
    # Ensure datetime column
    if df[time_column].dtype != 'datetime64[ns]':
        df_imputed[time_column] = pd.to_datetime(df_imputed[time_column])
    
    # Sort by time
    df_imputed = df_imputed.sort_values(time_column)
    
    numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        if df_imputed[col].isnull().any():
            if method == 'interpolation':
                df_imputed[col] = df_imputed[col].interpolate(method='linear')
            elif method == 'forward_fill':
                df_imputed[col] = df_imputed[col].fillna(method='ffill')
            elif method == 'backward_fill':
                df_imputed[col] = df_imputed[col].fillna(method='bfill')
            elif method == 'seasonal':
                # Simple seasonal imputation (replace with seasonal decomposition for production)
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
    
    print(f"Applied time series {method} imputation")
    return df_imputed

def geographic_imputation(df, location_columns, method='regional_median'):
    """
    Geographic/spatial imputation methods
    """
    df_imputed = df.copy()
    
    # Example: impute based on regional/geographical groupings
    if method == 'regional_median':
        for location_col in location_columns:
            if location_col in df.columns:
                numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
                
                for num_col in numerical_cols:
                    if df_imputed[num_col].isnull().any():
                        # Group by location and fill with median
                        location_medians = df_imputed.groupby(location_col)[num_col].median()
                        
                        for location, median_val in location_medians.items():
                            mask = (df_imputed[location_col] == location) & (df_imputed[num_col].isnull())
                            df_imputed.loc[mask, num_col] = median_val
    
    print(f"Applied geographic {method} imputation")
    return df_imputed
```

## 💻 Code Examples

### Complete Imputation Pipeline

```python
class ImputationPipeline:
    """
    Complete imputation pipeline with validation and logging
    """
    
    def __init__(self, strategy_config=None):
        self.strategy_config = strategy_config or {}
        self.imputation_log = []
        self.original_stats = {}
        self.imputed_stats = {}
    
    def fit_transform(self, df, validation_split=0.2):
        """
        Complete imputation pipeline
        """
        print("=== STARTING IMPUTATION PIPELINE ===")
        
        # 1. Store original statistics
        self._store_original_stats(df)
        
        # 2. Analyze missing data
        missing_analysis = analyze_missing_data(df)
        
        # 3. Split data for validation
        if validation_split > 0:
            train_df, val_df = self._split_for_validation(df, validation_split)
        else:
            train_df, val_df = df.copy(), None
        
        # 4. Apply imputation strategies
        imputed_df = self._apply_imputation_strategies(train_df)
        
        # 5. Validate imputation quality
        if val_df is not None:
            self._validate_imputation(val_df, imputed_df)
        
        # 6. Store final statistics
        self._store_imputed_stats(imputed_df)
        
        # 7. Generate imputation report
        self._generate_report()
        
        return imputed_df
    
    def _store_original_stats(self, df):
        """Store original data statistics"""
        self.original_stats = {
            'shape': df.shape,
            'missing_count': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100
        }
        
        # Store column-wise statistics
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                self.original_stats[f'{col}_mean'] = df[col].mean()
                self.original_stats[f'{col}_std'] = df[col].std()
    
    def _apply_imputation_strategies(self, df):
        """Apply different imputation strategies based on configuration"""
        df_imputed = df.copy()
        
        # Default strategies if not specified
        default_strategies = {
            'numerical_simple': {'method': 'median', 'threshold': 0.05},
            'categorical_simple': {'method': 'most_frequent', 'threshold': 0.05},
            'numerical_advanced': {'method': 'knn', 'threshold': 0.20},
            'categorical_advanced': {'method': 'unknown', 'threshold': 0.20}
        }
        
        strategies = {**default_strategies, **self.strategy_config}
        
        # Get missing data summary
        missing_summary = df.isnull().sum() / len(df)
        
        # Apply strategies based on data type and missing percentage
        for col in df.columns:
            missing_pct = missing_summary[col]
            
            if missing_pct == 0:
                continue
            
            if df[col].dtype in ['int64', 'float64']:
                # Numerical column
                if missing_pct <= strategies['numerical_simple']['threshold']:
                    # Simple imputation
                    if strategies['numerical_simple']['method'] == 'median':
                        df_imputed[col].fillna(df[col].median(), inplace=True)
                    elif strategies['numerical_simple']['method'] == 'mean':
                        df_imputed[col].fillna(df[col].mean(), inplace=True)
                
                elif missing_pct <= strategies['numerical_advanced']['threshold']:
                    # Advanced imputation
                    if strategies['numerical_advanced']['method'] == 'knn':
                        df_imputed = knn_imputation(df_imputed)
                        break  # KNN imputes all columns at once
                    elif strategies['numerical_advanced']['method'] == 'mice':
                        df_imputed = mice_imputation(df_imputed)
                        break  # MICE imputes all columns at once
                
                else:
                    # High missing percentage - consider dropping or indicator
                    print(f"Warning: {col} has {missing_pct:.1%} missing values")
            
            else:
                # Categorical column
                if missing_pct <= strategies['categorical_simple']['threshold']:
                    # Simple imputation
                    if strategies['categorical_simple']['method'] == 'most_frequent':
                        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                        df_imputed[col].fillna(mode_val, inplace=True)
                
                elif missing_pct <= strategies['categorical_advanced']['threshold']:
                    # Advanced imputation
                    df_imputed[col].fillna('Unknown', inplace=True)
                
                else:
                    # High missing percentage
                    print(f"Warning: {col} has {missing_pct:.1%} missing values")
            
            # Log the imputation
            self.imputation_log.append({
                'column': col,
                'missing_percentage': missing_pct,
                'method': 'applied_strategy',
                'data_type': str(df[col].dtype)
            })
        
        return df_imputed
    
    def _split_for_validation(self, df, validation_split):
        """Split data for validation"""
        from sklearn.model_selection import train_test_split
        
        # Only split rows with complete data for validation
        complete_rows = df.dropna()
        
        if len(complete_rows) < 10:
            print("Warning: Too few complete rows for validation")
            return df, None
        
        train_idx, val_idx = train_test_split(
            complete_rows.index, 
            test_size=validation_split, 
            random_state=42
        )
        
        train_df = df.drop(val_idx)
        val_df = df.loc[val_idx]
        
        return train_df, val_df
    
    def _validate_imputation(self, original_df, imputed_df):
        """Validate imputation quality"""
        print("\n=== IMPUTATION VALIDATION ===")
        
        # Compare distributions
        numerical_cols = original_df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in imputed_df.columns:
                original_mean = original_df[col].mean()
                imputed_mean = imputed_df[col].mean()
                
                print(f"{col}: Original mean = {original_mean:.3f}, "
                      f"Imputed mean = {imputed_mean:.3f}, "
                      f"Difference = {abs(original_mean - imputed_mean):.3f}")
    
    def _store_imputed_stats(self, df):
        """Store imputed data statistics"""
        self.imputed_stats = {
            'shape': df.shape,
            'missing_count': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100
        }
    
    def _generate_report(self):
        """Generate imputation report"""
        print("\n=== IMPUTATION REPORT ===")
        print(f"Original missing values: {self.original_stats['missing_count']}")
        print(f"Final missing values: {self.imputed_stats['missing_count']}")
        print(f"Missing percentage reduced from {self.original_stats['missing_percentage']:.2f}% to {self.imputed_stats['missing_percentage']:.2f}%")
        
        print("\nImputation log:")
        for entry in self.imputation_log:
            print(f"  {entry['column']}: {entry['method']} ({entry['missing_percentage']:.1%} missing)")

# Usage example
def complete_imputation_example(df):
    """
    Complete example of using the imputation pipeline
    """
    # Define custom strategies
    custom_strategies = {
        'numerical_simple': {'method': 'median', 'threshold': 0.05},
        'numerical_advanced': {'method': 'knn', 'threshold': 0.30},
        'categorical_simple': {'method': 'most_frequent', 'threshold': 0.10},
        'categorical_advanced': {'method': 'unknown', 'threshold': 0.40}
    }
    
    # Create and run pipeline
    pipeline = ImputationPipeline(strategy_config=custom_strategies)
    imputed_df = pipeline.fit_transform(df, validation_split=0.2)
    
    return imputed_df
```

## ✅ Validation and Quality Checks

### Imputation Quality Assessment

```python
def assess_imputation_quality(original_df, imputed_df):
    """
    Comprehensive assessment of imputation quality
    """
    print("=== IMPUTATION QUALITY ASSESSMENT ===")
    
    # 1. Basic completeness check
    original_missing = original_df.isnull().sum().sum()
    imputed_missing = imputed_df.isnull().sum().sum()
    
    print(f"Missing values before: {original_missing}")
    print(f"Missing values after: {imputed_missing}")
    print(f"Improvement: {original_missing - imputed_missing} values imputed")
    
    # 2. Distribution comparison for numerical columns
    numerical_cols = original_df.select_dtypes(include=[np.number]).columns
    
    print("\n=== DISTRIBUTION COMPARISON ===")
    for col in numerical_cols:
        if original_df[col].isnull().any():
            original_stats = original_df[col].describe()
            imputed_stats = imputed_df[col].describe()
            
            print(f"\n{col}:")
            print(f"  Original mean: {original_stats['mean']:.3f} -> Imputed mean: {imputed_stats['mean']:.3f}")
            print(f"  Original std:  {original_stats['std']:.3f} -> Imputed std:  {imputed_stats['std']:.3f}")
            
            # Statistical test for distribution similarity
            from scipy.stats import ks_2samp
            original_complete = original_df[col].dropna()
            imputed_complete = imputed_df[col].dropna()
            
            if len(original_complete) > 0 and len(imputed_complete) > 0:
                ks_stat, p_value = ks_2samp(original_complete, imputed_complete)
                print(f"  KS test p-value: {p_value:.4f} ({'Similar' if p_value > 0.05 else 'Different'} distributions)")
    
    # 3. Categorical distribution comparison
    categorical_cols = original_df.select_dtypes(include=['object']).columns
    
    print("\n=== CATEGORICAL DISTRIBUTION COMPARISON ===")
    for col in categorical_cols:
        if original_df[col].isnull().any():
            original_counts = original_df[col].value_counts(normalize=True)
            imputed_counts = imputed_df[col].value_counts(normalize=True)
            
            print(f"\n{col} - Top categories:")
            for cat in original_counts.head(3).index:
                orig_pct = original_counts.get(cat, 0) * 100
                imp_pct = imputed_counts.get(cat, 0) * 100
                print(f"  {cat}: {orig_pct:.1f}% -> {imp_pct:.1f}%")
    
    # 4. Correlation preservation
    print("\n=== CORRELATION PRESERVATION ===")
    if len(numerical_cols) > 1:
        original_corr = original_df[numerical_cols].corr()
        imputed_corr = imputed_df[numerical_cols].corr()
        
        # Calculate correlation between correlation matrices
        corr_preservation = np.corrcoef(
            original_corr.values.flatten(),
            imputed_corr.values.flatten()
        )[0, 1]
        
        print(f"Correlation structure preservation: {corr_preservation:.3f}")
        print("(1.0 = perfect preservation, <0.8 = significant change)")

def create_imputation_report(original_df, imputed_df, output_file='imputation_report.html'):
    """
    Create a comprehensive HTML report of imputation results
    """
    from datetime import datetime
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Missing Data Imputation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; }}
            .section {{ margin: 20px 0; }}
            .stats-table {{ border-collapse: collapse; width: 100%; }}
            .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .stats-table th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Missing Data Imputation Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            <table class="stats-table">
                <tr><th>Metric</th><th>Before Imputation</th><th>After Imputation</th></tr>
                <tr><td>Total Missing Values</td><td>{original_df.isnull().sum().sum()}</td><td>{imputed_df.isnull().sum().sum()}</td></tr>
                <tr><td>Missing Percentage</td><td>{(original_df.isnull().sum().sum() / original_df.size * 100):.2f}%</td><td>{(imputed_df.isnull().sum().sum() / imputed_df.size * 100):.2f}%</td></tr>
                <tr><td>Columns with Missing Data</td><td>{(original_df.isnull().any()).sum()}</td><td>{(imputed_df.isnull().any()).sum()}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Column-wise Analysis</h2>
            <table class="stats-table">
                <tr><th>Column</th><th>Data Type</th><th>Missing Before</th><th>Missing After</th><th>Imputation Rate</th></tr>
    """
    
    for col in original_df.columns:
        missing_before = original_df[col].isnull().sum()
        missing_after = imputed_df[col].isnull().sum()
        imputation_rate = ((missing_before - missing_after) / max(missing_before, 1)) * 100
        
        html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{original_df[col].dtype}</td>
                    <td>{missing_before}</td>
                    <td>{missing_after}</td>
                    <td>{imputation_rate:.1f}%</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Imputation report saved to {output_file}")
```

## ⭐ Best Practices

### Production-Ready Imputation Guidelines

1. **Always Understand Your Data First**
   - Conduct thorough EDA before any imputation
   - Understand the business context behind missing values
   - Test different imputation strategies on validation data

2. **Choose Appropriate Methods**
   - Simple methods for low missing percentages (< 5%)
   - Advanced methods for moderate missing percentages (5-20%)
   - Consider dropping features with very high missing percentages (> 50%)

3. **Preserve Data Relationships**
   - Use methods that maintain correlations between variables
   - Consider creating missing value indicators for informative missingness
   - Validate that imputation doesn't distort underlying patterns

4. **Handle Different Data Types Appropriately**
   - Numerical: Mean/median for MCAR, KNN/MICE for MAR
   - Categorical: Mode or "Unknown" category
   - Ordinal: Preserve order relationships
   - Time series: Use temporal methods (interpolation, seasonal)

5. **Validate Imputation Quality**
   - Compare distributions before and after imputation
   - Check correlation preservation
   - Use holdout validation when possible
   - Monitor model performance impact

6. **Document Everything**
   - Record imputation methods and rationale
   - Save imputation parameters for reproducibility
   - Create comprehensive reports for stakeholders

7. **Consider Computational Efficiency**
   - Simple methods for real-time applications
   - Advanced methods for batch processing
   - Balance accuracy vs. speed based on use case

## ⚠️ Common Pitfalls

### Mistakes to Avoid During Imputation

1. **Imputing Without Understanding**
   - **Problem**: Applying imputation blindly without analyzing missingness patterns
   - **Solution**: Always conduct thorough missing data analysis first
   - **Impact**: Can introduce bias and distort relationships

2. **Using Mean Imputation for Everything**
   - **Problem**: Mean imputation reduces variance and can distort distributions
   - **Solution**: Use median for skewed data, consider advanced methods for MAR data
   - **Impact**: Underestimates uncertainty and correlations

3. **Ignoring Missing Data Mechanism**
   - **Problem**: Using MCAR methods for MNAR data
   - **Solution**: Understand why data is missing and choose appropriate methods
   - **Impact**: Can perpetuate bias and lead to incorrect conclusions

4. **Data Leakage in Imputation**
   - **Problem**: Using test data to compute imputation parameters
   - **Solution**: Fit imputation models only on training data
   - **Impact**: Overoptimistic model performance estimates

5. **Not Validating Imputation Quality**
   - **Problem**: Assuming imputation worked well without checking
   - **Solution**: Always validate distributions and relationships after imputation
   - **Impact**: Poor model performance due to distorted data

6. **Over-Imputing High Missing Percentages**
   - **Problem**: Imputing columns with >50% missing data
   - **Solution**: Consider dropping the feature or using missing indicators
   - **Impact**: Creates artificial patterns that don't exist in reality

7. **Ignoring Business Logic**
   - **Problem**: Using statistical methods that violate business rules
   - **Solution**: Incorporate domain knowledge into imputation strategy
   - **Impact**: Imputed values that are impossible or unrealistic

8. **Not Handling Different Data Types Appropriately**
   - **Problem**: Using numerical methods for categorical data
   - **Solution**: Use appropriate methods for each data type
   - **Impact**: Meaningless imputed values

## 📝 Summary

This comprehensive guide provides a systematic approach to handling missing data and imputation for machine learning projects. Here are the key takeaways:

### Essential Missing Data Handling Steps

1. **Analysis First** - Always understand your missing data patterns before imputing
2. **Choose Appropriate Methods** - Match imputation technique to data type and missingness mechanism
3. **Validate Quality** - Check that imputation preserves important data characteristics
4. **Document Decisions** - Record your approach for reproducibility and stakeholder communication

### Imputation Method Selection Guide

| Scenario | Recommended Approach |
|----------|---------------------|
| **< 5% missing, any type** | Simple imputation (mean/median/mode) |
| **5-20% missing, MCAR** | Simple imputation with validation |
| **5-20% missing, MAR** | KNN or MICE imputation |
| **20-50% missing** | Advanced methods + missing indicators |
| **> 50% missing** | Consider dropping or creating missing indicators |
| **Time series data** | Interpolation or temporal methods |
| **High-cardinality categorical** | "Unknown" category or frequency-based methods |

### Quality Validation Checklist

- ✅ Distribution preservation check
- ✅ Correlation structure maintained
- ✅ No introduction of impossible values
- ✅ Model performance validation
- ✅ Business logic compliance

### Next Steps After Imputation

1. **Feature Engineering** - Create new features based on cleaned data
2. **Feature Selection** - Remove redundant or low-value features
3. **Data Preprocessing** - Scale, encode, and transform features for modeling
4. **Model Development** - Build and validate your machine learning models
5. **Production Pipeline** - Implement imputation in your deployment pipeline

Remember that imputation is both an art and a science. While this guide provides systematic approaches and best practices, always consider your specific domain knowledge and use case requirements. The goal is not perfect imputation, but rather imputation that preserves the meaningful patterns in your data while enabling successful machine learning.
