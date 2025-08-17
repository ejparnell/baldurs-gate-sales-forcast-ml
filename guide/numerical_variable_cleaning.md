# 🔢 Complete Numerical Variable Cleaning Guide

This guide is ideal for data analysts, data scientists, and ML engineers working in Python who want a systematic, step-by-step approach to clean and prepare numerical variables for machine learning models.

This guide is a part of a larger series on Machine Learning Pipelines. Each section is designed to be modular, allowing you to adapt the code and techniques to your specific dataset and analysis needs.

**The guide before:** [Categorical Variable Cleaning](./categorical_variable_cleaning.md)  
**The guide after:** Feature Engineering and Selection (coming soon)

## 📋 Table of Contents

1. [📖 Overview](#-overview)
2. [🔢 Understanding Numerical Variables](#-understanding-numerical-variables)
3. [🔍 Numerical Data Quality Issues](#-numerical-data-quality-issues)
4. [🧹 Data Type Optimization](#-data-type-optimization)
5. [📊 Distribution Analysis and Normalization](#-distribution-analysis-and-normalization)
6. [🔄 Scaling and Transformation](#-scaling-and-transformation)
7. [💻 Code Examples and Implementation](#-code-examples-and-implementation)
8. [✅ Validation and Quality Checks](#-validation-and-quality-checks)
9. [⭐ Best Practices](#-best-practices)
10. [⚠️ Common Pitfalls](#️-common-pitfalls)
11. [📝 Summary](#-summary)

## 📖 Overview

Numerical variables form the backbone of most machine learning algorithms, yet they often require extensive preprocessing to achieve optimal model performance. Unlike categorical variables, numerical data can suffer from scale differences, distribution skewness, precision issues, and hidden patterns that significantly impact model training and prediction accuracy.

### Why Numerical Variable Cleaning Matters

**Data Quality Issues:**

- **Scale Differences**: Variables with vastly different ranges can dominate model learning
- **Distribution Skewness**: Non-normal distributions can violate algorithm assumptions
- **Precision Problems**: Floating-point errors and inappropriate data types waste memory
- **Hidden Patterns**: Temporal trends, seasonality, and correlation structures need proper handling

**Model Performance Impact:**

- **Convergence Issues**: Poorly scaled data can prevent optimization algorithms from converging
- **Feature Dominance**: Large-scale features can overshadow smaller but important variables
- **Algorithm Sensitivity**: Many algorithms assume normal distributions or specific data ranges
- **Computational Efficiency**: Proper data types and scaling improve training speed and memory usage

### Core Principles

1. **Understand Data Distribution**: Analyze the statistical properties of each numerical variable
2. **Preserve Information**: Maintain meaningful relationships while improving data quality
3. **Algorithm Awareness**: Consider the requirements of your chosen ML algorithms
4. **Scalability**: Ensure preprocessing steps work efficiently with large datasets
5. **Reproducibility**: Create consistent transformations for training and production data

## 🔢 Understanding Numerical Variables

### Types of Numerical Variables

**1. Continuous Variables**

- Can take any value within a range
- Examples: price, temperature, weight, time duration
- Considerations: Scale, distribution shape, outliers

**2. Discrete Variables**

- Countable, distinct values
- Examples: number of purchases, page views, inventory count
- Considerations: Zero-inflation, upper bounds, integer constraints

**3. Ratio Variables**

- Have meaningful zero point and ratios
- Examples: income, distance, age
- Transformations: Logarithmic, square root often appropriate

**4. Interval Variables**

- Equal intervals but arbitrary zero point
- Examples: temperature in Celsius, standardized test scores
- Transformations: Standardization preferred over ratio-based transforms

**5. Derived Variables**

- Created from other variables through calculations
- Examples: ratios, differences, aggregations
- Considerations: Correlation with source variables, interpretability

### Common Numerical Data Sources

- **Measurements**: Sensor data, physical measurements, financial metrics
- **Counts**: Event frequencies, transaction volumes, user interactions
- **Rates/Percentages**: Conversion rates, growth rates, percentages
- **Aggregations**: Sums, averages, medians from grouped data
- **Temporal Features**: Time-based calculations, rolling statistics

## 🔍 Numerical Data Quality Issues

### Comprehensive Numerical Data Profiler

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

class NumericalDataProfiler:
    def __init__(self, df):
        self.df = df.copy()
        self.numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.profile_results = {}
    
    def comprehensive_numerical_profile(self):
        """Generate comprehensive numerical data profile"""
        profile = {}
        
        for col in self.numerical_columns:
            col_data = self.df[col].dropna()
            
            if len(col_data) == 0:
                profile[col] = {'error': 'No non-null values'}
                continue
            
            # Basic statistics
            basic_stats = {
                'count': len(self.df[col]),
                'non_null_count': col_data.count(),
                'null_count': self.df[col].isnull().sum(),
                'null_percentage': (self.df[col].isnull().sum() / len(self.df)) * 100,
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'range': col_data.max() - col_data.min()
            }
            
            # Distribution analysis
            distribution_stats = self._analyze_distribution(col_data)
            
            # Data quality issues
            quality_issues = self._detect_quality_issues(col_data)
            
            # Scale analysis
            scale_analysis = self._analyze_scale_properties(col_data)
            
            profile[col] = {
                **basic_stats,
                **distribution_stats,
                **quality_issues,
                **scale_analysis
            }
        
        return profile
    
    def _analyze_distribution(self, series):
        """Analyze distribution properties"""
        try:
            skewness = stats.skew(series)
            kurtosis = stats.kurtosis(series)
            
            # Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(series.sample(min(5000, len(series)))) if len(series) > 0 else (0, 1)
            
            # Distribution classification
            distribution_type = self._classify_distribution(skewness, kurtosis)
            
            return {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'shapiro_stat': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05,
                'distribution_type': distribution_type
            }
        except:
            return {
                'skewness': np.nan,
                'kurtosis': np.nan,
                'shapiro_stat': np.nan,
                'shapiro_p_value': np.nan,
                'is_normal': False,
                'distribution_type': 'unknown'
            }
    
    def _classify_distribution(self, skewness, kurtosis):
        """Classify distribution based on skewness and kurtosis"""
        if abs(skewness) < 0.5 and abs(kurtosis) < 3:
            return 'normal'
        elif skewness > 1:
            return 'right_skewed'
        elif skewness < -1:
            return 'left_skewed'
        elif kurtosis > 3:
            return 'heavy_tailed'
        elif kurtosis < -1:
            return 'light_tailed'
        else:
            return 'slightly_non_normal'
    
    def _detect_quality_issues(self, series):
        """Detect common data quality issues"""
        issues = []
        
        # Infinite values
        inf_count = np.isinf(series).sum()
        if inf_count > 0:
            issues.append('infinite_values')
        
        # Negative values where they shouldn't be
        negative_count = (series < 0).sum()
        if negative_count > 0:
            issues.append('negative_values')
        
        # Potential data entry errors (too many repeated values)
        value_counts = series.value_counts()
        most_frequent_percentage = value_counts.iloc[0] / len(series) * 100 if len(value_counts) > 0 else 0
        if most_frequent_percentage > 50:
            issues.append('excessive_repetition')
        
        # Potential precision issues
        if series.dtype == 'float64':
            # Check for values that could be integers
            is_integer_like = (series == series.astype(int)).all()
            if is_integer_like:
                issues.append('integer_like_float')
        
        # Suspicious zeros
        zero_percentage = (series == 0).mean() * 100
        if zero_percentage > 20:
            issues.append('high_zero_percentage')
        
        return {
            'quality_issues': issues,
            'infinite_count': inf_count,
            'negative_count': negative_count,
            'zero_percentage': zero_percentage,
            'most_frequent_percentage': most_frequent_percentage
        }
    
    def _analyze_scale_properties(self, series):
        """Analyze scale and magnitude properties"""
        
        # Order of magnitude analysis
        abs_series = np.abs(series[series != 0])
        if len(abs_series) > 0:
            log_values = np.log10(abs_series)
            magnitude_range = log_values.max() - log_values.min()
            typical_magnitude = np.median(log_values)
        else:
            magnitude_range = 0
            typical_magnitude = 0
        
        # Data type efficiency
        current_dtype = series.dtype
        memory_usage = series.memory_usage(deep=True)
        
        # Suggest optimal data type
        optimal_dtype = self._suggest_optimal_dtype(series)
        
        return {
            'magnitude_range': magnitude_range,
            'typical_magnitude': typical_magnitude,
            'current_dtype': str(current_dtype),
            'memory_usage_bytes': memory_usage,
            'optimal_dtype': optimal_dtype,
            'scale_category': self._categorize_scale(magnitude_range)
        }
    
    def _suggest_optimal_dtype(self, series):
        """Suggest optimal data type for memory efficiency"""
        if series.dtype == 'object':
            return 'needs_conversion'
        
        # For integer-like data
        if series.dtype in ['int64', 'float64'] and (series == series.astype(int, errors='ignore')).all():
            min_val, max_val = series.min(), series.max()
            
            if min_val >= 0:  # Unsigned integers
                if max_val < 256:
                    return 'uint8'
                elif max_val < 65536:
                    return 'uint16'
                elif max_val < 4294967296:
                    return 'uint32'
                else:
                    return 'uint64'
            else:  # Signed integers
                if min_val >= -128 and max_val < 128:
                    return 'int8'
                elif min_val >= -32768 and max_val < 32768:
                    return 'int16'
                elif min_val >= -2147483648 and max_val < 2147483648:
                    return 'int32'
                else:
                    return 'int64'
        
        # For float data
        elif series.dtype == 'float64':
            # Check if float32 would be sufficient
            float32_series = series.astype('float32')
            if np.allclose(series.dropna(), float32_series.dropna(), equal_nan=True):
                return 'float32'
        
        return str(series.dtype)
    
    def _categorize_scale(self, magnitude_range):
        """Categorize the scale of the variable"""
        if magnitude_range < 2:
            return 'similar_scale'
        elif magnitude_range < 4:
            return 'moderate_scale_difference'
        else:
            return 'large_scale_difference'
    
    def visualize_numerical_distributions(self, columns=None, max_columns=6):
        """Create comprehensive distribution visualizations"""
        if columns is None:
            columns = self.numerical_columns[:max_columns]
        
        n_cols = min(len(columns), max_columns)
        fig, axes = plt.subplots(3, n_cols, figsize=(5*n_cols, 15))
        
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(columns[:n_cols]):
            col_data = self.df[col].dropna()
            
            # Histogram
            axes[0, i].hist(col_data, bins=50, alpha=0.7, edgecolor='black')
            axes[0, i].set_title(f'Distribution: {col}')
            axes[0, i].grid(True, alpha=0.3)
            
            # Box plot
            axes[1, i].boxplot(col_data)
            axes[1, i].set_title(f'Box Plot: {col}')
            axes[1, i].grid(True, alpha=0.3)
            
            # Q-Q plot for normality
            stats.probplot(col_data, dist="norm", plot=axes[2, i])
            axes[2, i].set_title(f'Q-Q Plot: {col}')
            axes[2, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def detect_scale_differences(self):
        """Detect variables with significantly different scales"""
        scale_info = {}
        
        for col in self.numerical_columns:
            col_data = self.df[col].dropna()
            if len(col_data) > 0:
                scale_info[col] = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'range': col_data.max() - col_data.min()
                }
        
        # Calculate coefficient of variation for scale comparison
        cv_values = {}
        ranges = {}
        
        for col, stats in scale_info.items():
            cv_values[col] = abs(stats['std'] / stats['mean']) if stats['mean'] != 0 else np.inf
            ranges[col] = stats['range']
        
        # Identify problematic scale differences
        max_range = max(ranges.values()) if ranges else 0
        min_range = min(ranges.values()) if ranges else 0
        range_ratio = max_range / min_range if min_range > 0 else np.inf
        
        recommendations = []
        if range_ratio > 1000:
            recommendations.append("Large scale differences detected - consider scaling")
        
        if any(cv > 2 for cv in cv_values.values()):
            recommendations.append("High coefficient of variation - consider robust scaling")
        
        return {
            'scale_info': scale_info,
            'range_ratio': range_ratio,
            'cv_values': cv_values,
            'recommendations': recommendations
        }
```

## 🧹 Data Type Optimization

### Memory Efficient Data Types

```python
class DataTypeOptimizer:
    def __init__(self):
        self.optimization_log = []
        self.original_memory = 0
        self.optimized_memory = 0
    
    def optimize_data_types(self, df, aggressive=False):
        """Optimize data types for memory efficiency"""
        optimized_df = df.copy()
        self.original_memory = df.memory_usage(deep=True).sum()
        
        optimization_details = {}
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'object']:
                original_dtype = df[col].dtype
                original_memory = df[col].memory_usage(deep=True)
                
                # Optimize the column
                optimized_series, new_dtype = self._optimize_column(df[col], aggressive)
                optimized_df[col] = optimized_series
                
                new_memory = optimized_df[col].memory_usage(deep=True)
                memory_saved = original_memory - new_memory
                
                optimization_details[col] = {
                    'original_dtype': str(original_dtype),
                    'new_dtype': str(new_dtype),
                    'memory_saved_bytes': memory_saved,
                    'memory_saved_mb': memory_saved / (1024 * 1024),
                    'reduction_percentage': (memory_saved / original_memory) * 100 if original_memory > 0 else 0
                }
        
        self.optimized_memory = optimized_df.memory_usage(deep=True).sum()
        total_savings = self.original_memory - self.optimized_memory
        
        self.optimization_log.append({
            'timestamp': pd.Timestamp.now(),
            'total_memory_saved_mb': total_savings / (1024 * 1024),
            'reduction_percentage': (total_savings / self.original_memory) * 100,
            'details': optimization_details
        })
        
        return optimized_df, optimization_details
    
    def _optimize_column(self, series, aggressive=False):
        """Optimize a single column's data type"""
        
        # Handle object columns that might be numeric
        if series.dtype == 'object':
            # Try to convert to numeric
            numeric_series = pd.to_numeric(series, errors='coerce')
            if not numeric_series.isna().all():
                # Successfully converted to numeric
                series = numeric_series
            else:
                # Keep as object but try categorical
                if aggressive and series.nunique() / len(series) < 0.1:
                    return series.astype('category'), 'category'
                return series, series.dtype
        
        # Optimize integer columns
        if np.issubdtype(series.dtype, np.integer):
            return self._optimize_integer(series), self._get_optimal_integer_dtype(series)
        
        # Optimize float columns
        elif np.issubdtype(series.dtype, np.floating):
            return self._optimize_float(series, aggressive), self._get_optimal_float_dtype(series, aggressive)
        
        return series, series.dtype
    
    def _optimize_integer(self, series):
        """Optimize integer data type"""
        min_val, max_val = series.min(), series.max()
        
        # Check if values fit in smaller integer types
        if min_val >= 0:  # Unsigned integers
            if max_val < 256:
                return series.astype('uint8')
            elif max_val < 65536:
                return series.astype('uint16')
            elif max_val < 4294967296:
                return series.astype('uint32')
            else:
                return series.astype('uint64')
        else:  # Signed integers
            if min_val >= -128 and max_val <= 127:
                return series.astype('int8')
            elif min_val >= -32768 and max_val <= 32767:
                return series.astype('int16')
            elif min_val >= -2147483648 and max_val <= 2147483647:
                return series.astype('int32')
            else:
                return series.astype('int64')
    
    def _optimize_float(self, series, aggressive=False):
        """Optimize float data type"""
        
        # Check if all values are actually integers
        if (series.dropna() == series.dropna().astype(int)).all():
            # Convert to integer
            return self._optimize_integer(series.astype(int))
        
        # Try float32 if values don't lose precision
        if aggressive:
            float32_series = series.astype('float32')
            if np.allclose(series.dropna(), float32_series.dropna(), equal_nan=True, rtol=1e-6):
                return float32_series
        
        return series
    
    def _get_optimal_integer_dtype(self, series):
        """Get optimal integer data type name"""
        min_val, max_val = series.min(), series.max()
        
        if min_val >= 0:
            if max_val < 256:
                return 'uint8'
            elif max_val < 65536:
                return 'uint16'
            elif max_val < 4294967296:
                return 'uint32'
            else:
                return 'uint64'
        else:
            if min_val >= -128 and max_val <= 127:
                return 'int8'
            elif min_val >= -32768 and max_val <= 32767:
                return 'int16'
            elif min_val >= -2147483648 and max_val <= 2147483647:
                return 'int32'
            else:
                return 'int64'
    
    def _get_optimal_float_dtype(self, series, aggressive=False):
        """Get optimal float data type name"""
        if (series.dropna() == series.dropna().astype(int)).all():
            return self._get_optimal_integer_dtype(series.astype(int))
        
        if aggressive:
            float32_series = series.astype('float32')
            if np.allclose(series.dropna(), float32_series.dropna(), equal_nan=True, rtol=1e-6):
                return 'float32'
        
        return 'float64'
    
    def generate_optimization_report(self):
        """Generate memory optimization report"""
        if not self.optimization_log:
            return "No optimization performed yet."
        
        latest_log = self.optimization_log[-1]
        
        report = f"""
        Data Type Optimization Report
        ============================
        Date: {latest_log['timestamp']}
        
        Memory Usage:
        - Original: {self.original_memory / (1024 * 1024):.2f} MB
        - Optimized: {self.optimized_memory / (1024 * 1024):.2f} MB
        - Saved: {latest_log['total_memory_saved_mb']:.2f} MB ({latest_log['reduction_percentage']:.1f}%)
        
        Column-wise Optimizations:
        """
        
        for col, details in latest_log['details'].items():
            if details['memory_saved_bytes'] > 0:
                report += f"\n  {col}:"
                report += f"\n    {details['original_dtype']} → {details['new_dtype']}"
                report += f"\n    Saved: {details['memory_saved_mb']:.2f} MB ({details['reduction_percentage']:.1f}%)"
        
        return report
```

## 📊 Distribution Analysis and Normalization

### Advanced Distribution Analysis

```python
from scipy import stats
from scipy.stats import normaltest, jarque_bera, anderson

class DistributionAnalyzer:
    def __init__(self):
        self.distribution_tests = {}
        self.transformation_recommendations = {}
    
    def comprehensive_distribution_analysis(self, df, numerical_columns=None):
        """Perform comprehensive distribution analysis"""
        
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        analysis_results = {}
        
        for col in numerical_columns:
            series = df[col].dropna()
            if len(series) < 3:
                continue
            
            # Basic distribution metrics
            basic_metrics = self._calculate_basic_metrics(series)
            
            # Normality tests
            normality_tests = self._perform_normality_tests(series)
            
            # Distribution fitting
            distribution_fit = self._fit_distributions(series)
            
            # Transformation recommendations
            transformation_rec = self._recommend_transformations(series)
            
            analysis_results[col] = {
                'basic_metrics': basic_metrics,
                'normality_tests': normality_tests,
                'distribution_fit': distribution_fit,
                'transformation_recommendations': transformation_rec
            }
        
        return analysis_results
    
    def _calculate_basic_metrics(self, series):
        """Calculate basic distribution metrics"""
        return {
            'mean': series.mean(),
            'median': series.median(),
            'mode': series.mode().iloc[0] if len(series.mode()) > 0 else np.nan,
            'std': series.std(),
            'variance': series.var(),
            'skewness': stats.skew(series),
            'kurtosis': stats.kurtosis(series),
            'range': series.max() - series.min(),
            'iqr': series.quantile(0.75) - series.quantile(0.25),
            'cv': series.std() / series.mean() if series.mean() != 0 else np.inf
        }
    
    def _perform_normality_tests(self, series):
        """Perform multiple normality tests"""
        sample_size = min(5000, len(series))
        sample = series.sample(sample_size) if len(series) > sample_size else series
        
        tests = {}
        
        # Shapiro-Wilk test (most powerful for small samples)
        if len(sample) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(sample)
            tests['shapiro'] = {'statistic': shapiro_stat, 'p_value': shapiro_p, 'is_normal': shapiro_p > 0.05}
        
        # D'Agostino and Pearson's test
        if len(sample) >= 20:
            dagostino_stat, dagostino_p = normaltest(sample)
            tests['dagostino'] = {'statistic': dagostino_stat, 'p_value': dagostino_p, 'is_normal': dagostino_p > 0.05}
        
        # Jarque-Bera test
        if len(sample) >= 2000:
            jb_stat, jb_p = jarque_bera(sample)
            tests['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_p, 'is_normal': jb_p > 0.05}
        
        # Anderson-Darling test
        if len(sample) >= 25:
            ad_stat, ad_critical, ad_significance = anderson(sample, dist='norm')
            # Use 5% significance level (index 2)
            tests['anderson'] = {
                'statistic': ad_stat, 
                'critical_value': ad_critical[2], 
                'is_normal': ad_stat < ad_critical[2]
            }
        
        # Consensus
        normal_votes = sum(1 for test in tests.values() if test.get('is_normal', False))
        total_tests = len(tests)
        tests['consensus'] = {
            'normal_votes': normal_votes,
            'total_tests': total_tests,
            'is_likely_normal': normal_votes >= total_tests * 0.5
        }
        
        return tests
    
    def _fit_distributions(self, series):
        """Fit common distributions and find best fit"""
        
        distributions_to_test = [
            stats.norm, stats.expon, stats.gamma, stats.beta, 
            stats.lognorm, stats.weibull_min, stats.chi2
        ]
        
        distribution_fits = {}
        aic_scores = {}
        
        for dist in distributions_to_test:
            try:
                # Fit distribution
                params = dist.fit(series)
                
                # Calculate AIC
                log_likelihood = np.sum(dist.logpdf(series, *params))
                k = len(params)  # number of parameters
                n = len(series)
                aic = 2 * k - 2 * log_likelihood
                
                distribution_fits[dist.name] = {
                    'parameters': params,
                    'aic': aic,
                    'log_likelihood': log_likelihood
                }
                aic_scores[dist.name] = aic
                
            except Exception:
                continue
        
        # Find best fitting distribution
        if aic_scores:
            best_distribution = min(aic_scores, key=aic_scores.get)
            distribution_fits['best_fit'] = {
                'distribution': best_distribution,
                'aic': aic_scores[best_distribution]
            }
        
        return distribution_fits
    
    def _recommend_transformations(self, series):
        """Recommend transformations based on distribution characteristics"""
        recommendations = []
        
        skewness = stats.skew(series)
        kurtosis = stats.kurtosis(series)
        has_negative = (series < 0).any()
        has_zeros = (series == 0).any()
        
        # Skewness-based recommendations
        if skewness > 1:  # Right-skewed
            if not has_negative and not has_zeros:
                recommendations.append('log_transform')
            elif not has_negative:
                recommendations.append('log1p_transform')
            recommendations.append('sqrt_transform')
            recommendations.append('box_cox')
        
        elif skewness < -1:  # Left-skewed
            recommendations.append('reflect_and_log')
            recommendations.append('yeo_johnson')
        
        # Kurtosis-based recommendations
        if abs(kurtosis) > 3:  # Heavy or light tails
            recommendations.append('robust_scaling')
            recommendations.append('quantile_transform')
        
        # General recommendations
        if abs(skewness) > 0.5 or abs(kurtosis) > 3:
            recommendations.append('power_transform')
            recommendations.append('quantile_transform')
        
        # Scale-based recommendations
        if series.std() > series.mean() * 10:  # High variance
            recommendations.append('standardization')
            recommendations.append('robust_scaling')
        
        return list(set(recommendations))  # Remove duplicates
    
    def visualize_distribution_analysis(self, df, column):
        """Create comprehensive distribution visualization"""
        series = df[column].dropna()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Distribution Analysis: {column}', fontsize=16)
        
        # Original distribution
        axes[0, 0].hist(series, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Original Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Log transformation (if applicable)
        if (series > 0).all():
            log_series = np.log(series)
            axes[0, 1].hist(log_series, bins=50, alpha=0.7, edgecolor='black', color='orange')
            axes[0, 1].set_title('Log Transformed')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Log transform\nnot applicable\n(non-positive values)', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Square root transformation
        if (series >= 0).all():
            sqrt_series = np.sqrt(series)
            axes[0, 2].hist(sqrt_series, bins=50, alpha=0.7, edgecolor='black', color='green')
            axes[0, 2].set_title('Square Root Transformed')
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'Square root transform\nnot applicable\n(negative values)', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
        
        # Q-Q plot
        stats.probplot(series, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1, 1].boxplot(series)
        axes[1, 1].set_title('Box Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Density plot with normal overlay
        axes[1, 2].hist(series, bins=50, density=True, alpha=0.7, edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = series.mean(), series.std()
        x = np.linspace(series.min(), series.max(), 100)
        axes[1, 2].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal PDF')
        axes[1, 2].set_title('Density with Normal Overlay')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
```

## 🔄 Scaling and Transformation

### Comprehensive Scaling and Transformation Pipeline

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class NumericalTransformer:
    def __init__(self):
        self.transformers = {}
        self.transformation_log = []
        self.fitted_transformers = {}
    
    def standard_scaling(self, df, columns):
        """Apply standard scaling (z-score normalization)"""
        scaled_df = df.copy()
        scaler = StandardScaler()
        
        scaled_df[columns] = scaler.fit_transform(df[columns])
        
        self.fitted_transformers['standard_scaler'] = scaler
        self.transformation_log.append({
            'method': 'standard_scaling',
            'columns': columns,
            'parameters': {'mean': scaler.mean_, 'scale': scaler.scale_}
        })
        
        return scaled_df
    
    def min_max_scaling(self, df, columns, feature_range=(0, 1)):
        """Apply min-max scaling"""
        scaled_df = df.copy()
        scaler = MinMaxScaler(feature_range=feature_range)
        
        scaled_df[columns] = scaler.fit_transform(df[columns])
        
        self.fitted_transformers['minmax_scaler'] = scaler
        self.transformation_log.append({
            'method': 'min_max_scaling',
            'columns': columns,
            'parameters': {'min': scaler.data_min_, 'max': scaler.data_max_}
        })
        
        return scaled_df
    
    def robust_scaling(self, df, columns):
        """Apply robust scaling (uses median and IQR)"""
        scaled_df = df.copy()
        scaler = RobustScaler()
        
        scaled_df[columns] = scaler.fit_transform(df[columns])
        
        self.fitted_transformers['robust_scaler'] = scaler
        self.transformation_log.append({
            'method': 'robust_scaling',
            'columns': columns,
            'parameters': {'center': scaler.center_, 'scale': scaler.scale_}
        })
        
        return scaled_df
    
    def power_transform(self, df, columns, method='yeo-johnson'):
        """Apply power transformation (Box-Cox or Yeo-Johnson)"""
        transformed_df = df.copy()
        transformer = PowerTransformer(method=method, standardize=True)
        
        transformed_df[columns] = transformer.fit_transform(df[columns])
        
        self.fitted_transformers['power_transformer'] = transformer
        self.transformation_log.append({
            'method': f'power_transform_{method}',
            'columns': columns,
            'parameters': {'lambdas': transformer.lambdas_}
        })
        
        return transformed_df
    
    def quantile_transform(self, df, columns, output_distribution='uniform'):
        """Apply quantile transformation"""
        transformed_df = df.copy()
        transformer = QuantileTransformer(output_distribution=output_distribution, 
                                        random_state=42)
        
        transformed_df[columns] = transformer.fit_transform(df[columns])
        
        self.fitted_transformers['quantile_transformer'] = transformer
        self.transformation_log.append({
            'method': f'quantile_transform_{output_distribution}',
            'columns': columns,
            'parameters': {'n_quantiles': transformer.n_quantiles_}
        })
        
        return transformed_df
    
    def log_transform(self, df, columns, add_constant=1):
        """Apply logarithmic transformation"""
        transformed_df = df.copy()
        
        for col in columns:
            if (df[col] <= 0).any():
                print(f"Warning: {col} contains non-positive values. Using log1p transformation.")
                transformed_df[col] = np.log1p(df[col])
            else:
                transformed_df[col] = np.log(df[col] + add_constant)
        
        self.transformation_log.append({
            'method': 'log_transform',
            'columns': columns,
            'parameters': {'add_constant': add_constant}
        })
        
        return transformed_df
    
    def sqrt_transform(self, df, columns):
        """Apply square root transformation"""
        transformed_df = df.copy()
        
        for col in columns:
            if (df[col] < 0).any():
                print(f"Warning: {col} contains negative values. Using signed square root.")
                transformed_df[col] = np.sign(df[col]) * np.sqrt(np.abs(df[col]))
            else:
                transformed_df[col] = np.sqrt(df[col])
        
        self.transformation_log.append({
            'method': 'sqrt_transform',
            'columns': columns,
            'parameters': {}
        })
        
        return transformed_df
    
    def apply_recommended_transformations(self, df, analysis_results):
        """Apply transformations based on distribution analysis"""
        transformed_df = df.copy()
        applied_transformations = {}
        
        for col, analysis in analysis_results.items():
            if col not in df.columns:
                continue
            
            recommendations = analysis.get('transformation_recommendations', [])
            
            if not recommendations:
                continue
            
            # Apply the first recommended transformation
            best_recommendation = recommendations[0]
            
            try:
                if best_recommendation == 'log_transform':
                    transformed_df = self.log_transform(transformed_df, [col])
                    applied_transformations[col] = 'log_transform'
                
                elif best_recommendation == 'sqrt_transform':
                    transformed_df = self.sqrt_transform(transformed_df, [col])
                    applied_transformations[col] = 'sqrt_transform'
                
                elif best_recommendation == 'box_cox':
                    transformed_df = self.power_transform(transformed_df, [col], method='box-cox')
                    applied_transformations[col] = 'box_cox'
                
                elif best_recommendation == 'yeo_johnson':
                    transformed_df = self.power_transform(transformed_df, [col], method='yeo-johnson')
                    applied_transformations[col] = 'yeo_johnson'
                
                elif best_recommendation == 'quantile_transform':
                    transformed_df = self.quantile_transform(transformed_df, [col])
                    applied_transformations[col] = 'quantile_transform'
                
                elif best_recommendation == 'standardization':
                    transformed_df = self.standard_scaling(transformed_df, [col])
                    applied_transformations[col] = 'standardization'
                
                elif best_recommendation == 'robust_scaling':
                    transformed_df = self.robust_scaling(transformed_df, [col])
                    applied_transformations[col] = 'robust_scaling'
                
            except Exception as e:
                print(f"Failed to apply {best_recommendation} to {col}: {str(e)}")
                continue
        
        return transformed_df, applied_transformations
    
    def compare_transformations(self, df, column):
        """Compare multiple transformations and their effect on normality"""
        series = df[column].dropna()
        
        transformations = {}
        
        # Original
        transformations['original'] = {
            'data': series,
            'skewness': stats.skew(series),
            'kurtosis': stats.kurtosis(series)
        }
        
        # Log transform (if applicable)
        if (series > 0).all():
            log_data = np.log(series)
            transformations['log'] = {
                'data': log_data,
                'skewness': stats.skew(log_data),
                'kurtosis': stats.kurtosis(log_data)
            }
        
        # Square root transform (if applicable)
        if (series >= 0).all():
            sqrt_data = np.sqrt(series)
            transformations['sqrt'] = {
                'data': sqrt_data,
                'skewness': stats.skew(sqrt_data),
                'kurtosis': stats.kurtosis(sqrt_data)
            }
        
        # Box-Cox transform (if applicable)
        if (series > 0).all():
            try:
                boxcox_data, _ = stats.boxcox(series)
                transformations['boxcox'] = {
                    'data': boxcox_data,
                    'skewness': stats.skew(boxcox_data),
                    'kurtosis': stats.kurtosis(boxcox_data)
                }
            except:
                pass
        
        # Yeo-Johnson transform
        try:
            transformer = PowerTransformer(method='yeo-johnson')
            yj_data = transformer.fit_transform(series.values.reshape(-1, 1)).flatten()
            transformations['yeo_johnson'] = {
                'data': yj_data,
                'skewness': stats.skew(yj_data),
                'kurtosis': stats.kurtosis(yj_data)
            }
        except:
            pass
        
        # Rank best transformations by normality (lowest abs(skewness) + abs(kurtosis))
        normality_scores = {}
        for name, transform in transformations.items():
            normality_scores[name] = abs(transform['skewness']) + abs(transform['kurtosis'])
        
        best_transformation = min(normality_scores, key=normality_scores.get)
        
        return transformations, best_transformation, normality_scores
```

## 💻 Code Examples and Implementation

### Complete Numerical Processing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class NumericalProcessingPipeline:
    def __init__(self):
        self.profiler = None
        self.optimizer = DataTypeOptimizer()
        self.analyzer = DistributionAnalyzer()
        self.transformer = NumericalTransformer()
        self.processing_history = []
    
    def process_numerical_data(self, df, numerical_columns=None, 
                             target_column=None,
                             optimize_dtypes=True,
                             handle_distributions=True,
                             scaling_strategy='auto'):
        """Complete numerical data processing pipeline"""
        
        processed_df = df.copy()
        
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column and target_column in numerical_columns:
                numerical_columns.remove(target_column)
        
        print("=== NUMERICAL DATA PROCESSING PIPELINE ===")
        print(f"Processing {len(numerical_columns)} numerical columns")
        
        # Step 1: Profile the data
        self.profiler = NumericalDataProfiler(processed_df)
        profile = self.profiler.comprehensive_numerical_profile()
        scale_analysis = self.profiler.detect_scale_differences()
        
        print(f"\nScale Analysis:")
        print(f"Range ratio: {scale_analysis['range_ratio']:.2f}")
        for rec in scale_analysis['recommendations']:
            print(f"- {rec}")
        
        # Step 2: Optimize data types
        if optimize_dtypes:
            print(f"\nOptimizing data types...")
            processed_df, optimization_details = self.optimizer.optimize_data_types(
                processed_df, aggressive=False
            )
            print(self.optimizer.generate_optimization_report())
        
        # Step 3: Handle distributions
        transformation_results = {}
        if handle_distributions:
            print(f"\nAnalyzing distributions...")
            distribution_analysis = self.analyzer.comprehensive_distribution_analysis(
                processed_df, numerical_columns
            )
            
            # Apply transformations
            processed_df, applied_transformations = self.transformer.apply_recommended_transformations(
                processed_df, distribution_analysis
            )
            transformation_results = applied_transformations
            
            print(f"Applied transformations: {applied_transformations}")
        
        # Step 4: Apply scaling
        scaling_results = self._apply_scaling_strategy(
            processed_df, numerical_columns, scaling_strategy, scale_analysis
        )
        processed_df = scaling_results['scaled_df']
        
        # Log processing results
        self.processing_history.append({
            'timestamp': pd.Timestamp.now(),
            'columns_processed': numerical_columns,
            'original_shape': df.shape,
            'final_shape': processed_df.shape,
            'optimization_details': optimization_details if optimize_dtypes else {},
            'transformation_results': transformation_results,
            'scaling_strategy': scaling_results['strategy_used'],
            'scale_analysis': scale_analysis
        })
        
        return processed_df
    
    def _apply_scaling_strategy(self, df, numerical_columns, strategy, scale_analysis):
        """Apply appropriate scaling strategy"""
        
        scaled_df = df.copy()
        strategy_used = strategy
        
        if strategy == 'auto':
            # Automatic strategy selection based on data characteristics
            if scale_analysis['range_ratio'] > 1000:
                # Large scale differences - use robust scaling
                scaled_df = self.transformer.robust_scaling(scaled_df, numerical_columns)
                strategy_used = 'robust_scaling'
            elif any(cv > 2 for cv in scale_analysis['cv_values'].values()):
                # High coefficient of variation - use standard scaling
                scaled_df = self.transformer.standard_scaling(scaled_df, numerical_columns)
                strategy_used = 'standard_scaling'
            elif scale_analysis['range_ratio'] > 100:
                # Moderate scale differences - use min-max scaling
                scaled_df = self.transformer.min_max_scaling(scaled_df, numerical_columns)
                strategy_used = 'min_max_scaling'
            else:
                # Similar scales - no scaling needed
                strategy_used = 'no_scaling'
        
        elif strategy == 'standard':
            scaled_df = self.transformer.standard_scaling(scaled_df, numerical_columns)
            strategy_used = 'standard_scaling'
        
        elif strategy == 'minmax':
            scaled_df = self.transformer.min_max_scaling(scaled_df, numerical_columns)
            strategy_used = 'min_max_scaling'
        
        elif strategy == 'robust':
            scaled_df = self.transformer.robust_scaling(scaled_df, numerical_columns)
            strategy_used = 'robust_scaling'
        
        elif strategy == 'none':
            strategy_used = 'no_scaling'
        
        return {
            'scaled_df': scaled_df,
            'strategy_used': strategy_used
        }
    
    def compare_preprocessing_strategies(self, df, numerical_columns, target_column):
        """Compare different preprocessing strategies"""
        
        strategies = {
            'no_preprocessing': df.copy(),
            'dtype_optimization': None,
            'distribution_handling': None,
            'scaling_only': None,
            'full_pipeline': None
        }
        
        # Strategy 1: Data type optimization only
        strategies['dtype_optimization'], _ = self.optimizer.optimize_data_types(df.copy())
        
        # Strategy 2: Distribution handling only
        distribution_analysis = self.analyzer.comprehensive_distribution_analysis(df, numerical_columns)
        strategies['distribution_handling'], _ = self.transformer.apply_recommended_transformations(
            df.copy(), distribution_analysis
        )
        
        # Strategy 3: Scaling only
        strategies['scaling_only'] = self.transformer.standard_scaling(df.copy(), numerical_columns)
        
        # Strategy 4: Full pipeline
        strategies['full_pipeline'] = self.process_numerical_data(
            df.copy(), numerical_columns, target_column
        )
        
        # Compare model performance
        if target_column in df.columns:
            performance_results = self._compare_model_performance(
                strategies, numerical_columns, target_column
            )
            return strategies, performance_results
        
        return strategies, None
    
    def _compare_model_performance(self, strategies, feature_columns, target_column):
        """Compare model performance across different preprocessing strategies"""
        
        results = {}
        
        for strategy_name, df in strategies.items():
            if df is None:
                continue
            
            try:
                # Prepare data
                X = df[feature_columns]
                y = df[target_column]
                
                # Handle any remaining NaN values
                X = X.fillna(X.mean())
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train simple model
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[strategy_name] = {
                    'mse': mse,
                    'r2': r2,
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
                }
                
            except Exception as e:
                results[strategy_name] = {'error': str(e)}
        
        return results
    
    def generate_processing_report(self):
        """Generate comprehensive processing report"""
        
        if not self.processing_history:
            return "No processing history available."
        
        latest_processing = self.processing_history[-1]
        
        report = f"""
        Numerical Data Processing Report
        ===============================
        Processing Date: {latest_processing['timestamp']}
        
        Data Shape Changes:
        - Original: {latest_processing['original_shape']}
        - Final: {latest_processing['final_shape']}
        
        Columns Processed: {len(latest_processing['columns_processed'])}
        Scaling Strategy: {latest_processing['scaling_strategy']}
        
        Scale Analysis:
        - Range Ratio: {latest_processing['scale_analysis']['range_ratio']:.2f}
        - Recommendations: {latest_processing['scale_analysis']['recommendations']}
        
        Applied Transformations:
        """
        
        for col, transformation in latest_processing['transformation_results'].items():
            report += f"\n  - {col}: {transformation}"
        
        if latest_processing['optimization_details']:
            total_memory_saved = sum(
                details['memory_saved_mb'] 
                for details in latest_processing['optimization_details'].values()
            )
            report += f"\n\nMemory Optimization: {total_memory_saved:.2f} MB saved"
        
        return report

# Example usage with comprehensive demonstration
def run_numerical_processing_example():
    """Comprehensive example of numerical processing pipeline"""
    
    # Create sample data with various numerical challenges
    np.random.seed(42)
    n_samples = 1000
    
    # Create data with different characteristics
    numerical_data = {
        'normal_feature': np.random.normal(50, 15, n_samples),
        'skewed_feature': np.random.exponential(2, n_samples),
        'large_scale_feature': np.random.normal(100000, 25000, n_samples),
        'small_scale_feature': np.random.normal(0.001, 0.0002, n_samples),
        'integer_like_float': np.random.randint(1, 100, n_samples).astype(float),
        'discrete_count': np.random.poisson(5, n_samples),
        'target': np.random.normal(100, 25, n_samples)
    }
    
    df = pd.DataFrame(numerical_data)
    
    # Add some data quality issues
    # Introduce some infinite values
    df.loc[df.index[:10], 'large_scale_feature'] = np.inf
    
    # Add some zeros to create sparsity
    zero_indices = np.random.choice(df.index, 100, replace=False)
    df.loc[zero_indices, 'skewed_feature'] = 0
    
    # Initialize pipeline
    pipeline = NumericalProcessingPipeline()
    
    print("Original Data Summary:")
    print(df.describe())
    print(f"\nOriginal Memory Usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    # Process numerical data
    feature_columns = ['normal_feature', 'skewed_feature', 'large_scale_feature', 
                      'small_scale_feature', 'integer_like_float', 'discrete_count']
    
    processed_df = pipeline.process_numerical_data(
        df,
        numerical_columns=feature_columns,
        target_column='target',
        optimize_dtypes=True,
        handle_distributions=True,
        scaling_strategy='auto'
    )
    
    print(f"\nProcessed Data Summary:")
    print(processed_df[feature_columns].describe())
    print(f"\nProcessed Memory Usage: {processed_df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    # Generate comprehensive report
    print("\n" + pipeline.generate_processing_report())
    
    # Compare strategies
    print(f"\nComparing preprocessing strategies...")
    strategies, performance = pipeline.compare_preprocessing_strategies(
        df, feature_columns, 'target'
    )
    
    if performance:
        print(f"\nModel Performance Comparison:")
        for strategy, metrics in performance.items():
            if 'error' not in metrics:
                print(f"  {strategy}:")
                print(f"    R²: {metrics['r2']:.4f}")
                print(f"    MSE: {metrics['mse']:.4f}")
                print(f"    Memory: {metrics['memory_usage_mb']:.2f} MB")
    
    return processed_df, strategies, performance

# Visualization helper
def visualize_preprocessing_impact(original_df, processed_df, columns):
    """Visualize the impact of preprocessing on data distributions"""
    
    n_cols = len(columns)
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
    
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, col in enumerate(columns):
        # Original distribution
        axes[0, i].hist(original_df[col].dropna(), bins=50, alpha=0.7, 
                       color='red', label='Original')
        axes[0, i].set_title(f'Original: {col}')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Processed distribution
        axes[1, i].hist(processed_df[col].dropna(), bins=50, alpha=0.7, 
                       color='blue', label='Processed')
        axes[1, i].set_title(f'Processed: {col}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run the comprehensive example
# processed_data, strategies, performance = run_numerical_processing_example()
```

## ✅ Validation and Quality Checks

### Numerical Data Validation Framework

```python
class NumericalDataValidator:
    def __init__(self, original_df, processed_df):
        self.original_df = original_df
        self.processed_df = processed_df
        self.validation_results = {}
    
    def validate_data_integrity(self, numerical_columns):
        """Comprehensive validation of numerical data processing"""
        
        validation_results = {}
        
        for col in numerical_columns:
            if col not in self.original_df.columns or col not in self.processed_df.columns:
                continue
            
            original_series = self.original_df[col].dropna()
            processed_series = self.processed_df[col].dropna()
            
            # Basic integrity checks
            integrity_checks = self._check_basic_integrity(original_series, processed_series)
            
            # Distribution changes
            distribution_changes = self._analyze_distribution_changes(original_series, processed_series)
            
            # Scale and range changes
            scale_changes = self._analyze_scale_changes(original_series, processed_series)
            
            # Outlier handling validation
            outlier_validation = self._validate_outlier_handling(original_series, processed_series)
            
            validation_results[col] = {
                'integrity_checks': integrity_checks,
                'distribution_changes': distribution_changes,
                'scale_changes': scale_changes,
                'outlier_validation': outlier_validation
            }
        
        return validation_results
    
    def _check_basic_integrity(self, original, processed):
        """Check basic data integrity"""
        
        return {
            'data_count_preserved': len(original) == len(processed),
            'no_new_nulls': not processed.isnull().any(),
            'no_infinite_values': not np.isinf(processed).any(),
            'finite_values_only': np.isfinite(processed).all(),
            'original_null_count': original.isnull().sum(),
            'processed_null_count': processed.isnull().sum()
        }
    
    def _analyze_distribution_changes(self, original, processed):
        """Analyze how distributions changed"""
        
        try:
            original_skew = stats.skew(original)
            processed_skew = stats.skew(processed)
            
            original_kurtosis = stats.kurtosis(original)
            processed_kurtosis = stats.kurtosis(processed)
            
            # Normality improvement
            _, original_normal_p = stats.normaltest(original)
            _, processed_normal_p = stats.normaltest(processed)
            
            return {
                'skewness_change': processed_skew - original_skew,
                'kurtosis_change': processed_kurtosis - original_kurtosis,
                'normality_improved': processed_normal_p > original_normal_p,
                'original_normal_p': original_normal_p,
                'processed_normal_p': processed_normal_p,
                'distribution_more_normal': abs(processed_skew) < abs(original_skew)
            }
        except:
            return {'error': 'Could not analyze distribution changes'}
    
    def _analyze_scale_changes(self, original, processed):
        """Analyze scale and range changes"""
        
        return {
            'original_range': original.max() - original.min(),
            'processed_range': processed.max() - processed.min(),
            'original_std': original.std(),
            'processed_std': processed.std(),
            'scale_factor': processed.std() / original.std() if original.std() != 0 else np.inf,
            'mean_preserved': abs(original.mean() - processed.mean()) < 1e-10,
            'relative_scale_change': (processed.std() - original.std()) / original.std() if original.std() != 0 else np.inf
        }
    
    def _validate_outlier_handling(self, original, processed):
        """Validate outlier handling"""
        
        # Calculate outliers using IQR method
        def count_outliers(series):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return ((series < lower_bound) | (series > upper_bound)).sum()
        
        original_outliers = count_outliers(original)
        processed_outliers = count_outliers(processed)
        
        return {
            'original_outlier_count': original_outliers,
            'processed_outlier_count': processed_outliers,
            'outliers_reduced': processed_outliers < original_outliers,
            'outlier_reduction': original_outliers - processed_outliers,
            'outlier_reduction_percentage': ((original_outliers - processed_outliers) / original_outliers * 100) if original_outliers > 0 else 0
        }
    
    def detect_processing_anomalies(self):
        """Detect potential issues with numerical processing"""
        
        anomalies = []
        
        numerical_cols = self.original_df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col not in self.processed_df.columns:
                anomalies.append(f"Column {col} missing from processed data")
                continue
            
            original = self.original_df[col].dropna()
            processed = self.processed_df[col].dropna()
            
            if len(original) == 0 or len(processed) == 0:
                continue
            
            # Check for extreme transformations
            if abs(processed.std() - original.std()) > original.std() * 10:
                anomalies.append(f"Extreme scale change in {col}")
            
            # Check for loss of variance
            if processed.std() < original.std() * 0.01:
                anomalies.append(f"Severe loss of variance in {col}")
            
            # Check for constant values
            if processed.nunique() == 1:
                anomalies.append(f"Column {col} became constant after processing")
            
            # Check for extreme values
            if np.isinf(processed).any():
                anomalies.append(f"Infinite values introduced in {col}")
            
            if processed.isna().sum() > original.isna().sum():
                anomalies.append(f"New missing values introduced in {col}")
        
        return anomalies
    
    def generate_validation_report(self, numerical_columns):
        """Generate comprehensive validation report"""
        
        validation_results = self.validate_data_integrity(numerical_columns)
        anomalies = self.detect_processing_anomalies()
        
        report = """
        Numerical Data Validation Report
        ===============================
        
        """
        
        # Overall summary
        total_columns = len(validation_results)
        successful_columns = sum(
            1 for result in validation_results.values() 
            if result['integrity_checks']['finite_values_only']
        )
        
        report += f"Columns Validated: {successful_columns}/{total_columns}\n"
        
        # Anomalies
        if anomalies:
            report += f"\nAnomalies Detected:\n"
            for anomaly in anomalies:
                report += f"  ⚠️  {anomaly}\n"
        else:
            report += f"\n✅ No anomalies detected\n"
        
        # Column-by-column results
        report += f"\nColumn-wise Validation:\n"
        for col, results in validation_results.items():
            integrity = results['integrity_checks']
            distribution = results['distribution_changes']
            
            report += f"\n  {col}:"
            report += f"\n    Data Integrity: {'✅' if integrity['finite_values_only'] else '❌'}"
            
            if 'distribution_more_normal' in distribution:
                report += f"\n    Distribution Improved: {'✅' if distribution['distribution_more_normal'] else '➖'}"
            
            if 'outliers_reduced' in results['outlier_validation']:
                outlier_result = results['outlier_validation']['outliers_reduced']
                report += f"\n    Outliers Handled: {'✅' if outlier_result else '➖'}"
        
        return report

    def create_validation_visualizations(self, columns):
        """Create visualizations for validation"""
        
        n_cols = len(columns)
        fig, axes = plt.subplots(3, n_cols, figsize=(5*n_cols, 15))
        
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(columns):
            if col not in self.original_df.columns or col not in self.processed_df.columns:
                continue
            
            original = self.original_df[col].dropna()
            processed = self.processed_df[col].dropna()
            
            # Distribution comparison
            axes[0, i].hist(original, bins=50, alpha=0.7, label='Original', color='red')
            axes[0, i].hist(processed, bins=50, alpha=0.7, label='Processed', color='blue')
            axes[0, i].set_title(f'Distribution: {col}')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Q-Q plots
            stats.probplot(original, dist="norm", plot=axes[1, i])
            axes[1, i].set_title(f'Q-Q Original: {col}')
            axes[1, i].grid(True, alpha=0.3)
            
            stats.probplot(processed, dist="norm", plot=axes[2, i])
            axes[2, i].set_title(f'Q-Q Processed: {col}')
            axes[2, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
```

## ⭐ Best Practices

### 1. **Systematic Data Type Optimization**

```python
def optimize_data_types_systematically(df):
    """Best practice approach to data type optimization"""
    
    optimized_df = df.copy()
    memory_savings = {}
    
    for col in df.columns:
        original_memory = df[col].memory_usage(deep=True)
        
        if df[col].dtype == 'object':
            # Try to convert to numeric first
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if not numeric_series.isna().all():
                optimized_df[col] = numeric_series
            else:
                # Consider categorical for low cardinality
                if df[col].nunique() / len(df[col]) < 0.05:
                    optimized_df[col] = df[col].astype('category')
        
        elif df[col].dtype == 'int64':
            # Downcast integers
            optimized_df[col] = pd.to_numeric(df[col], downcast='integer')
        
        elif df[col].dtype == 'float64':
            # Downcast floats
            optimized_df[col] = pd.to_numeric(df[col], downcast='float')
        
        new_memory = optimized_df[col].memory_usage(deep=True)
        memory_savings[col] = original_memory - new_memory
    
    total_savings = sum(memory_savings.values())
    print(f"Total memory saved: {total_savings / (1024*1024):.2f} MB")
    
    return optimized_df
```

### 2. **Robust Scaling Strategy Selection**

```python
def choose_scaling_strategy(df, numerical_columns, algorithm_type='tree_based'):
    """Choose optimal scaling strategy based on data and algorithm"""
    
    # Analyze data characteristics
    scale_ratios = []
    distributions = []
    
    for col in numerical_columns:
        series = df[col].dropna()
        if len(series) > 0:
            scale_ratios.append(series.max() / series.min() if series.min() != 0 else np.inf)
            distributions.append(abs(stats.skew(series)))
    
    max_scale_ratio = max(scale_ratios) if scale_ratios else 1
    avg_skewness = np.mean(distributions) if distributions else 0
    
    # Decision logic
    if algorithm_type in ['tree_based', 'ensemble']:
        # Tree-based algorithms are scale-invariant
        return 'none'
    
    elif algorithm_type in ['linear', 'svm', 'neural_network']:
        if max_scale_ratio > 1000 or avg_skewness > 2:
            return 'robust'  # Robust to outliers and skewness
        elif max_scale_ratio > 100:
            return 'standard'  # Good for moderate differences
        else:
            return 'minmax'  # Good for bounded ranges
    
    elif algorithm_type == 'distance_based':
        return 'standard'  # Essential for KNN, clustering
    
    else:
        return 'standard'  # Safe default

# Usage example
scaling_strategy = choose_scaling_strategy(df, numerical_cols, 'neural_network')
print(f"Recommended scaling: {scaling_strategy}")
```

### 3. **Production-Safe Transformation Pipeline**

```python
class ProductionNumericalProcessor:
    def __init__(self):
        self.fitted_transformers = {}
        self.transformation_params = {}
        self.is_fitted = False
    
    def fit(self, df, numerical_columns):
        """Fit transformers on training data"""
        
        for col in numerical_columns:
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            
            # Store statistics for validation
            self.transformation_params[col] = {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'skewness': stats.skew(series)
            }
            
            # Fit appropriate transformer
            if abs(stats.skew(series)) > 1 and (series > 0).all():
                # Use log transform for skewed positive data
                transformer = PowerTransformer(method='box-cox')
                self.fitted_transformers[col] = transformer.fit(series.values.reshape(-1, 1))
            else:
                # Use standard scaling
                transformer = StandardScaler()
                self.fitted_transformers[col] = transformer.fit(series.values.reshape(-1, 1))
        
        self.is_fitted = True
    
    def transform(self, df):
        """Transform new data using fitted transformers"""
        
        if not self.is_fitted:
            raise ValueError("Must fit transformers before transform")
        
        transformed_df = df.copy()
        
        for col, transformer in self.fitted_transformers.items():
            if col not in df.columns:
                continue
            
            # Validate data characteristics
            series = df[col].dropna()
            if len(series) > 0:
                self._validate_data_drift(col, series)
            
            # Apply transformation
            transformed_values = transformer.transform(df[[col]])
            transformed_df[col] = transformed_values.flatten()
        
        return transformed_df
    
    def _validate_data_drift(self, column, series):
        """Detect significant data drift"""
        
        original_params = self.transformation_params[column]
        
        current_mean = series.mean()
        current_std = series.std()
        
        # Check for significant drift (more than 2 standard deviations)
        mean_drift = abs(current_mean - original_params['mean']) / original_params['std']
        std_drift = abs(current_std - original_params['std']) / original_params['std']
        
        if mean_drift > 2:
            print(f"Warning: Significant mean drift detected in {column}")
        
        if std_drift > 0.5:
            print(f"Warning: Significant variance drift detected in {column}")
    
    def fit_transform(self, df, numerical_columns):
        """Fit and transform in one step"""
        self.fit(df, numerical_columns)
        return self.transform(df)
```

### 4. **Cross-Validation Aware Processing**

```python
from sklearn.model_selection import KFold
from sklearn.base import clone

def cv_aware_numerical_processing(df, numerical_columns, target_column, cv_folds=5):
    """Apply numerical processing with cross-validation to prevent leakage"""
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    processed_dfs = []
    
    for train_idx, val_idx in kf.split(df):
        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]
        
        # Fit transformers on training data only
        processor = ProductionNumericalProcessor()
        processor.fit(train_data, numerical_columns)
        
        # Transform both training and validation data
        train_transformed = processor.transform(train_data)
        val_transformed = processor.transform(val_data)
        
        # Store with original indices
        train_transformed.index = train_idx
        val_transformed.index = val_idx
        
        processed_dfs.extend([train_transformed, val_transformed])
    
    # Combine all folds
    final_df = pd.concat(processed_dfs).sort_index()
    
    return final_df
```

## ⚠️ Common Pitfalls

### 1. **Scaling Before Train/Test Split**

```python
# ❌ Wrong: Scaling entire dataset before splitting
def bad_scaling_practice(df, numerical_columns, target_column):
    # This causes data leakage!
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    X = df[numerical_columns]
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

# ✅ Correct: Fit on training data, transform both
def proper_scaling_practice(df, numerical_columns, target_column):
    X = df[numerical_columns]
    y = df[target_column]
    
    # Split first
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use fitted scaler
    
    return X_train_scaled, X_test_scaled, y_train, y_test
```

### 2. **Ignoring Data Distribution When Choosing Transformations**

```python
# ❌ Wrong: Applying log transform without checking assumptions
def bad_log_transform(df, columns):
    for col in columns:
        df[col] = np.log(df[col])  # Fails if any values <= 0
    return df

# ✅ Better: Check assumptions before transformation
def smart_log_transform(df, columns):
    transformed_df = df.copy()
    
    for col in columns:
        series = df[col]
        
        # Check if log transform is appropriate
        if (series <= 0).any():
            print(f"Warning: {col} contains non-positive values. Using log1p instead.")
            transformed_df[col] = np.log1p(series)
        elif stats.skew(series) < 0.5:
            print(f"Info: {col} is not highly skewed. Log transform may not be beneficial.")
            # Apply anyway but warn user
            transformed_df[col] = np.log(series)
        else:
            transformed_df[col] = np.log(series)
    
    return transformed_df
```

### 3. **Not Handling Infinite and Missing Values**

```python
# ❌ Wrong: Not checking for infinite values
def bad_data_cleaning(df):
    return df.fillna(df.mean())  # Ignores infinite values

# ✅ Correct: Comprehensive data cleaning
def robust_data_cleaning(df, numerical_columns):
    cleaned_df = df.copy()
    
    for col in numerical_columns:
        # Handle infinite values
        inf_mask = np.isinf(cleaned_df[col])
        if inf_mask.any():
            print(f"Warning: {inf_mask.sum()} infinite values found in {col}")
            # Replace with NaN for consistent handling
            cleaned_df.loc[inf_mask, col] = np.nan
        
        # Handle missing values with appropriate strategy
        if cleaned_df[col].isnull().any():
            if cleaned_df[col].dtype in ['int64', 'float64']:
                # Use median for numerical data (more robust)
                fill_value = cleaned_df[col].median()
                cleaned_df[col].fillna(fill_value, inplace=True)
    
    return cleaned_df
```

### 4. **Inappropriate Data Type Choices**

```python
# ❌ Wrong: Using float64 for everything
def wasteful_data_types(df):
    # Wastes memory
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ✅ Better: Choose appropriate data types
def efficient_data_types(df):
    optimized_df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            
            if not numeric_series.isna().all():
                # Successfully converted - now optimize
                if (numeric_series == numeric_series.astype(int)).all():
                    # Integer-like values
                    min_val, max_val = numeric_series.min(), numeric_series.max()
                    
                    if min_val >= 0 and max_val < 256:
                        optimized_df[col] = numeric_series.astype('uint8')
                    elif min_val >= -128 and max_val < 128:
                        optimized_df[col] = numeric_series.astype('int8')
                    # ... continue with other integer types
                else:
                    # Float values - check if float32 is sufficient
                    float32_series = numeric_series.astype('float32')
                    if np.allclose(numeric_series, float32_series, equal_nan=True):
                        optimized_df[col] = float32_series
                    else:
                        optimized_df[col] = numeric_series
            else:
                # Keep as object or convert to category
                if df[col].nunique() / len(df[col]) < 0.05:
                    optimized_df[col] = df[col].astype('category')
    
    return optimized_df
```

## 📝 Summary

Effective numerical variable cleaning is fundamental to successful machine learning projects. This comprehensive guide covered all essential aspects of numerical data preprocessing:

### Key Takeaways

1. **Understand Your Data**: Profile numerical variables to identify distribution patterns, scale differences, and quality issues

2. **Optimize Memory Usage**: Choose appropriate data types to reduce memory consumption and improve processing speed

3. **Handle Distributions Thoughtfully**: Apply transformations based on statistical analysis, not assumptions

4. **Scale Appropriately**: Select scaling strategies based on algorithm requirements and data characteristics

5. **Validate Transformations**: Always check that preprocessing improves rather than degrades data quality

### Preprocessing Strategy Guidelines

- **Tree-based algorithms**: Minimal scaling needed, focus on data quality
- **Linear algorithms**: Standard scaling or robust scaling for outlier-prone data
- **Neural networks**: StandardScaler or MinMaxScaler depending on activation functions
- **Distance-based algorithms**: Scaling is critical for performance

### Production Considerations

- **Avoid Data Leakage**: Always fit transformers on training data only
- **Handle Drift**: Monitor for significant changes in data characteristics
- **Preserve Relationships**: Ensure transformations don't break meaningful correlations
- **Document Decisions**: Record transformation rationale for reproducibility

### Next Steps

With clean, properly scaled numerical variables, you're ready for:

- **Feature Engineering**: Creating new numerical features through combinations and calculations
- **Feature Selection**: Identifying the most predictive numerical variables
- **Model Training**: Building models with robust numerical representations
- **Performance Monitoring**: Tracking numerical feature drift in production

Remember: Numerical preprocessing is often iterative. Start with simple transformations and add complexity only when needed. Always validate that each step improves your model's performance on held-out data.

---

**Related Guides in This Series:**

- [Exploratory Data Analysis](./exploratory_data_analysis.md)
- [Missing Data Imputation](./missing_data_imputation.md)
- [Duplication and Outlier Handling](./duplication_outlier_handling.md)
- [Categorical Variable Cleaning](./categorical_variable_cleaning.md)
- Feature Engineering and Selection (coming soon)
- Model Selection and Validation (coming soon)