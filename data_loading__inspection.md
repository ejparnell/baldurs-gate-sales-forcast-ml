# Complete Data Cleaning Guide: Loading and Initial Inspection

A comprehensive guide for data loading and initial inspection using Python and Pandas.

## Table of Contents

1. [Data Loading Strategies](#data-loading-strategies)
2. [Initial Data Inspection Framework](#initial-data-inspection-framework)
3. [Best Practices and Real-World Considerations](#best-practices-and-real-world-considerations)
4. [Professional Workflow Guidelines](#professional-workflow-guidelines)

## Data Loading Strategies

### Data Source Assessment

Before writing any code, conduct a systematic assessment of your data source:

#### File Format Analysis

Understanding your data format is crucial for selecting the right loading approach:

- **CSV**: Universal format, lightweight, but prone to type inference issues
  - File extension: `.csv`
- **JSON**: Excellent for nested data, but memory intensive for large datasets
  - File extension: `.json`
- **Parquet**: Compressed and typed, requires specific libraries but offers superior performance
  - File extension: `.parquet`
- **Excel**: Supports multiple sheets, but slow and memory heavy for large files
  - File extension: `.xlsx`
- **SQLite**: Relational structure with query capabilities, requires SQL knowledge
  - File extension: `.sqlite`
- **PostgreSQL/MySQL**: Enterprise features and scalability, but complex connection setup
  - File extension: `.sql`

**Note**: `.db` files are typically database files and can be handled with SQLite or similar libraries.

When you are looking for data sometimes you don't have a choice of format. That's okay, once it's in a dataframe it's just data.

#### Size and Complexity Estimation

Choose your loading strategy based on data volume:

- **Small Data** (< 100MB): Load entirely into memory for interactive analysis
- **Medium Data** (100MB - 5GB): Use chunking, sampling, or selective loading strategies
- **Large Data** (> 5GB): Consider distributed processing (Dask), database querying, or cloud solutions

<details>
<summary>Example of small data loading</summary>

```python
import pandas as pd
# Load small CSV file into a DataFrame
df = pd.read_csv('data/small_data.csv')
```

</details>

<details>
<summary>Example of medium data loading</summary>

```python
import pandas as pd
# Load medium CSV file in chunks
chunk_size = 100000  # Adjust based on memory capacity
chunks = pd.read_csv('data/medium_data.csv', chunksize=chunk_size)
df = pd.concat(chunks, ignore_index=True)
```

</details>

<details>
<summary>Example of large data loading</summary>

```python
import dask.dataframe as dd
# Load large Parquet file using Dask for distributed processing
df = dd.read_parquet('data/large_data.parquet')
```

</details>

### Systematic Loading Approach

#### Robust Data Loading Principles

1. **Error Handling**: Always implement try-catch blocks and validate data integrity
2. **Format Auto-Detection**: Detect file formats automatically when possible
3. **Metadata Capture**: Store information about data shape, columns, and types
4. **Connection Management**: Properly open and close database connections
5. **Memory Efficiency**: Monitor and optimize memory usage during loading

<details>
<summary>Example of robust loading function</summary>

```python
import os
import pandas as pd
import psutil

def detect_file_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return 'csv'
    elif ext == '.json':
        return 'json'
    elif ext == '.parquet':
        return 'parquet'
    else:
        return None

def get_metadata(df):
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict()
    }

def load_data(file_path, **kwargs):
    file_type = detect_file_type(file_path)
    if not file_type:
        print(f"Unsupported file type for: {file_path}")
        return None, None

    try:
        # Monitor memory before loading
        mem_before = psutil.Process().memory_info().rss
        if file_type == 'csv':
            df = pd.read_csv(file_path, **kwargs)
        elif file_type == 'json':
            df = pd.read_json(file_path, **kwargs)
        elif file_type == 'parquet':
            df = pd.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Validate data integrity
        if df.empty:
            print("Warning: Loaded dataframe is empty.")
        metadata = get_metadata(df)

        # Monitor memory after loading
        mem_after = psutil.Process().memory_info().rss
        print(f"Memory used for loading: {(mem_after - mem_before)/1024**2:.2f} MB")

        return df, metadata
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

# Example for database connection management
import sqlite3

def load_from_sqlite(db_path, query):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        metadata = get_metadata(df)
        return df, metadata
    except Exception as e:
        print(f"Database error: {e}")
        return None, None
    finally:
        if conn:
            conn.close()
```

</details>

#### Multi-Table Database Loading

For relational databases, implement systematic approaches:

1. **Schema Discovery**: Automatically identify all available tables and views
2. **Relationship Mapping**: Detect potential foreign key relationships
3. **Selective Loading**: Load only required tables based on analysis needs
4. **Metadata Collection**: Capture primary keys, nullable columns, and data types
5. **Batch Processing**: Load large tables in manageable chunks to avoid memory overflow

<details>
<summary>Example of multi-table database loading</summary>

```python
import sqlite3
import pandas as pd

def discover_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # List all tables and views
    cursor.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')")
    schema = cursor.fetchall()
    conn.close()
    return schema

def get_table_metadata(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Get column info
    cursor.execute(f"PRAGMA table_info('{table_name}')")
    columns = cursor.fetchall()
    # Find primary key columns
    pk_cols = [col[1] for col in columns if col[5] == 1]
    # Find nullable columns
    nullable_cols = [col[1] for col in columns if col[3] == 0]
    # Get foreign key info
    cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
    fks = cursor.fetchall()
    conn.close()
    return {
        "columns": [col[1] for col in columns],
        "dtypes": [col[2] for col in columns],
        "primary_keys": pk_cols,
        "nullable": nullable_cols,
        "foreign_keys": fks
    }

def load_table_in_batches(db_path, table_name, batch_size=10000):
    conn = sqlite3.connect(db_path)
    query = f"SELECT COUNT(*) FROM {table_name}"
    total_rows = conn.execute(query).fetchone()[0]
    for offset in range(0, total_rows, batch_size):
        batch_query = f"SELECT * FROM {table_name} LIMIT {batch_size} OFFSET {offset}"
        df = pd.read_sql_query(batch_query, conn)
        yield df
    conn.close()

# Example usage:
db_path = "your_database.sqlite"
schema = discover_schema(db_path)
for name, obj_type in schema:
    metadata = get_table_metadata(db_path, name)
    print(f"Table/View: {name}, Metadata: {metadata}")
    # Load only required tables in batches
    if obj_type == "table" and name in ["table1", "table2"]:
        for batch_df in load_table_in_batches(db_path, name):
            # Process batch_df as needed
            pass
```

</details>

## Initial Data Inspection Framework

### Systematic Inspection Checklist

A comprehensive data inspection should follow three distinct phases, each building upon the previous:

- **Phase 1: Structural Analysis** - Understand the basic structure and organization of your data
- **Phase 2: Content Analysis** - Examine the actual data values and patterns
- **Phase 3: Quality Analysis** - Assess the overall quality and reliability of your data

#### Phase 1: Structural Analysis

**Objectives**: Understand the basic structure and organization of your data

**Key Elements to Examine**:

- **Data Dimensions**: Number of rows and columns, memory footprint
- **Column Analysis**: Data types, column names, and type distribution
- **Index Structure**: Index type, range, and potential duplicates
- **Basic Integrity**: Structural consistency and format compliance

<details>
<summary>Example of structural analysis</summary>

```python
# Data Dimensions
print("Shape (rows, columns):", df.shape)
print("Memory footprint: {:.2f} MB".format(df.memory_usage(deep=True).sum() / 1024**2))

# Column Analysis
print("Data types:\n", df.dtypes)
print("Column names:", df.columns.tolist())
print("Type distribution:\n", df.dtypes.value_counts())

# Index Structure
print("Index type:", type(df.index))
print("Index range:", df.index.min(), "to", df.index.max())
print("Duplicate index values:", df.index.duplicated().sum())

# Basic Integrity
print("Missing values per column:\n", df.isnull().sum())
# Example format compliance for an 'email' column
if 'email' in df.columns:
    valid_email_rate = df['email'].str.contains(r'^[^@]+@[^@]+\.[^@]+$', na=False).mean()
    print("Valid email format rate:", valid_email_rate)
```

</details>

**Questions to Answer**:

- How large is this dataset and can it fit in memory?
- What types of data are we working with (numeric, categorical, dates)?
- Are there any obvious structural issues?
- Is the data properly indexed for analysis?

**FAQs**:

<details>
<summary>How large is to large for a single DataFrame?</summary>

In general, a DataFrame should be small enough to fit comfortably in memory. This often means:

- **For small datasets** (a few hundred thousand rows or less), you can usually load the entire dataset into a single DataFrame without issues.
- **For medium datasets** (hundreds of thousands to a few million rows), you might need to consider chunking the data or using Dask for out-of-core computation.
- **For large datasets** (millions of rows or more), you should definitely look into distributed computing frameworks like Dask or PySpark, or use database solutions to handle the data.

Ultimately, the "too large" threshold depends on your system's memory capacity and the complexity of the operations you need to perform.

</details>

<details>
<summary>What kind of obvious structural issues should I look for?</summary>

When inspecting the structure of your data, look for:

- Missing or null values in important columns
- Inconsistent data types within a column
- Unexpectedly high cardinality in categorical columns
- Duplicated rows or columns
- Improperly formatted dates or strings
- Columns with mixed data types (e.g., numeric and string values in the same column)
- Inconsistent naming conventions for columns (e.g., snake_case vs camelCase)
- Unexpectedly large or small values in numeric columns (e.g., negative prices)
- Columns that should be unique but contain duplicates (e.g., user IDs, email addresses)

</details>

<details>
<summary>What does a properly indexed DataFrame look like?</summary>

A properly indexed DataFrame should have:

- A unique index for each row, allowing for fast lookups and joins
- A meaningful index that reflects the data (e.g., timestamps for time series data)
- No missing or null values in the index
- A consistent index type (e.g., all integers or all dates)

</details>

#### Phase 2: Content Analysis

**Objectives**: Examine the actual data values and patterns

**Key Elements to Examine**:

- **Data Samples**: Representative samples from beginning and end of dataset
- **Statistical Summaries**: Descriptive statistics for numerical columns
- **Categorical Patterns**: Unique values, frequencies, and distributions
- **Temporal Patterns**: Date ranges, chronological order, and time gaps

<details>
<summary>Example of content analysis</summary>

```python
# Data Samples
print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nRandom sample of 5 rows:")
print(df.sample(5))

# Statistical Summaries
print("Descriptive statistics for numerical columns:")
print(df.describe())

# Categorical Patterns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    print(f"\n{col} - Unique values: {df[col].nunique()}")
    print(f"Most frequent values:\n{df[col].value_counts().head()}")

# Temporal Patterns (assuming 'date' column exists)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total time span: {df['date'].max() - df['date'].min()}")
    print(f"Date gaps larger than 1 day: {(df['date'].diff() > pd.Timedelta(days=1)).sum()}")
```

</details>

**Questions to Answer**:

- What do the actual data values look like?
- Are there obvious outliers or anomalies?
- How diverse are the categorical variables?
- Do date/time columns show expected patterns?

#### Phase 3: Quality Assessment

**Objectives**: Identify data quality issues and potential problems

**Key Elements to Examine**:

- **Missing Value Analysis**: Patterns, percentages, and systematic gaps
- **Duplicate Detection**: Complete duplicates and ID column uniqueness
- **Suspicious Patterns**: Placeholder values, test data, and anomalies
- **Data Consistency**: Cross-column validation and logical consistency

<details>
<summary>Example of quality assessment</summary>

```python
# Missing Value Analysis
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_data.index,
    'Missing Count': missing_data.values,
    'Missing Percentage': missing_percent.values
}).sort_values('Missing Percentage', ascending=False)
print("Missing data analysis:")
print(missing_df[missing_df['Missing Count'] > 0])

# Duplicate Detection
print(f"\nTotal duplicate rows: {df.duplicated().sum()}")
if 'id' in df.columns:
    print(f"Duplicate IDs: {df['id'].duplicated().sum()}")

# Suspicious Patterns
for col in df.select_dtypes(include=['object']).columns:
    suspicious_values = ['unknown', 'n/a', 'null', 'test', '999', '-1']
    for val in suspicious_values:
        count = df[col].astype(str).str.lower().str.contains(val, na=False).sum()
        if count > 0:
            print(f"Column '{col}' contains '{val}': {count} times")

# Data Consistency (example for price and quantity columns)
if 'price' in df.columns and 'quantity' in df.columns:
    print(f"\nNegative prices: {(df['price'] < 0).sum()}")
    print(f"Zero quantities: {(df['quantity'] == 0).sum()}")
```

</details>

**Questions to Answer**:

- How much data is missing and why?
- Are there duplicate records that need handling?
- Do I see obvious data quality flags (999, "unknown", etc.)?
- Are there values that violate business logic?

**FAQs**:

<details>
<summary>How much missing data is too much?</summary>

The acceptable amount of missing data depends on your analysis goals:

- **Less than 5%**: Generally acceptable for most analyses
- **5-15%**: May require imputation strategies
- **15-30%**: Significant impact on analysis; consider if column is necessary
- **More than 30%**: Often better to exclude the column unless it's critical

Consider the pattern of missing data:

- **Missing Completely at Random (MCAR)**: Easiest to handle
- **Missing at Random (MAR)**: Can be imputed using other variables
- **Missing Not at Random (MNAR)**: Most challenging; may indicate systematic issues

</details>

<details>
<summary>What constitutes a suspicious pattern in data?</summary>

Common suspicious patterns include:

- **Placeholder values**: "unknown", "N/A", "TBD", "999", "-1"
- **Test data**: Names like "Test User", emails like "test@test.com"
- **System defaults**: Dates like "1900-01-01", "1970-01-01"
- **Impossible values**: Negative ages, future birth dates, prices of $0.00
- **Inconsistent formatting**: Mixed case, extra spaces, special characters
- **Extreme outliers**: Values that are orders of magnitude different from others

</details>

### Business Logic Validation

Understanding your domain is crucial for effective data validation. Different industries have different rules and expectations.

#### Common Business Rule Categories

**Financial Data Rules**:

- Amounts should be positive (prices, costs, revenues)
- Debits and credits should balance
- Transaction dates should be reasonable
- Account balances should reconcile

<details>
<summary>Example of financial data validation</summary>

```python
# Financial validation examples
if 'amount' in df.columns:
    print(f"Negative amounts: {(df['amount'] < 0).sum()}")
    
if 'transaction_date' in df.columns:
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    future_dates = df['transaction_date'] > pd.Timestamp.now()
    print(f"Future transaction dates: {future_dates.sum()}")
    
# Balance validation (if debit/credit columns exist)
if 'debit' in df.columns and 'credit' in df.columns:
    balance_diff = abs(df['debit'].sum() - df['credit'].sum())
    print(f"Debit/Credit imbalance: ${balance_diff:.2f}")
```

</details>

**Customer Data Rules**:

- Email addresses should follow valid format
- Ages should be within reasonable bounds
- Phone numbers should match expected patterns
- Customer IDs should be unique

<details>
<summary>Example of customer data validation</summary>

```python
# Customer validation examples
if 'email' in df.columns:
    email_pattern = r'^[^@]+@[^@]+\.[^@]+$'
    valid_emails = df['email'].str.match(email_pattern, na=False)
    print(f"Invalid email formats: {(~valid_emails).sum()}")

if 'age' in df.columns:
    unreasonable_ages = (df['age'] < 0) | (df['age'] > 120)
    print(f"Unreasonable ages: {unreasonable_ages.sum()}")

if 'customer_id' in df.columns:
    duplicate_customers = df['customer_id'].duplicated().sum()
    print(f"Duplicate customer IDs: {duplicate_customers}")
```

</details>

**Inventory/Sales Rules**:

- Quantities should be positive
- Prices should be reasonable for product categories
- Product IDs should exist in catalog
- Sale dates should be chronological

**Temporal Data Rules**:

- Dates should not be in the future (unless expected)
- Created dates should precede updated dates
- Time series should be chronologically ordered
- Date ranges should align with business operations

#### Domain-Specific Considerations

Different industries have specific validation requirements that should guide your inspection process.

**E-commerce Platforms**:

- Order totals should match item sum plus taxes/shipping
- Customer purchase history should be chronological
- Product categories should be standardized
- Return dates should follow purchase dates

**Healthcare Data**:

- Patient ages should align with procedure appropriateness
- Medication dosages should be within safe ranges
- Visit dates should follow logical sequences
- Diagnostic codes should be valid and current

**IoT/Sensor Data**:

- Sensor readings should be within physical limits
- Timestamps should be regularly spaced
- Device IDs should be registered and active
- Environmental readings should be correlated

## Best Practices and Real-World Considerations

- **Intelligent Data Type Optimization**: Optimize data types to reduce memory usage and improve performance.
- **Memory Monitoring Principles**: Track memory usage before and after optimizations, calculate savings, and adjust strategies as needed.
- **Sample-Based Analysis Approach**: For large datasets, work with representative samples to speed up initial inspection.
- **Chunked Processing Strategies**: Process large files in manageable chunks to avoid memory overflow.

<details>
<summary>Example of data type optimization</summary>

```python
import numpy as np

def optimize_dataframe(df):
    """Optimize DataFrame memory usage by downcasting numeric types"""
    original_memory = df.memory_usage(deep=True).sum()
    
    # Optimize integers
    for col in df.select_dtypes(include=['int']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min >= 0:  # Unsigned integers
            if col_max < np.iinfo(np.uint8).max:
                df[col] = df[col].astype(np.uint8)
            elif col_max < np.iinfo(np.uint16).max:
                df[col] = df[col].astype(np.uint16)
            elif col_max < np.iinfo(np.uint32).max:
                df[col] = df[col].astype(np.uint32)
        else:  # Signed integers
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    
    # Optimize floats
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert high-cardinality object columns to category
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    new_memory = df.memory_usage(deep=True).sum()
    print(f"Memory optimization saved: {(original_memory - new_memory) / 1024**2:.2f} MB")
    print(f"Memory reduction: {((original_memory - new_memory) / original_memory) * 100:.1f}%")
    
    return df

# Apply optimization
df_optimized = optimize_dataframe(df.copy())
```

</details>

<details>
<summary>Example of sample-based analysis</summary>

```python
def create_representative_sample(df, sample_size=10000, stratify_column=None):
    """Create a representative sample for initial analysis"""
    
    if len(df) <= sample_size:
        return df
    
    if stratify_column and stratify_column in df.columns:
        # Stratified sampling to maintain distribution
        sample_df = df.groupby(stratify_column, group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(df)))))
        )
    else:
        # Simple random sampling
        sample_df = df.sample(n=sample_size, random_state=42)
    
    print(f"Created sample: {len(sample_df)} rows from {len(df)} total rows")
    print(f"Sample represents {len(sample_df)/len(df)*100:.1f}% of the data")
    
    return sample_df

# Create and analyze sample
sample_df = create_representative_sample(df, sample_size=5000, stratify_column='category')
# Perform initial inspection on sample_df
```

</details>

<details>
<summary>Example of chunked processing</summary>

```python
def process_large_file_in_chunks(file_path, chunk_size=10000):
    """Process large CSV files in chunks to avoid memory issues"""
    
    chunk_stats = []
    total_rows = 0
    
    for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        # Process each chunk
        chunk_info = {
            'chunk_number': chunk_num,
            'rows': len(chunk),
            'memory_mb': chunk.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': chunk.isnull().sum().sum(),
            'duplicates': chunk.duplicated().sum()
        }
        chunk_stats.append(chunk_info)
        total_rows += len(chunk)
        
        if chunk_num % 10 == 0:  # Progress update every 10 chunks
            print(f"Processed chunk {chunk_num}, total rows so far: {total_rows}")
    
    # Aggregate statistics
    stats_df = pd.DataFrame(chunk_stats)
    print(f"\nProcessing complete:")
    print(f"Total rows: {total_rows}")
    print(f"Total chunks: {len(chunk_stats)}")
    print(f"Average chunk size: {stats_df['rows'].mean():.0f} rows")
    print(f"Total missing values: {stats_df['missing_values'].sum()}")
    print(f"Total duplicates: {stats_df['duplicates'].sum()}")
    
    return stats_df

# Process large file
chunk_stats = process_large_file_in_chunks('data/large_dataset.csv')
```

</details>

### Error Handling and Logging

#### Error Management

Robust data processing requires comprehsive error handling that anticipates common failure scenarios like:

- File not found
- Memory overflow
- Parsing errors
- Data type mismatches
- Network issues (for remote data sources)
- Database connection failures
- Unexpected data formats

<details>
<summary>Example of robust error handling</summary>

```python
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def robust_data_loader(file_path, backup_path=None):
    """Load data with comprehensive error handling"""
    
    try:
        logging.info(f"Starting data load from: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path) / 1024**2  # MB
        logging.info(f"File size: {file_size:.2f} MB")
        
        if file_size > 1000:  # Warn for files > 1GB
            logging.warning(f"Large file detected ({file_size:.2f} MB), consider chunked processing")
        
        # Attempt to load data
        df = pd.read_csv(file_path)
        
        # Validate loaded data
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")
        
        logging.info(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        if backup_path and os.path.exists(backup_path):
            logging.info(f"Attempting to load backup file: {backup_path}")
            return robust_data_loader(backup_path)
        return None
        
    except pd.errors.EmptyDataError:
        logging.error("File is empty or contains no data")
        return None
        
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing file: {e}")
        # Try with different parameters
        try:
            logging.info("Attempting to load with alternative parameters")
            df = pd.read_csv(file_path, sep=';', encoding='latin-1')
            logging.info("Successfully loaded with alternative parameters")
            return df
        except Exception as e2:
            logging.error(f"Alternative loading also failed: {e2}")
            return None
            
    except Exception as e:
        logging.error(f"Unexpected error loading data: {e}")
        return None

# Use the robust loader
df = robust_data_loader('data/my_data.csv', backup_path='data/backup_data.csv')
```

</details>

### Documentation and Reproducibility

#### Automated Documentation Generation

Creating comprehensive documentation should be an automated part of your data inspection process.

<details>
<summary>Example of automated data profiling</summary>

```python
import json
from datetime import datetime

def generate_data_profile(df, output_path=None):
    """Generate comprehensive data profile report"""
    
    profile = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        },
        'columns': {},
        'data_quality': {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
    }
    
    # Analyze each column
    for col in df.columns:
        col_profile = {
            'dtype': str(df[col].dtype),
            'missing_count': df[col].isnull().sum(),
            'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
            'unique_values': df[col].nunique(),
            'unique_percentage': (df[col].nunique() / len(df)) * 100
        }
        
        if df[col].dtype in ['int64', 'float64']:
            col_profile.update({
                'min': float(df[col].min()) if pd.notna(df[col].min()) else None,
                'max': float(df[col].max()) if pd.notna(df[col].max()) else None,
                'mean': float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                'std': float(df[col].std()) if pd.notna(df[col].std()) else None,
                'outliers_iqr': detect_outliers_iqr(df[col])
            })
        elif df[col].dtype == 'object':
            col_profile.update({
                'most_frequent': df[col].value_counts().head().to_dict(),
                'average_length': df[col].astype(str).str.len().mean()
            })
        
        profile['columns'][col] = col_profile
    
    # Save profile if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)
        print(f"Data profile saved to: {output_path}")
    
    return profile

def detect_outliers_iqr(series):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    return int(outliers)

# Generate profile
profile = generate_data_profile(df, 'data_profile_report.json')
print(f"Data quality score: {100 - profile['data_quality']['missing_percentage']:.1f}%")
```

</details>

## Workflow Guidelines

### Complete Data Loading and Inspection Workflow

The following workflow provides a systematic approach that can be adapted to any data source or domain.

#### Workflow Stages

**Complete Data Loading and Initial Inspection Checklist**

- [ ] Preparation and Assessment
    - [ ] Identify data source format and estimate size
    - [ ] Assess available system memory and processing capabilities
    - [ ] Set up Python environment with required packages
    - [ ] Configure logging and error handling infrastructure
    - [ ] Prepare backup strategies for large or critical datasets

- [ ] Data Loading
    - [ ] Select appropriate loading strategy based on data format and size
    - [ ] Implement robust error handling during loading
    - [ ] Capture metadata (shape, columns, types) for loaded data
    - [ ] Validate successful loading and data integrity

- [ ] Systematic Inspection
    - [ ] Perform structural analysis (dimensions, types, index, integrity)
    - [ ] Conduct content analysis (samples, statistics, patterns)
    - [ ] Assess data quality (missing values, duplicates, suspicious patterns)
    - [ ] Apply domain-specific business logic validation

- [ ] Optimization and Documentation
    - [ ] Optimize data types and memory usage
    - [ ] Document inspection findings and data quality metrics
    - [ ] Generate automated data profiling reports
    - [ ] Establish reproducible workflows and version control

### Domain-Specific Configuration Guidelines

Understanding your domain helps tailor the inspection process to identify the most relevant data quality issues.

#### E-commerce Platform Data

**Expected Table Structure**:

- Core entities: customers, orders, products, order_items
- Supporting tables: categories, reviews, inventory
- Key relationships: customer_id, product_id, order_id

**Business Validation Focus**:

- Price consistency between orders and catalog
- Email format validation for customer data
- Inventory levels vs. sales quantities
- Order total calculations and tax logic

<details>
<summary>Example of e-commerce validation</summary>

```python
# E-commerce specific validations
def validate_ecommerce_data(orders_df, products_df, customers_df):
    """Validate e-commerce data relationships and business rules"""
    
    validation_results = {}
    
    # Price consistency check
    if 'product_id' in orders_df.columns and 'product_id' in products_df.columns:
        merged = orders_df.merge(products_df, on='product_id', suffixes=('_order', '_catalog'))
        price_mismatches = (merged['price_order'] != merged['price_catalog']).sum()
        validation_results['price_mismatches'] = price_mismatches
    
    # Email validation
    if 'email' in customers_df.columns:
        email_pattern = r'^[^@]+@[^@]+\.[^@]+$'
        invalid_emails = (~customers_df['email'].str.match(email_pattern, na=False)).sum()
        validation_results['invalid_emails'] = invalid_emails
    
    # Order total validation
    if all(col in orders_df.columns for col in ['subtotal', 'tax', 'shipping', 'total']):
        calculated_total = orders_df['subtotal'] + orders_df['tax'] + orders_df['shipping']
        total_mismatches = (abs(calculated_total - orders_df['total']) > 0.01).sum()
        validation_results['total_calculation_errors'] = total_mismatches
    
    return validation_results

# Run validation
validation_results = validate_ecommerce_data(orders_df, products_df, customers_df)
for check, result in validation_results.items():
    print(f"{check}: {result} issues found")
```

</details>

**Common Issues to Watch**:

- Customer duplicate detection across systems
- Product catalog inconsistencies
- Abandoned cart vs. completed order distinction
- Seasonal sales pattern validation

#### Financial Data Systems

**Expected Table Structure**:

- Core entities: transactions, accounts, customers
- Supporting tables: product_types, branches, regulations
- Key relationships: account_id, customer_id, transaction_id

**Business Validation Focus**:

- Transaction balance validation (debits = credits)
- Date consistency for financial periods
- Account balance reconciliation
- Regulatory compliance field validation

**Common Issues to Watch**:

- Currency conversion accuracy
- Transaction timing and settlement dates
- Account closure and reactivation handling
- Cross-border transaction complexity

#### IoT and Sensor Data

**Expected Table Structure**:

- Core entities: sensor_readings, sensors, locations
- Supporting tables: device_types, maintenance_logs
- Key relationships: sensor_id, location_id, device_id

**Business Validation Focus**:

- Sensor reading ranges within physical limits
- Timestamp chronological ordering
- Device online/offline status consistency
- Environmental correlation between sensors

**Common Issues to Watch**:

- Time zone handling across global deployments
- Sensor calibration drift over time
- Missing data during maintenance windows
- Network connectivity gaps in readings

### Quality Assurance and Validation

#### Data Quality Metrics Framework

Establishing quantitative metrics helps track data quality over time and communicate issues to stakeholders.

<details>
<summary>Example of quality metrics calculation</summary>

```python
def calculate_quality_metrics(df):
    """Calculate comprehensive data quality metrics"""
    
    metrics = {}
    
    # Completeness metrics
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    metrics['completeness_rate'] = ((total_cells - missing_cells) / total_cells) * 100
    
    # Column-level completeness
    col_completeness = {}
    for col in df.columns:
        col_completeness[col] = ((len(df) - df[col].isnull().sum()) / len(df)) * 100
    metrics['column_completeness'] = col_completeness
    
    # Uniqueness metrics (for potential ID columns)
    uniqueness_metrics = {}
    for col in df.columns:
        if 'id' in col.lower() or col.lower().endswith('_key'):
            uniqueness_metrics[col] = (df[col].nunique() / len(df)) * 100
    metrics['uniqueness_rates'] = uniqueness_metrics
    
    # Consistency metrics
    metrics['duplicate_rate'] = (df.duplicated().sum() / len(df)) * 100
    
    # Data type consistency
    type_consistency = {}
    for col in df.select_dtypes(include=['object']).columns:
        # Check if numeric strings exist in object columns
        numeric_strings = pd.to_numeric(df[col], errors='coerce').notna().sum()
        if numeric_strings > 0:
            type_consistency[col] = f"{numeric_strings} numeric values in text column"
    metrics['type_consistency_issues'] = type_consistency
    
    return metrics

# Calculate and display metrics
quality_metrics = calculate_quality_metrics(df)
print(f"Overall completeness: {quality_metrics['completeness_rate']:.2f}%")
print(f"Duplicate rate: {quality_metrics['duplicate_rate']:.2f}%")

# Identify columns with quality issues
poor_quality_cols = [col for col, rate in quality_metrics['column_completeness'].items() if rate < 90]
if poor_quality_cols:
    print(f"Columns with <90% completeness: {poor_quality_cols}")
```

</details>

**FAQs**:

<details>
<summary>What constitutes good data quality?</summary>

Data quality benchmarks vary by industry and use case, but general guidelines include:

- **Completeness**: >95% for critical fields, >85% for supporting fields
- **Accuracy**: >99% for calculated fields, >95% for manually entered data
- **Consistency**: <1% duplicate rate, consistent formatting across records
- **Timeliness**: Data freshness appropriate for business needs
- **Validity**: >95% compliance with business rules and formats

These are starting points - your specific requirements may be higher or lower based on how the data will be used.

</details>

<details>
<summary>How do I prioritize data quality issues?</summary>

Prioritize data quality issues based on:

1. **Business Impact**: Issues affecting revenue, compliance, or customer experience
2. **Frequency**: Problems occurring in high-volume or frequently-used data
3. **Downstream Effects**: Issues that propagate to multiple systems or reports
4. **Fix Complexity**: Balance impact against the effort required to resolve
5. **Data Criticality**: Issues in core business entities vs. supporting data

Focus on high-impact, high-frequency issues first, then work through the priority matrix.

</details>

## Summary and Next Steps

This comprehensive guide provides a professional framework for data loading and initial inspection that can be applied to any data source or domain. The systematic approach ensures consistent, reliable results while maintaining focus on practical business outcomes.

### Core Principles

1. **Systematic Approach**: Follow structured workflows for consistent, reliable results
2. **Error Resilience**: Anticipate and gracefully handle data loading and quality issues
3. **Business Context**: Apply domain-specific knowledge and validation rules
4. **Performance Awareness**: Optimize for memory usage and processing efficiency
5. **Documentation Standards**: Generate comprehensive, reproducible documentation
6. **Quality Focus**: Implement comprehensive quality assessment and monitoring

### Implementation Checklist

**Before You Start**:

- [ ] Assess data source format, size, and complexity
- [ ] Set up appropriate Python environment and logging
- [ ] Understand the business domain and expected data patterns
- [ ] Plan for error handling and fallback strategies

**During Data Loading**:

- [ ] Implement robust loading with comprehensive error handling
- [ ] Monitor memory usage and optimize data types
- [ ] Capture metadata and document any loading issues
- [ ] Validate successful loading across all data sources

**During Inspection**:

- [ ] Execute systematic structural, content, and quality analysis
- [ ] Apply domain-specific business logic validation
- [ ] Generate automated data quality reports
- [ ] Document findings and recommendations for stakeholders

**After Inspection**:

- [ ] Create data quality metrics and monitoring dashboards
- [ ] Establish version control for data processing scripts
- [ ] Generate comprehensive documentation for future reference
- [ ] Plan next steps for data cleaning and preparation

### Immediate Next Steps

After completing data loading and initial inspection, the natural progression includes:

- **Data Cleaning and Standardization**: Address identified quality issues
- **Missing Data Treatment**: Implement appropriate imputation strategies
- **Duplicate Resolution**: Systematic identification and handling of duplicates
- **Feature Engineering**: Create analytical features from clean, validated data
- **Data Preparation for Analysis**: Structure data for specific analytical goals

This guide serves as a comprehensive reference that data professionals can adapt and extend based on their specific requirements, data sources, and organizational needs. The combination of systematic methodology, practical examples, and domain-specific guidance ensures that your data loading and inspection process will be thorough, efficient, and professional.
