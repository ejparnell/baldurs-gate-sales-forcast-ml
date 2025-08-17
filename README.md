# 🏪⚔️ Baldur's Gate Shop Sales Forecasting ML

*A comprehensive machine learning project for analyzing and forecasting shop sales in the fantasy world of Baldur's Gate*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data Science](https://img.shields.io/badge/Data%20Science-ML%20Pipeline-purple)](https://github.com/ejparnell/baldurs-gate-sales-forcast-ml)

## 📖 Overview

This project demonstrates a comprehensive data preprocessing and feature engineering pipeline for sales data from adventurer shops in the Baldur's Gate universe. The project focuses on best practices in data science workflows, covering everything from initial data exploration through advanced feature engineering techniques.

**Important Note**: This is a data preprocessing and feature engineering project. It does not include trained machine learning models, as the focus is on demonstrating robust data cleaning and feature engineering techniques. During the data cleaning process, it was discovered that the temporal data contained default/placeholder timestamps rather than actual transaction times, which influenced the preprocessing approach.

### 🎯 Project Objectives

- **Data Analysis**: Comprehensive exploration of fantasy shop sales patterns
- **Data Quality**: Implement robust data cleaning and preprocessing techniques
- **Feature Engineering**: Extract meaningful insights from temporal and categorical data
- **Pipeline Development**: Build reusable preprocessing workflows
- **Documentation**: Provide comprehensive guides for ML pipeline development
- **Educational Resource**: Demonstrate real-world data quality challenges and solutions

### 🌟 Key Features

- **Complete Data Pipeline**: End-to-end workflow from raw data to engineered features
- **Comprehensive Guides**: Detailed documentation for each preprocessing step
- **Best Practices**: Production-ready code following industry standards
- **Fantasy Domain**: Unique application in gaming/fantasy retail context
- **Educational Resource**: Suitable for learning data preprocessing and feature engineering
- **Real-world Challenges**: Demonstrates handling of data quality issues like placeholder timestamps

## 🗂️ Project Structure

```
baldurs-gate-shop-sales-ml/
├── README.md                              # Project documentation
├── data/
│   └── adventurer_mart.db                 # SQLite database with sales data
├── data_intermediate/                     # Processed data files
├── guide/                                 # Comprehensive ML guides
│   ├── data_loading_inspection.md         # Data loading and initial inspection
│   ├── exploratory_data_analysis.md       # EDA techniques and insights
│   ├── missing_data_imputation.md         # Handling missing values
│   ├── duplication_outlier_handling.md    # Data quality and outlier detection
│   ├── categorical_variable_cleaning.md   # Categorical data preprocessing
│   ├── numerical_variable_cleaning.md     # Numerical data preprocessing
│   └── time_date_feature_engineering.md   # Temporal feature engineering
├── notebooks/                             # Jupyter notebooks
│   ├── 01_data_loading_preview.ipynb      # Initial data exploration
│   ├── 02_exploratory_data_analysis.ipynb # Comprehensive EDA
│   ├── 03_missing_values_handling.ipynb   # Missing data strategies
│   ├── 04_duplicate_handling.ipynb        # Duplicate detection and removal
│   ├── 05_categorical_variables_cleaning.ipynb # Categorical preprocessing
│   ├── 06_numerical_variables_cleaning.ipynb   # Numerical preprocessing
│   ├── 07_outlier_detection_handling.ipynb     # Outlier analysis
│   └── 08_date_time_feature_engineering.ipynb  # Temporal features
└── requirements.txt                       # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git (for cloning the repository)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ejparnell/baldurs-gate-sales-forcast-ml.git
   cd baldurs-gate-sales-forcast-ml
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

### Quick Start

1. Start with `01_data_loading_preview.ipynb` to understand the dataset
2. Follow the numbered notebooks in sequence for the complete pipeline
3. Refer to the corresponding guides in the `guide/` folder for detailed explanations

## 📊 Dataset Description

The project uses a fantasy retail dataset containing sales transactions from adventurer shops in Baldur's Gate:

- **Source**: SQLite database (`adventurer_mart.db`)
- **Domain**: Fantasy/Gaming retail transactions
- **Features**: Customer data, product information, transaction details, timestamps
- **Size**: Multiple tables with relational structure
- **Data Quality Note**: During preprocessing, it was discovered that temporal data contained default/placeholder timestamps rather than actual transaction times, which required specialized handling approaches

### Key Data Tables

- **Transactions**: Sales records with timestamps and amounts
- **Customers**: Adventurer profiles and demographics
- **Products**: Magical items, weapons, and supplies
- **Shop Information**: Store locations and characteristics

### Data Quality Challenges Addressed

- **Placeholder Timestamps**: Handled default datetime values that didn't represent actual transaction times
- **Missing Values**: Various imputation strategies for different data types
- **Categorical Inconsistencies**: Standardization of adventurer classes and item categories
- **Outlier Detection**: Statistical identification of unusual sales patterns

## 🛠️ Data Processing Pipeline

### 1. Data Loading and Inspection

- Database connection and table exploration
- Initial data quality assessment
- Schema understanding and relationship mapping

### 2. Exploratory Data Analysis (EDA)

- Statistical summaries and distributions
- Correlation analysis and feature relationships
- Visualization of patterns and trends
- Business insights discovery

### 3. Data Quality and Preprocessing

- **Missing Data**: Advanced imputation strategies
- **Duplicates**: Detection and intelligent removal
- **Outliers**: Statistical identification and handling
- **Data Validation**: Consistency checks and quality metrics

### 4. Feature Engineering

- **Categorical Variables**: Encoding and transformation
- **Numerical Variables**: Scaling and normalization
- **Temporal Features**: Handling placeholder timestamps and extracting meaningful time-based features
- **Domain Features**: Fantasy-specific business logic

### 5. Advanced Feature Engineering

- Cyclical encoding for time patterns (when applicable)
- Lag features for temporal dependencies
- Aggregation features for customer behavior
- Business calendar integration

**Note**: This project focuses on the data preprocessing and feature engineering stages. No machine learning models are trained, as the emphasis is on demonstrating comprehensive data cleaning and feature engineering techniques.

## 📚 Documentation Guides

This project includes comprehensive guides that serve as both documentation and educational resources:

| Guide | Description | Key Topics |
|-------|-------------|------------|
| [Data Loading & Inspection](guide/data_loading_inspection.md) | Database connection and initial exploration | SQLite, schema analysis, data profiling |
| [Exploratory Data Analysis](guide/exploratory_data_analysis.md) | Statistical analysis and visualization | Distributions, correlations, business insights |
| [Missing Data Imputation](guide/missing_data_imputation.md) | Handling missing values strategically | Imputation strategies, validation techniques |
| [Duplication & Outlier Handling](guide/duplication_outlier_handling.md) | Data quality and anomaly detection | Duplicate detection, outlier analysis |
| [Categorical Variable Cleaning](guide/categorical_variable_cleaning.md) | Categorical data preprocessing | Encoding techniques, cardinality handling |
| [Numerical Variable Cleaning](guide/numerical_variable_cleaning.md) | Numerical data preprocessing | Scaling, transformation, distribution handling |
| [Time & Date Feature Engineering](guide/time_date_feature_engineering.md) | Temporal feature extraction | Cyclical encoding, lag features, seasonality |

## 🎯 Key Insights and Results

### Data Quality Discoveries

- **Temporal Data Issues**: Identified that timestamp data contained default/placeholder values rather than actual transaction times
- **Data Completeness**: Achieved high data quality through systematic preprocessing
- **Feature Engineering**: Successfully extracted meaningful features despite data quality challenges
- **Processing Efficiency**: Developed robust pipeline for handling real-world data issues

### Technical Achievements

- **Data Quality**: Comprehensive handling of missing values and inconsistencies
- **Feature Engineering**: 50+ meaningful features extracted from raw data
- **Pipeline Robustness**: Handles various data quality issues automatically
- **Documentation Quality**: Comprehensive guides for each preprocessing step
- **Educational Value**: Demonstrates real-world data preprocessing challenges

### Lessons Learned

- **Data Discovery**: Importance of thorough data exploration before feature engineering
- **Temporal Challenges**: Strategies for handling placeholder or invalid timestamps
- **Feature Adaptation**: Adjusting feature engineering approach based on data quality findings
- **Documentation**: Value of comprehensive documentation for complex preprocessing workflows

## 🔧 Technologies Used

### Core Technologies

- **Python 3.8+**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive development environment

### Database and Storage

- **SQLite**: Local database storage
- **Pickle**: Model and data serialization

### Development Tools

- **Git**: Version control
- **Jupyter Notebooks**: Interactive analysis
- **Markdown**: Documentation

## 📈 Future Enhancements

### Planned Features

- [ ] **Machine Learning Models**: Train forecasting models once temporal data issues are resolved
- [ ] **Advanced Time Series**: Implement specialized time series techniques for placeholder timestamp handling
- [ ] **Real-time Processing**: Streaming data pipeline for live transaction processing
- [ ] **Dashboard**: Interactive visualization dashboard for data quality monitoring
- [ ] **Data Quality Metrics**: Automated data quality scoring and monitoring

### Advanced Techniques

- [ ] **Deep Learning**: Neural networks for complex pattern recognition (pending model training phase)
- [ ] **Ensemble Methods**: Model stacking and blending (future modeling phase)
- [ ] **Feature Selection**: Automated feature importance analysis
- [ ] **Synthetic Data**: Generate realistic timestamps to replace placeholder values
- [ ] **Data Pipeline Automation**: Fully automated preprocessing pipeline

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Contribution Guidelines

- Follow existing code style and documentation patterns
- Add tests for new functionality
- Update documentation for any new features
- Ensure all notebooks run without errors
- Provide clear commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Baldur's Gate Universe**: Inspiration for the fantasy retail domain
- **Data Science Community**: Best practices and methodologies
- **Open Source Libraries**: Pandas, Scikit-learn, and the Python ecosystem
- **Educational Resources**: Various online courses and tutorials that informed the approach

## 📞 Contact

**Elizabeth Parnell** - [@ejparnell](https://github.com/ejparnell)

**Project Link**: [https://github.com/ejparnell/baldurs-gate-sales-forcast-ml](https://github.com/ejparnell/baldurs-gate-sales-forcast-ml)

---

*Built with ❤️ for the data science and fantasy gaming communities*
