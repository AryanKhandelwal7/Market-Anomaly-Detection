# Market-Anomaly-Detection

**Overview**

The Market Anomaly Detection (MAD) project is designed to act as an early warning system for predicting potential financial market crashes. Using historical data and machine learning models, the system identifies unusual market patterns, classifies market conditions, and suggests investment strategies to minimize risks and optimize returns.

**Features**
- Data Loading and Cleaning: Handles historical financial data, converts timestamps, and ensures the dataset is clean and ready for analysis.
- Time-Series Visualization: Creates color-coded bar plots to distinguish between stable periods (Y=0) and market crashes (Y=1).
- Feature Engineering: Calculates rolling statistics (e.g., 7-day rolling mean) to enhance data for machine learning models.
- Machine Learning Models: Trains and evaluates three models:
- Decision Tree
- Random Forest
- Gradient Boosting Classifier
- Performance Evaluation: Compares models using metrics like accuracy, precision, recall, F1-score, and cross-validation scores.

**How It Works**
1. Load and Prepare Data: Load the dataset, clean it, and transform it for analysis.
2. Visualize Data: Generate time-series plots to understand trends and anomalies in the market.
3. Enhance Data: Add rolling features to capture short-term market patterns.
4. Train Models: Train and evaluate machine learning models to predict market crashes.
5. Analyze Results: Compare model performance and visualize key metrics in a bar chart.

**Results**
The system highlights patterns in market data and successfully predicts market crashes using the best-performing model, which can further be used to design effective investment strategies.
