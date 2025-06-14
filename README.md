wind-data-imputation/
├── data/
│ └── .gitkeep                  # downloaded CSV file
├── notebooks/
│ └── analysis.ipynb            # Jupyter Notebook for analysis and visualization
├── src/
│ └── __init__.py
│ └── custom_imputer.py         # new imputation class will go
├── .gitignore                  # standard gitignore for Python
├── README.md                   # project description
└── requirements.txt            # project dependencies

# Multiple Data-Driven Missing Imputation

This project is a Python implementation and benchmarking of the missing data imputation method described in the scientific paper "Multiple Data-Driven Missing Imputation" by Sergii Kavun and Alina Zamula.

The implementation is implemented as a scikit-learn-compatible class, which makes it easy to use in machine learning pipelines.

## Project structure

- `src/custom_imputer.py`: Implementation of the `KZImputer` imputer.
- `data/`: Directory for storing data.
- `notebooks/analysis.ipynb`: Jupyter Notebook with the full analysis, including:
- Testing on synthetic data.
- Downloading and preparing real data from Kaggle.
- Benchmarking with 8 other popular imputation methods.
- Visualization of results.
- `requirements.txt`: List of project dependencies.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/wind-data-imputation.git
   cd wind-data-imputation
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data

The analysis uses the [Wind Turbine Scada Dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset) from Kaggle.

1. Download the dataset (Kaggle authentication is required).
2. Unzip the archive and place the `T1.csv` file in the `data/` directory.

## Usage

Open and execute the cells in Jupyter Notebook `notebooks/analysis.ipynb` to reproduce the analysis.

## Results

A comparative analysis on real wind turbine power data showed that the `KavunZamulaImputer` method is competitive, especially compared to simple methods (mean, ffill). In our test, it showed results close to spline and linear interpolation, which turned out to be the best for this type of smooth time series.

![RMSE comparison](path/to/your/rmse_plot.png) <!-- Insert a screenshot of the plot here -->