# Hotel Bookings Demand Analysis

This repository contains the code and resources for analyzing hotel bookings demand. Below is a brief explanation of the files and their purposes.

## Files and Directories

- **`take_home_data_scientist_2.ipynb`**: The main Jupyter Notebook where the analysis and modeling are performed. It includes data exploration, feature engineering, and machine learning modeling.

- **`pipeline.py`**: Contains the implementation of a scikit-learn pipeline for data preprocessing and feature transformation.

- **`transformers.py`**: Includes custom scikit-learn transformers for feature engineering and data cleaning.

- **`plotly_utils.py`**: Utility functions for creating visualizations using Plotly.

- **`Pipfile`**: Specifies the Python dependencies and environment setup for the project.

- **`Pipfile.lock`**: A locked version of the dependencies to ensure reproducibility.

- **`data/`**: Contains the dataset used for the analysis.
  - `hotels.csv`: The main dataset with hotel booking information.

- **`imgs/`**: Stores generated visualizations and plots.
  - Example files: `average week stays.png`, `avg_daily_rate.png`, etc.

## How to Use

1. Clone the repository.
2. Set up the environment using the `Pipfile`:
   ```bash
   pipenv install
   pipenv shell
3. Open the Jupyter Notebook (take_home_data_scientist_2.ipynb) to explore the analysis and run the code.