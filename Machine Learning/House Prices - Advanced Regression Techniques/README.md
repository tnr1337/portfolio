## House Price Regression - Advanced Regression Techniques

### Project Title

House Prices - Advanced Regression Techniques

### Description

Predict house sale prices using advanced regression models and feature engineering techniques.

### Table of Contents

* [Installation](#installation)
* [Dataset](#dataset)
* [Usage](#usage)
* [Notebooks](#notebooks)
* [Model Training](#model-training)
* [Evaluation](#evaluation)
* [File Structure](#file-structure)
* [Contributing](#contributing)
* [License](#license)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd portfolio/Machine Learning/House Prices - Advanced Regression Techniques
# (Optional) create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate   # Windows
```

### Dataset

All data files are in the `data/` folder:

* `data/train.csv` — training data
* `data/test.csv` — test data
* `data/sample_submission.csv` — sample submission format
* `data/data_description.txt` — feature descriptions

Source: [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

### Notebooks

* `House_Prices.ipynb` — end-to-end EDA, feature engineering, and model building

### Model Training

* Implemented regression techniques: Linear Regression, Ridge, Lasso, Random Forest, XGBoost
* Hyperparameter tuning via GridSearchCV and cross-validation

### Evaluation

* Primary metric: Root Mean Squared Error (RMSE)
* Example performance: \~0.12 RMSE on validation set

### File Structure

```
├── data/
│   ├── data_description.txt
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
├── House_Prices.ipynb
└── README.md
```

### Contributing

Contributions are welcome! Please open an issue or pull request.

### License

This project is licensed under the MIT License.
