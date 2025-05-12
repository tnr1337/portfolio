## Titanic - Survival Prediction

### Project Title

Titanic - Survival Prediction

### Description

Predict passenger survival on the Titanic using classification algorithms and feature engineering.

### Table of Contents

* [Dataset](#dataset-1)

* [Notebooks](#notebooks-1)

* [Model Training](#model-training-1)

* [Evaluation](#evaluation-1)

* [File Structure](#file-structure-1)

* [Contributing](#contributing-1)

* [License](#license-1)

### Dataset

All data files are in the `data/` folder:

* `data/train.csv` — training set with labels
* `data/test.csv` — test set without labels
* `data/gender_submission.csv` — sample submission format

Source: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)

### Usage

```bash
# Open the notebook for EDA and modeling
jupyter notebook titanic.ipynb
```

### Notebooks

* `titanic.ipynb` — exploration, feature engineering, and modeling workflow

### Model Training

* Classification algorithms: Logistic Regression, Random Forest, Support Vector Machine, XGBoost
* Feature engineering: title extraction from names, age imputation, cabin presence indicator
* Hyperparameter tuning with GridSearchCV

### Evaluation

* Metrics: Accuracy, ROC-AUC
* Example performance: \~81% accuracy on validation set

### File Structure

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── titanic.ipynb
└── README.md
```

### Contributing

Contributions welcome! Please open an issue or pull request.

### License

This project is licensed under the MIT License.
