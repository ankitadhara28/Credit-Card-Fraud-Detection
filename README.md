# Credit Card Fraud Detection

![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)
![Scikit-learn Version](https://img.shields.io/badge/Scikit--learn-0.24%2B-orange.svg)

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project focuses on building a machine learning model to detect fraudulent credit card transactions. Credit card fraud is a significant challenge for financial institutions and consumers, leading to substantial financial losses and erosion of trust. By leveraging machine learning, we aim to accurately identify suspicious transactions, thereby minimizing fraudulent activities.

## Dataset
The dataset used for this project is `creditcard_2023.csv`, obtained from a publicly available source (e.g., Kaggle). It contains anonymized transaction data with the following key characteristics:

-   **Highly Imbalanced**: The dataset exhibits a severe class imbalance, with a significantly higher number of legitimate transactions (Class 0) compared to fraudulent ones (Class 1). Specifically, there are 18309 legitimate transactions and only 81 fraudulent transactions.
-   **Features**: The dataset comprises 30 features, primarily anonymized principal components (`V1` through `V28`), `Amount`, and `id`. The target variable, `Class`, indicates whether a transaction is legitimate (0) or fraudulent (1).

## Methodology
The fraud detection model was developed following these key steps:

1.  **Data Loading and Initial Exploration**:
    *   Loaded the `creditcard_2023.csv` file into a pandas DataFrame.
    *   Performed initial checks on the data, including viewing the first/last rows, inspecting data types (`.info()`), and identifying missing values (`.isnull().sum()`).
    *   Analyzed the distribution of the `Class` variable to understand the extent of class imbalance.

2.  **Handling Imbalanced Data**:
    *   To address the severe class imbalance, a balanced dataset was created. Legitimate transactions (Class 0) and fraudulent transactions (Class 1) were separated.
    *   A sample of legitimate transactions (`legit_sample`) was randomly selected to match the number of fraudulent transactions (81 samples).
    *   These two subsets (`legit_sample` and `fraud`) were then concatenated to form a new, balanced dataset (`new_dataset`).

3.  **Data Splitting**:
    *   The balanced dataset was split into features (`x`, dropping the 'Class' column) and the target variable (`y`, the 'Class' column).
    *   The data was further divided into training and testing sets (`x_train`, `x_test`, `y_train`, `y_test`) using `train_test_split`. A `test_size` of 0.2 (20% for testing) was used, with `stratify=y` to ensure that the class distribution is preserved in both training and testing sets, and `random_state=2` for reproducibility.

4.  **Model Training**:
    *   A Logistic Regression model (`LogisticRegression()`) was initialized.
    *   The model was trained using the `x_train` and `y_train` datasets.

5.  **Model Evaluation**:
    *   The trained model's performance was evaluated using `accuracy_score` on both the training and testing datasets.

## Results

The Logistic Regression model achieved the following accuracy scores:

-   **Accuracy on training data**: `0.9767` (approximately 97.67%)
-   **Accuracy on test data**: `0.9394` (approximately 93.94%)

These results indicate that the model performs well on both seen and unseen data, demonstrating its ability to generalize. The balancing technique applied was effective in allowing the model to learn from the minority class.

## Future Improvements

To further enhance the model's performance and robustness, consider the following:

-   **Exploring other algorithms**: Evaluate other classification algorithms such as Random Forest, Gradient Boosting, or Support Vector Machines.
-   **Hyperparameter Tuning**: Optimize the hyperparameters of the chosen model using techniques like GridSearchCV or RandomizedSearchCV.
-   **Advanced Resampling Techniques**: Investigate more sophisticated resampling methods (e.g., SMOTE, ADASYN) for handling imbalanced datasets.
-   **Feature Engineering**: Create new features from existing ones that might provide more predictive power.
-   **Cross-Validation**: Implement k-fold cross-validation for a more robust evaluation of the model's performance.

## Installation

To run this notebook, you will need the following Python libraries. You can install them using pip:

```bash
pip install pandas numpy scikit-learn
```

## Usage

1.  Clone this repository:
    ```bash
    git clone https://github.com/yourusername/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```
2.  Place the `creditcard_2023.csv` dataset in the same directory as the notebook.
3.  Open the `Credit Card Fraud Detection.ipynb` notebook in Google Colab or Jupyter Notebook.
4.  Run all cells to replicate the analysis and model training.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
