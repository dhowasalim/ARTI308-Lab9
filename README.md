# ARTI308 – Decision Trees and Random Forests

Classification with **Decision Trees** and **Random Forests** using `scikit-learn`.

| File | Description |
|------|-------------|
| `01-Decision_Trees_and_Random_Forests.ipynb` | Lab: predicting Kyphosis after spinal surgery, with a tree visualization |
| `02-Decision_Trees_and_Random_Forest_Project.ipynb` | Project: predicting whether a LendingClub borrower fully paid back their loan |
| `kyphosis.csv`, `loan_data.csv` | Datasets |

## Project – LendingClub loans
EDA (FICO by credit policy / repayment, loan purpose, FICO vs interest rate) → dummy variables for `purpose` → 70/30 split (`random_state=101`) → Decision Tree vs Random Forest (600 trees).

| Model | Accuracy | Recall (not fully paid) |
|---|---|---|
| Decision Tree | ≈ 0.73 | ≈ 0.24 |
| Random Forest | ≈ 0.85 | ≈ 0.02 |

The Random Forest has higher overall accuracy, but neither model predicts the minority class (not fully paid) well because the data is imbalanced.

## Run it
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```
