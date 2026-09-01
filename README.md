# Customer Churn and Missing-Data Analysis

An exploratory analysis of missing customer attributes and their impact on churn-model evaluation. The project compares missing-data strategies on a 3,000-row customer dataset and asks whether retaining imputed rows changes downstream churn scores.

## Project Files

- `analysis.py`: extracted analysis workflow from the original coursework notebook.
- `customer_missing_data.csv`: project input data copied from the coursework folder.
- `report.html` and `report.pdf`: project summary and findings.
- `requirements.txt`: Python dependencies.

## Questions and Methods

- How much data is missing, and does missingness vary with observed customer attributes?
- How do median, K-nearest neighbors, iterative Bayesian-ridge imputation (MICE-style), and ExtraTrees iterative imputation compare with complete-case deletion?
- On artificially masked values, how do the approaches compare on normalized RMSE, marginal distribution distance (Kolmogorov-Smirnov statistic), and correlation-matrix distance?
- How do the approaches compare on exploratory five-fold churn ROC-AUC and F1?

The notebook reports differences by segment, region, and tenure. These are diagnostics of observed associations, not proof that any field is Missing At Random (MAR) or Missing Not At Random (MNAR).

## Recorded Results

The saved notebook outputs report 3,000 rows and 1,420 complete cases. On the saved 15%-mask experiment, MICE has the lowest normalized RMSE (0.2660), ExtraTrees imputation the lowest average KS distance (0.0298), and KNN the lowest Frobenius distance between correlation matrices (0.1175). No one method wins on every metric.

The notebook's saved five-fold Random Forest comparison reports ROC-AUC values from 0.6777 (median imputation) to 0.6855 (complete-case deletion), with F1 values from 0.4410 to 0.4705. Treat these as exploratory recorded results, not unbiased generalization estimates.

## Important Limitations

- Imputation, scaling, and one-hot encoding are fitted before cross-validation in the original workflow. This can leak information across folds and make model scores optimistic. For a defensible estimate, put preprocessing and the classifier in an sklearn `Pipeline` evaluated within each fold.
- The missingness checks support associations with observed fields; they cannot establish MAR. The notebook's proposed MNAR explanation for credit utilization is a hypothesis, not a conclusion identified by these data.
- Masked-value results depend on the artificial masking mechanism and the complete-case subset; they may not represent the true missing-data process.
- This is coursework using a supplied dataset, not a deployed retention system or causal churn study.

## Run

Use Python 3.10 or newer. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python analysis.py
```

The script displays exploratory plots and prints tables to the console. The recorded figures above come from the original notebook outputs; rerunning may produce differences if package versions or the source data change.