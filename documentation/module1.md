# Module 1 — Data Understanding

## Dataset overview

The project uses the PaySim synthetic mobile money transaction dataset, containing 6,362,620 transactions and 11 columns.

The dataset includes transaction characteristics, sender and receiver account balances, transaction types, simulation time steps, and fraud labels.

## Data quality

- No missing values were detected.
- No duplicate rows were detected.
- The dataset contains approximately 1 GB of data in memory.

## Class distribution

The dataset is highly imbalanced:

- Legitimate transactions: 6,354,407 (99.8709%).
- Fraudulent transactions: 8,213 (0.1291%).

This imbalance makes accuracy an insufficient evaluation metric. Precision, recall, F2-score, MCC and PR-AUC will be important for evaluating the models.

## Fraud patterns

Fraudulent transactions occur in the TRANSFER and CASH_OUT categories in the observed dataset.

TRANSFER has a fraud rate of approximately 0.7688%, while CASH_OUT has a fraud rate of approximately 0.1840%.

The median fraudulent transaction amount is approximately 441,423, compared with 74,685 for legitimate transactions.

## Temporal characteristics

The dataset contains 743 unique simulation time steps. The `step` variable will be used to preserve chronological order during dataset splitting.

## Existing detection flag

The existing `isFlaggedFraud` rule flagged 16 transactions, all of which were fraudulent. However, its recall was only approximately 0.195%.

This variable will be excluded from model training to avoid using an existing detection result as a predictor.

## Initial conclusions

Transaction amount, transaction type, account balances and temporal behavior are potential sources of predictive information.

Further preprocessing and feature engineering are required before model training.