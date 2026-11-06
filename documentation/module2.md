# Module 2 --- Data Preprocessing

**Project:** Explainable Fraud Detection\
**Dataset:** PaySim (`PS_20174392719_1491204439457_log.csv`)\
**Status:** Initial preprocessing completed

------------------------------------------------------------------------

## 1. Objective

The purpose of this module is to prepare the PaySim transaction data for
machine-learning experiments.

The workflow covers: - Initial feature selection - Chronological
train/validation/test splitting - Temporal leakage checks - Separation
of predictors and target - Numerical standardization - Categorical
one-hot encoding - Validation of the preprocessing pipeline

The original dataset is kept unchanged so experiments remain
reproducible.

## 2. Dataset overview

  Property                        Value
  ------------------------- -----------
  Transactions                6,362,620
  Original columns                   11
  Simulation steps                  743
  Fraudulent transactions         8,213
  Legitimate transactions     6,354,407
  Fraud rate                    0.1291%

The initial checks found no missing values and no fully duplicated rows.

The target column is `isFraud`: `0` means legitimate and `1` means
fraudulent. The dataset is highly imbalanced, so accuracy alone is not
an adequate evaluation metric.

## 3. Detection-time assumption

The system is assumed to detect fraud **immediately after transaction
execution**.

Under this assumption, the transaction has been processed and the
resulting account balances are available. Therefore, the initial feature
set includes both pre-transaction and post-transaction balance columns.

This assumption must be stated when presenting the project. If the
detection moment changes to before execution, the post-transaction
balance fields must be removed.

## 4. Feature selection

### 4.1 Initial model inputs

  Feature            Description
  ------------------ -----------------------------------------
  `step`             Simulation time step
  `type`             Transaction category
  `amount`           Transaction amount
  `oldbalanceOrg`    Sender balance before the transaction
  `newbalanceOrig`   Sender balance after the transaction
  `oldbalanceDest`   Receiver balance before the transaction
  `newbalanceDest`   Receiver balance after the transaction

### 4.2 Excluded columns

  -----------------------------------------------------------------------
  Column                              Reason
  ----------------------------------- -----------------------------------
  `isFraud`                           Target variable; not an input
                                      feature

  `isFlaggedFraud`                    Existing detection flag; excluded
                                      from model inputs

  `nameOrig`                          Raw sender identifier; retained for
                                      possible behavioral feature
                                      engineering

  `nameDest`                          Raw receiver identifier; retained
                                      for possible behavioral feature
                                      engineering
  -----------------------------------------------------------------------

The exploratory columns `orig_balance_error` and `dest_balance_error`
were removed from the working dataset. They were diagnostic
calculations, not approved model inputs.

## 5. Chronological data split

The data was split by complete `step` values to preserve time order and
prevent transactions from the same simulation hour from appearing in
different partitions.

The target proportions were 60% training, 20% validation and 20%
testing. Because complete time steps were kept together, the final
proportions are approximate.

  ---------------------------------------------------------------------------------
  Partition      Step range    Transactions        Share  Fraud cases    Fraud rate
  ------------ ------------ --------------- ------------ ------------ -------------
  Training           1--281       3,820,599       60.05%        3,193       0.0836%

  Validation       282--354       1,248,498       19.62%          762       0.0610%

  Testing          355--743       1,293,523       20.33%        4,258       0.3292%

  **Total**      **1--743**   **6,362,620**     **100%**    **8,213**   **0.1291%**
  ---------------------------------------------------------------------------------

### Temporal overlap verification

All overlap checks returned zero: - Training / validation overlapping
steps: `0` - Training / testing overlapping steps: `0` - Validation /
testing overlapping steps: `0`

### Important observation: changing fraud prevalence

The fraud rate differs across chronological partitions:

-   Training: 0.0836%
-   Validation: 0.0610%
-   Testing: 0.3292%

The test fraud rate is approximately 3.94 times the training fraud rate.
This indicates a temporal change in fraud prevalence in this dataset and
may affect model generalization. Results must be reported separately for
each partition.

## 6. Predictor and target separation

The target is `isFraud`. Raw predictors are created by excluding the
target, the existing fraud flag and raw account identifiers:

``` python
excluded_columns = [
    "isFraud",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest"
]

X_train = train_df.drop(columns=excluded_columns)
y_train = train_df["isFraud"]

X_val = val_df.drop(columns=excluded_columns)
y_val = val_df["isFraud"]

X_test = test_df.drop(columns=excluded_columns)
y_test = test_df["isFraud"]
```

Resulting shapes:

  Partition    X shape          y shape
  ------------ ---------------- --------------
  Training     `(3820599, 7)`   `(3820599,)`
  Validation   `(1248498, 7)`   `(1248498,)`
  Testing      `(1293523, 7)`   `(1293523,)`

The raw account identifiers remain available in `train_df`, `val_df` and
`test_df` for later behavioral feature engineering.

## 7. Preprocessing pipeline

The initial pipeline uses: - `StandardScaler` for numerical features -
`OneHotEncoder` for the categorical feature `type` - `ColumnTransformer`
to apply the appropriate transformation to each feature group

### Numerical and categorical features

``` python
NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

CATEGORICAL_FEATURES = ["type"]
```

### Reusable implementation

The implementation is stored in `src/preprocessing.py`:

``` python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

CATEGORICAL_FEATURES = ["type"]

EXCLUDED_COLUMNS = [
    "isFraud",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
]


def create_preprocessor():
    """Create the preprocessing pipeline."""

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype="float32",
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
```

### Fit and transform policy

The preprocessor is fitted on training data only:

``` python
preprocessor.fit(X_train)
```

Validation and test data must be transformed with this same fitted
preprocessor. Do not call `fit` or `fit_transform` on validation or test
data.

This prevents information from those partitions from influencing the
learned scaling statistics or category encoding.

## 8. Pipeline verification

The encoder learned these five categories from training data:

``` text
CASH_IN
CASH_OUT
DEBIT
PAYMENT
TRANSFER
```

The fitted preprocessor produces 11 output features: - 6 standardized
numerical features - 5 one-hot encoded transaction-type features

Feature names:

``` text
num__step
num__amount
num__oldbalanceOrg
num__newbalanceOrig
num__oldbalanceDest
num__newbalanceDest
cat__type_CASH_IN
cat__type_CASH_OUT
cat__type_DEBIT
cat__type_PAYMENT
cat__type_TRANSFER
```

A sample of 1,000 training rows was transformed successfully: - Original
sample shape: `(1000, 7)` - Processed sample shape: `(1000, 11)` -
Output format: SciPy CSR sparse matrix

A sample of 1,000 validation rows was also transformed successfully with
shape `(1000, 11)` using `transform()`.

The sparse output avoids unnecessarily storing all zero values from
one-hot encoded features.

## 9. Data leakage and imbalance safeguards

The workflow follows these safeguards:

1.  Split chronologically before fitting preprocessing transformations.
2.  Keep all transactions with the same `step` in one partition.
3.  Exclude `isFraud` from model inputs.
4.  Exclude `isFlaggedFraud` from model inputs.
5.  Fit the scaler and encoder on training data only.
6.  Use the fitted preprocessor to transform validation and test data.
7.  Do not apply oversampling or SMOTE before splitting.
8.  Compare imbalance-handling strategies using training and validation
    data; reserve the test set for final evaluation.

## 10. Known limitations and decisions for later modules

-   The chronological test period has a substantially higher fraud
    prevalence than the training period. This must be considered when
    interpreting results.
-   This is an initial shared preprocessing configuration. Different
    algorithms may require different preprocessing choices.
-   Scaling is useful for scale-sensitive algorithms, but tree-based
    models often do not require standardized numerical inputs.
-   The usefulness and operational availability of balance features
    should be reviewed in the context of the selected post-transaction
    detection scenario.
-   Raw account identifiers are excluded from initial model inputs but
    retained for possible historical behavioral features.
-   Behavioral features must use only information available at
    prediction time. Since `step` has hourly granularity, feature
    engineering must not assume that the within-hour order of
    transactions is known.

## 11. Next step --- Module 3: Feature Engineering

The next module will create behavioral features from sender and receiver
transaction histories, such as: - Historical transaction counts -
Historical transaction amounts - Account activity patterns - Sender and
receiver behavior

All historical aggregates must respect chronological order and avoid
future information.

------------------------------------------------------------------------

**Module 2 status:** Initial chronological splitting, feature selection,
preprocessing pipeline construction and sample transformation have been
completed and verified.
