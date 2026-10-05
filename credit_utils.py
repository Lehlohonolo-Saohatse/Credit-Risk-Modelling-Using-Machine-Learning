"""Shared feature definitions and preprocessing for the credit risk project.

Imported by both the notebook and the Streamlit app so that training and
inference apply exactly the same transformations.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC = ['Age', 'Job', 'Credit amount', 'Duration', 'Credit per month']
CATEGORICAL = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
RAW_FEATURES = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
                'Checking account', 'Credit amount', 'Duration', 'Purpose']

# Cost matrix from the original Statlog documentation:
# approving a bad applicant costs 5x as much as rejecting a good one.
COST_FN = 5   # predict good, actually bad
COST_FP = 1   # predict bad, actually good


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill 'no account' for missing balances and add the monthly-burden feature."""
    out = df.copy()
    for col in ['Saving accounts', 'Checking account']:
        out[col] = out[col].fillna('no account').astype(str)
    out['Credit per month'] = out['Credit amount'] / out['Duration']
    return out[NUMERIC + CATEGORICAL]


def make_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Feature engineering + encoding as a single sklearn step (no leakage)."""
    num = StandardScaler() if scale_numeric else 'passthrough'
    ct = ColumnTransformer([
        ('num', num, NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ])
    ct.set_output(transform='pandas')
    return ct


def feature_step() -> FunctionTransformer:
    return FunctionTransformer(add_features, validate=False)
