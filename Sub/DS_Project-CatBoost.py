#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Preprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

# Modelling
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor, Pool

path = Path("mdsb-2023")


# In[9]:


def train_test_split_temporal(X, y, delta_threshold="60 days"):

    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = X["date"] <= cutoff_date
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid


# In[2]:


def add_lags(X, cols_to_lag=["t", "u", "vv", "nnuage4"], lag_list=[2, -24, -2]):
    X = X.copy()

    feature_columns = [col for col in X.columns if col in cols_to_lag]

    for lag in lag_list:
        lag_columns = [f"{col}_lag{lag}" for col in feature_columns]
        X[lag_columns] = X[feature_columns].shift(periods=lag, axis=0)
        X[lag_columns] = (
            X[lag_columns]
            .interpolate(method="linear")
            .interpolate(method="bfill")
            .interpolate(method="ffill")
        )

    return X


def add_moving_average(
    X, cols_to_ma=["t", "u", "vv", "nnuage4"], window_list=[24 * 7, 24], centered=True
):
    X = X.copy()

    feature_columns = [col for col in X.columns if col in cols_to_ma]

    for w in window_list:
        ma_columns = [f"{col}_ma{w}" for col in feature_columns]
        X[ma_columns] = X[feature_columns].rolling(window=w, center=centered).mean()
        X[ma_columns] = (
            X[ma_columns]
            .interpolate(method="linear")
            .interpolate(method="bfill")
            .interpolate(method="ffill")
        )

    return X


# ### Define pipeline functions

# In[3]:


def _encode_dates(X, col_name="date"):
    X = X.copy()

    X["month"] = X[col_name].dt.month
    X["weekday"] = X[col_name].dt.weekday
    X["hour"] = X[col_name].dt.hour

    X[["month", "weekday", "hour"]] = X[["month", "weekday", "hour"]].astype("category")

    return X.drop(columns=[col_name])


# In[4]:


def _encode_covid(X, col_name="date"):
    X = X.copy()

    # Create masks for lockdown dates
    lockdown_1 = (X["date"] >= "2020-10-17") & (X["date"] <= "2020-12-14")

    lockdown_2 = (X["date"] >= "2020-12-15") & (X["date"] <= "2021-02-26")

    lockdown_3 = (X["date"] >= "2021-02-27") & (X["date"] <= "2021-05-02")

    X["Covid"] = 0
    X.loc[lockdown_1 | lockdown_2 | lockdown_3, "Covid"] = 1

    return X


# In[5]:


def _merge_external_data(X, include_lags=True, include_ma=True):
    to_keep = [
        "date",
        "hnuage4",
        "t",
        "ctype4",
        "nnuage4",
        "u",
        "etat_sol",
        "perssfrai",
        "tx12",
        "cm",
        "tn12",
        "tend24",
        "vv",
        "rafper",
        "rr24",
        "hnuage2",
        "td",
        "rr3",
        "hnuage3",
        "hnuage1",
    ]

    ext_data = pd.read_csv(path / "external_data.csv", parse_dates=["date"])[to_keep]

    ext_data.drop(columns=ext_data.columns[ext_data.isna().sum() > 1000])

    full_date_range = pd.date_range(
        start=np.min([data.date.min(), test.date.min()]),
        end=np.max([data.date.max(), test.date.max()]),
        freq="H",
    )

    full_date_range = pd.DataFrame({"date": full_date_range})

    ext_data = full_date_range.merge(ext_data, on="date", how="left")

    columns_to_interpolate = ext_data.drop(columns="date").columns
    ext_data[columns_to_interpolate] = (
        ext_data[columns_to_interpolate]
        .interpolate(method="polynomial", order=3)
        .interpolate(method="bfill")
        .interpolate(method="ffill")
    )

    if include_lags:
        ext_data = add_lags(ext_data)

    if include_ma:
        ext_data = add_moving_average(ext_data)

    to_drop = [
        "vv_lag2",
        "t_lag-24",
        "u_lag-24",
        "vv_lag-24",
        "tx12",
        "nnuage4_lag-2",
        "etat_sol",
        "vv_lag-2",
        "u",
        "nnuage4",
        "ctype4",
        "t",
        "u_lag2",
        "vv_ma24",
        "t_lag2",
        "tend24",
        "u_ma24",
        "nnuage4_ma24",
        "vv_ma168",
        "cm",
        "hnuage1",
        "hnuage3",
        "rr3",
        "td",
        "hnuage2",
        "rr24",
        "rafper",
        "vv",
        "u_ma168",
        "hnuage4",
    ]

    ext_data.drop(columns=to_drop, inplace=True)

    X = X.copy()

    X["date"] = X["date"].astype("datetime64[ns]")
    ext_data["date"] = ext_data["date"].astype("datetime64[ns]")

    X["orig_index"] = np.arange(X.shape[0])

    X = pd.merge_asof(X.sort_values("date"), ext_data.sort_values("date"), on="date")

    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]

    return X


# In[6]:


def _gas_price_encoder(X):
    X = X.copy()
    X["gas_price"] = 1

    gas_prices = np.array(
        [
            1.22,
            1.21,
            1.22,
            1.27,
            1.31,
            1.36,
            1.4,
            1.39,
            1.4,
            1.43,
            1.45,
            1.45,
            1.46,
            1.56,
        ]
    )

    years = [
        2020,
        2020,
        2020,
        2020,
        2021,
        2021,
        2021,
        2021,
        2021,
        2021,
        2021,
        2021,
        2021,
        2021,
    ]

    months = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for i, price in enumerate(gas_prices):
        X.loc[
            (X.date.dt.month == months[i]) & (X.date.dt.year == years[i]), "gas_price"
        ] = price

    return X


# ## Import main dataset

# In[7]:


data = pd.read_parquet(path / "train.parquet")
test = pd.read_parquet(path / "final_test.parquet")

targets = ["bike_count", "log_bike_count"]


# In[8]:


data.drop(
    columns=[
        "site_name",
        "counter_id",
        "site_id",
        "counter_installation_date",
        "coordinates",
        "counter_technical_id",
    ],
    inplace=True,
)

test.drop(
    columns=[
        "site_name",
        "counter_id",
        "site_id",
        "counter_installation_date",
        "coordinates",
        "counter_technical_id",
    ],
    inplace=True,
)


# ## Model

# In[10]:


X, y = data.drop(columns=targets), data["log_bike_count"]

target_name = "log_bike_count"

data_merger = FunctionTransformer(_merge_external_data, validate=False)
covid_encoder = FunctionTransformer(_encode_covid, validate=False)
gas_encoder = FunctionTransformer(_gas_price_encoder, validate=False)
date_encoder = FunctionTransformer(_encode_dates, validate=False)

date_cols = _encode_dates(X[["date"]]).columns.tolist()
categorical_cols = ["counter_name"] + date_cols


# In[11]:


# {'learning_rate': 0.16263414906434522, 'n_estimators': 633}
best_params = {
    "learning_rate": 0.16,
    "max_depth": 8,
    "n_estimators": 630,
    "subsample": 0.8,
    "od_pval": 1e-5,
}

regressor = CatBoostRegressor(**best_params)

pipe = Pipeline(
    [
        ("merge external", data_merger),
        ("gas prices encoder", gas_encoder),
        ("covid encoder", covid_encoder),
        ("date encoder", date_encoder),
        ("regressor", regressor),
    ]
)

# # Submission

# In[ ]:


val_pool = Pool(
    _encode_dates(_encode_covid(_gas_price_encoder(_merge_external_data(X)))),
    label=y,
    cat_features=categorical_cols,
)
pipe.fit(
    X,
    y,
    regressor__cat_features=categorical_cols,
    regressor__early_stopping_rounds=70,
    regressor__eval_set=val_pool,
)

prediction = pipe.predict(test)
prediction[prediction < 0] = 0


# In[ ]:


submission = pd.DataFrame({"log_bike_count": prediction})

submission = pd.DataFrame({"Id": test.index, "log_bike_count": prediction})

submission.to_csv("submission.csv", index=False)
