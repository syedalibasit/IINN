# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:03:19 2026

@author: 2687492Z
"""

"""
IINN Reviewer Response Experiment Runner - IINN + Stable DNN Version
====================================================================

Purpose
-------
Runs the reviewer-response experiment suite for the IINN paper using:
- 3GPP analytical baseline, simplified/calibrated
- Random Forest
- Extra Trees
- HistGradientBoosting
- XGBoost, if installed
- LightGBM, if installed
- Stabilized DNN baseline with target normalization
- IINN model supplied by user

Final model feature set
-----------------------
The training feature set is exactly:

1.  User_Downtilt
2.  Transmitter_Downtilt
3.  Half-power Vertical Beamwidth
4.  Vertical_Attenuation
5.  User_Azimuth
6.  Transmitter_Azimuth
7.  Half-power Horizontal Beamwidth
8.  Horizontal_Attenuation
9.  Efficiency
10. 3D_Distance
11. Frequency
12. Street_Width
13. Building_Height
14. User_Height
15. Transmitter_Height
16. LoS_Constant
17. NLoS_Constant
18. Max_Transmitter_Power

Target:
- RSRP

Metadata columns, used for splitting/plots if available:
- Transmitter
- User_X
- User_Y
- Clutter Class
- City
- Dataset_Name

Important
---------
- Categorical variables are NOT used as training features in this version.
- This is intentional because IINN expects a strict raw numerical input order.
- If you include User_X/User_Y as features, black-box models may spatially memorize.
  This version avoids that by using the same physically meaningful feature set for all models.

Install packages
----------------
pip install pandas numpy scikit-learn matplotlib openpyxl xgboost lightgbm tensorflow

If TensorFlow, XGBoost, or LightGBM are unavailable, the script skips the affected models.
"""

# ============================================================
# SECTION 1: IMPORTS AND GLOBAL SETTINGS
# ============================================================

import re
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, callbacks, models, optimizers
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_JOBS = -1
SAVE_ALL_PREDICTIONS = True

TARGET_COL = "RSRP"
TRANSMITTER_COL = "Transmitter"
SITE_COL = "Site_ID"
SECTOR_COL = "Sector_ID"
SITE_SECTOR_COL = "Site_Sector_ID"

FEATURE_COLS = [
    "User_Downtilt",
    "Transmitter_Downtilt",
    "Half-power Vertical Beamwidth",
    "Vertical_Attenuation",
    "User_Azimuth",
    "Transmitter_Azimuth",
    "Half-power Horizontal Beamwidth",
    "Horizontal_Attenuation",
    "Efficiency",
    "3D_Distance",
    "Frequency",
    "Street_Width",
    "Building_Height",
    "User_Height",
    "Transmitter_Height",
    "LoS_Constant",
    "NLoS_Constant",
    "Max_Transmitter_Power",
]

COLUMN_ALIASES = {
    "Frequency (MHz)": "Frequency",
    "Frequency Band.Reference Frequency (MHz)": "Frequency",
    "frequency": "Frequency",
    "Freq": "Frequency",
    "Transmitter_Mechanical_Downtilt": "Transmitter_Downtilt",
    "Transmitter_Mechnical_Downtilt": "Transmitter_Downtilt",
    "Antenna_Gain_(dBi)": "Antenna_Gain",
    "Antenna Gain": "Antenna_Gain",
    "Clutter_Class": "Clutter Class",
    "Clutter class": "Clutter Class",
    "RSRP (dBm)": "RSRP",
    "SS-RSRP": "RSRP",
    "(SS-)RSRP (DL) (dBm)": "RSRP",
}

# ============================================================
# SECTION 2: USER INPUT HELPERS
# ============================================================

def ask_path(prompt: str, must_exist: bool = True) -> Path:
    while True:
        p = Path(input(prompt).strip().strip('"').strip("'"))
        if not must_exist or p.exists():
            return p
        print(f"Path does not exist: {p}")


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(name))


def make_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

# ============================================================
# SECTION 3: METRICS
# ============================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def mbe(y_true, y_pred):
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def p95_abs_error(y_true, y_pred):
    return float(np.percentile(np.abs(np.asarray(y_pred) - np.asarray(y_true)), 95))


def median_abs_error_custom(y_true, y_pred):
    return float(np.median(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def nrmse_range(y_true, y_pred):
    y_true = np.asarray(y_true)
    denom = np.max(y_true) - np.min(y_true)
    return float(rmse(y_true, y_pred) / denom) if denom != 0 else np.nan


def pearson_corr(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "Mean_Bias_Error": mbe(y_true, y_pred),
        "P95_Absolute_Error": p95_abs_error(y_true, y_pred),
        "Median_Absolute_Error": median_abs_error_custom(y_true, y_pred),
        "NRMSE_Range": nrmse_range(y_true, y_pred),
        "Pearson_Correlation": pearson_corr(y_true, y_pred),
    }


def ood_degradation(id_rmse_value, ood_rmse_value):
    if id_rmse_value is None or np.isnan(id_rmse_value) or id_rmse_value == 0:
        return np.nan
    return float((ood_rmse_value - id_rmse_value) / id_rmse_value * 100.0)


def compute_aulc(df_results: pd.DataFrame, group_cols=None):
    if group_cols is None:
        group_cols = ["Model", "Experiment_Group"]
    rows = []
    if "Train_Fraction" not in df_results.columns:
        return pd.DataFrame()
    scarce = df_results.dropna(subset=["Train_Fraction", "RMSE"])
    for keys, g in scarce.groupby(group_cols):
        g = g.sort_values("Train_Fraction")
        x = g["Train_Fraction"].values.astype(float)
        y = g["RMSE"].values.astype(float)
        area = float(np.trapz(y, x)) if len(x) >= 2 else np.nan
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["AULC_RMSE"] = area
        rows.append(row)
    return pd.DataFrame(rows)

# ============================================================
# SECTION 4: DATA LOADING AND CLEANING
# ============================================================

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {}
    for old, new in COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            rename_map[old] = new
    df = df.rename(columns=rename_map)
    return df


def load_excel_dataset(path: Path, dataset_name: str, city: str, frequency_mhz: float, propagation_model: str = "3GPP") -> pd.DataFrame:
    df = pd.read_excel(path)
    df = standardise_columns(df)
    df["Dataset_Name"] = dataset_name
    df["City"] = city
    df["Frequency_MHz_Metadata"] = frequency_mhz
    df["Propagation_Model"] = propagation_model
    if "Frequency" not in df.columns:
        df["Frequency"] = frequency_mhz
    return df


def infer_site_from_transmitter(value):
    s = str(value)
    if "_" in s:
        return s.split("_")[0]
    return s


def infer_sector_from_transmitter(value):
    s = str(value)
    if "_" in s:
        return s.split("_")[-1]
    return s


def add_site_sector_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TRANSMITTER_COL in df.columns:
        df[SITE_COL] = df[TRANSMITTER_COL].apply(infer_site_from_transmitter)
        df[SECTOR_COL] = df[TRANSMITTER_COL].apply(infer_sector_from_transmitter)
        df[SITE_SECTOR_COL] = df[TRANSMITTER_COL].astype(str)
    else:
        print(f"WARNING: '{TRANSMITTER_COL}' column not found. Creating synthetic Site/Sector IDs.")
        df[SITE_COL] = "UnknownSite"
        df[SECTOR_COL] = "UnknownSector"
        df[SITE_SECTOR_COL] = "UnknownSite_UnknownSector"
    return df


def validate_required_columns(df: pd.DataFrame, dataset_name: str):
    required = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nDataset '{dataset_name}' is missing required columns:")
        for c in missing:
            print(f"  - {c}")
        print("\nAvailable columns:")
        print(list(df.columns))
        raise ValueError(f"Missing required columns in {dataset_name}: {missing}")


def prepare_xy(df: pd.DataFrame):
    dataset_name = df["Dataset_Name"].iloc[0] if "Dataset_Name" in df.columns and len(df) else "Unknown"
    validate_required_columns(df, dataset_name)
    work = df.dropna(subset=[TARGET_COL]).copy()
    X = work[FEATURE_COLS].copy()
    y = work[TARGET_COL].astype(float).copy()
    return X, y


def make_sklearn_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

# ============================================================
# SECTION 5: 3GPP ANALYTICAL BASELINE
# ============================================================

def three_gpp_uma_pathloss_los(distance_3d_m, frequency_mhz):
    """
    Simplified 3GPP-like UMa LOS pathloss baseline:
    PL_LOS = 28.0 + 22 log10(d_3D) + 20 log10(fc_GHz)

    Replace this with your exact IINN/3GPP equation before final paper reporting.
    """
    d = np.maximum(np.asarray(distance_3d_m, dtype=float), 1.0)
    fc_ghz = np.maximum(np.asarray(frequency_mhz, dtype=float) / 1000.0, 0.1)
    return 28.0 + 22.0 * np.log10(d) + 20.0 * np.log10(fc_ghz)


def analytical_3gpp_predict(df: pd.DataFrame):
    pathloss = three_gpp_uma_pathloss_los(df["3D_Distance"], df["Frequency"])
    tx_power = df["Max_Transmitter_Power"].astype(float).values
    vertical_att = df["Vertical_Attenuation"].astype(float).values
    horizontal_att = df["Horizontal_Attenuation"].astype(float).values
    efficiency = df["Efficiency"].astype(float).values
    return tx_power + efficiency - pathloss - vertical_att - horizontal_att


def fit_calibrated_analytical_offset(y_train, y_pred_train):
    return float(np.mean(np.asarray(y_train) - np.asarray(y_pred_train)))

# ============================================================
# SECTION 6: MODEL BUILDERS - ML BASELINES
# ============================================================

def build_sklearn_models():
    models_dict = {}

    models_dict["RandomForest"] = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    models_dict["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    models_dict["HistGradientBoosting"] = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.04,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE
    )

    if XGBOOST_AVAILABLE:
        models_dict["XGBoost"] = XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        )

    if LIGHTGBM_AVAILABLE:
        models_dict["LightGBM"] = LGBMRegressor(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=-1,
            force_col_wise=True
        )

    return models_dict

# ============================================================
# SECTION 7: MODEL BUILDERS - STABLE DNN
# ============================================================

def build_stable_dnn(input_dim: int):
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is not available.")
    tf.keras.backend.clear_session()
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        layers.BatchNormalization(),
        layers.Dropout(0.20),
        layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        layers.BatchNormalization(),
        layers.Dropout(0.20),
        layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss="mse",
        metrics=["mae"]
    )
    return model


def train_predict_dnn(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, out_model_dir: Path, exp_name: str):
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train_raw)
    X_val = scaler_x.transform(X_val_raw)
    X_test = scaler_x.transform(X_test_raw)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-8)
    y_train_n = (y_train.values - y_mean) / y_std
    y_val_n = (y_val.values - y_mean) / y_std

    model = build_stable_dnn(X_train.shape[1])
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
        callbacks.TerminateOnNaN(),
    ]

    start = time.perf_counter()
    history = model.fit(
        X_train, y_train_n,
        validation_data=(X_val, y_val_n),
        epochs=300,
        batch_size=64,
        verbose=0,
        callbacks=cb
    )
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred_n = model.predict(X_test, verbose=0).ravel()
    pred_time = time.perf_counter() - start
    y_pred = y_pred_n * y_std + y_mean

    history_df = pd.DataFrame(history.history)
    history_df.to_excel(out_model_dir / f"{safe_name(exp_name)}_DNN_training_history.xlsx", index=False)
    save_training_curve(history_df, out_model_dir / f"{safe_name(exp_name)}_DNN_training_curve.png", f"DNN Training Curve - {exp_name}")

    return model, scaler_x, y_pred, train_time, pred_time, history_df

# ============================================================
# SECTION 8: MODEL BUILDERS - IINN
# ============================================================

def lambda_2(inputs):
    epsilon = 1e-8
    return tf.where(
        tf.equal(inputs, 0),
        tf.zeros_like(inputs),
        tf.divide(tf.math.log(inputs), tf.math.log(10.0) + epsilon)
    )


def build_iinn_with_trace():
    w = 1
    ker_initializer = tf.keras.initializers.Constant(1.0)
    bi_initializer = tf.keras.initializers.Constant(1.0)

    trace = {}

    inputs_1 = tf.keras.layers.Input(shape=[1], name='UE_Tilt')
    output_1 = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=True, name='UE_Tilt_dense_layer')(inputs_1)
    trace["output_1"] = output_1

    inputs_2 = tf.keras.layers.Input(shape=[1], name='BS_Tilt')
    output_2 = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=True, name='BS_Tilt_dense_layer')(inputs_2)
    trace["output_2"] = output_2

    inputs_3 = tf.keras.layers.Input(shape=[1], name='bv')
    lambda_bv = tf.keras.layers.Lambda(lambda_2, name="lambda_bv")(inputs_3)
    trace["lambda_bv"] = lambda_bv
    output_3 = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, use_bias=False, name='bv_dense_layer')(inputs_3)
    trace["output_3"] = output_3

    inputs_4 = tf.keras.layers.Input(shape=[1], name='Av')
    output_4 = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=True, name='Av_dense_layer')(inputs_4)
    trace["output_4"] = output_4

    subtracted_12 = tf.keras.layers.Subtract(name="subtracted_12")([output_1, output_2])
    trace["subtracted_12"] = subtracted_12

    div_12 = tf.keras.layers.Lambda(lambda x: x[0] / (x[1] + 1e-8), name="div_12")([subtracted_12, output_3])
    trace["div_12"] = div_12

    sq_12 = tf.keras.layers.Lambda(lambda x: tf.math.square(x), name="sq_12")(div_12)
    trace["sq_12"] = sq_12

    minimum_14 = tf.keras.layers.Minimum(name="minimum_14")([sq_12, output_4])
    trace["minimum_14"] = minimum_14

    inputs_5 = tf.keras.layers.Input(shape=[1], name='UE_Azimuth')
    output_5 = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=True, name='UE_Azimuth_dense_layer')(inputs_5)
    trace["output_5"] = output_5

    inputs_6 = tf.keras.layers.Input(shape=[1], name='BS_Azimuth')
    output_6 = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=True, name='BS_Azimuth_dense_layer')(inputs_6)
    trace["output_6"] = output_6

    inputs_7 = tf.keras.layers.Input(shape=[1], name='bh')
    lambda_bh = tf.keras.layers.Lambda(lambda_2, name="lambda_bh")(inputs_7)
    trace["lambda_bh"] = lambda_bh
    output_7 = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=False, name='bh_dense_layer')(inputs_7)
    trace["output_7"] = output_7

    inputs_8 = tf.keras.layers.Input(shape=[1], name='Ah')
    output_8 = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=True, name='Ah_dense_layer')(inputs_8)
    trace["output_8"] = output_8

    subtracted_56 = tf.keras.layers.Subtract(name="subtracted_56")([output_5, output_6])
    trace["subtracted_56"] = subtracted_56

    div_56 = tf.keras.layers.Lambda(lambda x: x[0] / (x[1] + 1e-8), name="div_56")([subtracted_56, output_7])
    trace["div_56"] = div_56

    sq_56 = tf.keras.layers.Lambda(lambda x: tf.math.square(x), name="sq_56")(div_56)
    trace["sq_56"] = sq_56

    minimum_58 = tf.keras.layers.Minimum(name="minimum_58")([sq_56, output_8])
    trace["minimum_58"] = minimum_58

    output_18_input = tf.keras.layers.Add(name="output_18_input")([minimum_14, minimum_58])
    trace["output_18_input"] = output_18_input

    output_18 = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, use_bias=False, name='output_18_dense_layer')(output_18_input)
    trace["output_18"] = output_18

    inputs_9 = tf.keras.layers.Input(shape=[1], name='zeta')
    lambda_9 = tf.keras.layers.Lambda(lambda_2, name="lambda_9")(inputs_9)
    trace["lambda_9"] = lambda_9

    output_9 = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=True, name='zeta_dense_layer')(lambda_9)
    trace["output_9"] = output_9

    const_log4pi = np.log10(4 * np.pi)
    log_4pi = tf.keras.layers.Lambda(lambda x: tf.ones_like(x) * const_log4pi, name="log_4pi_constant")(lambda_bv)
    trace["log_4pi_constant"] = log_4pi

    add_bvh = tf.keras.layers.Add(name="add_log_bv_log_bh")([lambda_bv, lambda_bh])
    sub4pi_bvh = tf.keras.layers.Subtract(name="sub_log4pi_logbvh")([log_4pi, add_bvh])
    trace["add_log_bv_log_bh"] = add_bvh
    trace["sub_log4pi_logbvh"] = sub4pi_bvh

    output_4pi_bvh = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=True, name='4pi_bvh_dense_layer')(sub4pi_bvh)
    trace["output_4pi_bvh"] = output_4pi_bvh

    add_910 = tf.keras.layers.Add(name="add_910")([output_9, output_4pi_bvh])
    trace["add_910"] = add_910

    inputs_W = tf.keras.layers.Input(shape=[1], name='Width')
    lambda_W = tf.keras.layers.Lambda(lambda_2, name="lambda_W")(inputs_W)
    trace["lambda_W"] = lambda_W

    output_W = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=True, name='Width_dense_layer')(lambda_W)
    trace["output_W"] = output_W

    inputs_hob = tf.keras.layers.Input(shape=[1], name='Building_height')
    lambda_hob = tf.keras.layers.Lambda(lambda_2, name="lambda_hob")(inputs_hob)
    trace["lambda_hob"] = lambda_hob

    output_hob = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=True, name='Object_height_dense_layer')(lambda_hob)
    trace["output_hob"] = output_hob

    inputs_hue = tf.keras.layers.Input(shape=[1], name='UE_height')
    output_hue = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, use_bias=False, name='UE_height_dense_layer')(inputs_hue)
    trace["output_hue"] = output_hue

    inputs_hbs = tf.keras.layers.Input(shape=[1], name='BS_height')
    lambda_hbs = tf.keras.layers.Lambda(lambda_2, name="lambda_hbs")(inputs_hbs)
    trace["lambda_hbs"] = lambda_hbs

    output_hbs = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=True, name='BS_height_dense_layer')(lambda_hbs)
    trace["output_hbs"] = output_hbs

    inputs_c2 = tf.keras.layers.Input(shape=[1], name='NLoS_constant')
    output_c2 = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, use_bias=False, name='c2_dense_layer')(inputs_c2)
    trace["output_c2"] = output_c2

    inputs_Pt = tf.keras.layers.Input(shape=[1], name='Tx_Pwr')
    output_Pt = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=True, name='TX_Pwr_dense_layer')(inputs_Pt)
    trace["output_Pt"] = output_Pt

    inputs_d = tf.keras.layers.Input(shape=[1], name='Distance')
    lambda_d = tf.keras.layers.Lambda(lambda_2, name="lambda_d")(inputs_d)
    trace["lambda_d"] = lambda_d

    output_d = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, use_bias=False, name='Distance_dense_layer')(lambda_d)
    trace["output_d"] = output_d

    inputs_f = tf.keras.layers.Input(shape=[1], name='Frequency')
    lambda_f = tf.keras.layers.Lambda(lambda_2, name="lambda_f")(inputs_f)
    trace["lambda_f"] = lambda_f

    output_f = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=True, name='Frequency_dense_layer')(lambda_f)
    trace["output_f"] = output_f

    inputs_c1 = tf.keras.layers.Input(shape=[1], name='c1')
    output_c1 = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, use_bias=False, name='c1_dense_layer')(inputs_c1)
    trace["output_c1"] = output_c1

    LOS1 = tf.keras.layers.Add(name="LOS1")([output_d, output_f, output_c1])
    trace["LOS1"] = LOS1

    mul1_raw = tf.keras.layers.Multiply(name="mul1_raw")([lambda_d, lambda_hbs])
    trace["mul1_raw"] = mul1_raw

    mul1 = tf.keras.layers.Dense(w, activation='linear', kernel_initializer=ker_initializer, use_bias=False, name='mul1_dense_layer')(mul1_raw)
    trace["mul1"] = mul1

    sub1 = tf.keras.layers.Subtract(name="sub1")([output_f, output_W])
    trace["sub1"] = sub1

    add1 = tf.keras.layers.Add(name="add1")([sub1, output_hob])
    trace["add1"] = add1

    sub2 = tf.keras.layers.Subtract(name="sub2")([add1, output_hbs])
    trace["sub2"] = sub2

    sub3 = tf.keras.layers.Subtract(name="sub3")([sub2, output_hue])
    trace["sub3"] = sub3

    sub4 = tf.keras.layers.Subtract(name="sub4")([sub3, mul1])
    trace["sub4"] = sub4

    add2 = tf.keras.layers.Add(name="add2")([sub4, output_d])
    trace["add2"] = add2

    NLOS = tf.keras.layers.Add(name="NLOS")([add2, output_c2])
    trace["NLOS"] = NLOS

    max_PL = tf.keras.layers.Maximum(name="max_PL")([LOS1, NLOS])
    trace["max_PL"] = max_PL

    max_PL_Gain = tf.keras.layers.Add(name="max_PL_Gain")([output_18, add_910, max_PL])
    trace["max_PL_Gain"] = max_PL_Gain

    Rx_Pwr = tf.keras.layers.Subtract(name="Rx_Pwr")([output_Pt, max_PL_Gain])
    trace["Rx_Pwr"] = Rx_Pwr

    output_Rx_Pwr = tf.keras.layers.Dense(1, activation='linear', kernel_initializer=ker_initializer, bias_initializer=bi_initializer, use_bias=True, name='Rx_Pwr_dense_layer')(Rx_Pwr)
    trace["output_Rx_Pwr"] = output_Rx_Pwr

    model = tf.keras.Model(
        inputs=[inputs_1, inputs_2, inputs_3, inputs_4,
                inputs_5, inputs_6, inputs_7, inputs_8,
                inputs_9, inputs_d, inputs_f,
                inputs_W, inputs_hob, inputs_hue, inputs_hbs,
                inputs_c1, inputs_c2, inputs_Pt],
        outputs=output_Rx_Pwr,
        name="IINN_traceable"
    )

    return model, trace


def train_predict_iinn(X_train, y_train, X_val, y_val, X_test, out_model_dir: Path, exp_name: str):
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is not available; IINN cannot run.")

    tf.keras.backend.clear_session()
    model, trace_tensors = build_iinn_with_trace()

    # Freeze beamwidth layers as in your provided setup.
    model.get_layer("bh_dense_layer").trainable = False
    model.get_layer("bv_dense_layer").trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="mse"
    )

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
        callbacks.TerminateOnNaN(),
    ]

    start = time.perf_counter()
    history = model.fit(
        [X_train.iloc[:, i].values.astype(np.float32) for i in range(X_train.shape[1])],
        y_train.values.astype(np.float32),
        validation_data=(
            [X_val.iloc[:, i].values.astype(np.float32) for i in range(X_val.shape[1])],
            y_val.values.astype(np.float32)
        ),
        epochs=150,
        batch_size=32,
        verbose=0,
        callbacks=cb
    )
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(
        [X_test.iloc[:, i].values.astype(np.float32) for i in range(X_test.shape[1])],
        verbose=0
    ).ravel()
    pred_time = time.perf_counter() - start

    history_df = pd.DataFrame(history.history)
    history_df.to_excel(out_model_dir / f"{safe_name(exp_name)}_IINN_training_history.xlsx", index=False)
    save_training_curve(history_df, out_model_dir / f"{safe_name(exp_name)}_IINN_training_curve.png", f"IINN Training Curve - {exp_name}")

    # Save trained weights summary.
    weight_rows = []
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            for wi, w in enumerate(weights):
                weight_rows.append({
                    "Layer": layer.name,
                    "Weight_Index": wi,
                    "Shape": str(w.shape),
                    "Mean": float(np.mean(w)),
                    "Std": float(np.std(w)),
                    "Min": float(np.min(w)),
                    "Max": float(np.max(w)),
                    "Trainable": layer.trainable,
                })
    if weight_rows:
        pd.DataFrame(weight_rows).to_excel(out_model_dir / f"{safe_name(exp_name)}_IINN_weights_summary.xlsx", index=False)

    return model, y_pred, train_time, pred_time, history_df

# ============================================================
# SECTION 9: PLOTTING AND OUTPUTS
# ============================================================

def save_prediction_scatter(y_true, y_pred, out_path: Path, title: str):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.5)
    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("Original RSRP")
    plt.ylabel("Predicted RSRP")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_residual_plot(y_true, y_pred, out_path: Path, title: str):
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, s=8, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted RSRP")
    plt.ylabel("Residual: Predicted - Original")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_error_histogram(y_true, y_pred, out_path: Path, title: str):
    errors = np.asarray(y_pred) - np.asarray(y_true)
    plt.figure(figsize=(7, 5))
    plt.hist(errors, bins=50, alpha=0.8)
    plt.xlabel("Prediction Error: Predicted - Original")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_heatmap(df_pred: pd.DataFrame, x_col: str, y_col: str, value_col: str, out_path: Path, title: str):
    if not x_col or not y_col or x_col not in df_pred.columns or y_col not in df_pred.columns:
        return
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df_pred[x_col], df_pred[y_col], c=df_pred[value_col], s=8, alpha=0.9)
    plt.colorbar(sc, label=value_col)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_metric_barplot(df_results: pd.DataFrame, metric: str, out_path: Path, title: str):
    if metric not in df_results.columns or df_results.empty:
        return
    g = df_results.groupby("Model")[metric].mean().sort_values()
    plt.figure(figsize=(9, 5))
    g.plot(kind="bar")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_training_curve(history_df: pd.DataFrame, out_path: Path, title: str):
    if history_df is None or history_df.empty:
        return
    plt.figure(figsize=(7, 5))
    if "loss" in history_df.columns:
        plt.plot(history_df["loss"], label="Train loss")
    if "val_loss" in history_df.columns:
        plt.plot(history_df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_model_outputs(pred_df, y_true, y_pred, model_dir: Path, exp_name: str, model_name: str):
    name = safe_name(f"{exp_name}_{model_name}")
    pred_df.to_excel(model_dir / f"{name}_predictions.xlsx", index=False)
    save_prediction_scatter(y_true, y_pred, model_dir / f"{name}_prediction_scatter.png", f"{model_name} - {exp_name}")
    save_residual_plot(y_true, y_pred, model_dir / f"{name}_residual_plot.png", f"Residuals - {model_name} - {exp_name}")
    save_error_histogram(y_true, y_pred, model_dir / f"{name}_error_histogram.png", f"Error Histogram - {model_name} - {exp_name}")
    save_heatmap(pred_df, "User_X", "User_Y", "Original_RSRP", model_dir / f"{name}_original_rsrp_heatmap.png", f"Original RSRP - {model_name} - {exp_name}")
    save_heatmap(pred_df, "User_X", "User_Y", "Predicted_RSRP", model_dir / f"{name}_predicted_rsrp_heatmap.png", f"Predicted RSRP - {model_name} - {exp_name}")
    save_heatmap(pred_df, "User_X", "User_Y", "Error", model_dir / f"{name}_error_heatmap.png", f"Error Map - {model_name} - {exp_name}")


def save_permutation_importance(model, X_test, y_test, model_dir: Path, exp_name: str, model_name: str):
    try:
        r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error")
        imp = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance_Mean": r.importances_mean,
            "Importance_Std": r.importances_std,
        }).sort_values("Importance_Mean", ascending=False)
        name = safe_name(f"{exp_name}_{model_name}")
        imp.to_excel(model_dir / f"{name}_permutation_importance.xlsx", index=False)
        plt.figure(figsize=(8, 7))
        top = imp.head(18).iloc[::-1]
        plt.barh(top["Feature"], top["Importance_Mean"])
        plt.xlabel("Permutation Importance")
        plt.title(f"Permutation Importance - {model_name} - {exp_name}")
        plt.tight_layout()
        plt.savefig(model_dir / f"{name}_permutation_importance.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Permutation importance failed for {model_name}, {exp_name}: {e}")


def grouped_error_analysis(pred_df: pd.DataFrame, model_dir: Path, exp_name: str, model_name: str):
    rows = []
    group_cols = [
        "City", "Clutter Class", "Transmitter", SITE_COL, SECTOR_COL,
        "Transmitter_Height", "Transmitter_Azimuth", "Transmitter_Downtilt", "Frequency"
    ]
    for col in group_cols:
        if col not in pred_df.columns:
            continue
        for value, g in pred_df.groupby(col):
            if len(g) < 2:
                continue
            m = compute_metrics(g["Original_RSRP"], g["Predicted_RSRP"])
            rows.append({"Experiment": exp_name, "Model": model_name, "Group_Column": col, "Group_Value": value, "N": len(g), **m})
    if rows:
        out = pd.DataFrame(rows)
        out.to_excel(model_dir / f"{safe_name(exp_name)}_{model_name}_grouped_error_analysis.xlsx", index=False)

# ============================================================
# SECTION 10: TRAINING AND EVALUATION LOOP
# ============================================================

def train_predict_sklearn(model, X_train, y_train, X_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(X_test)
    pred_time = time.perf_counter() - start
    return model, pred, train_time, pred_time


def run_single_experiment(
    exp_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    models_to_run: list,
    train_fraction: float = None,
    experiment_group: str = None,
    validation_fraction: float = 0.15,
):
    print(f"\nRunning experiment: {exp_name}")
    exp_dir = make_dir(output_dir / "experiments" / safe_name(exp_name))

    train_df = train_df.dropna(subset=[TARGET_COL]).copy()
    test_df = test_df.dropna(subset=[TARGET_COL]).copy()

    if train_fraction is not None and train_fraction < 1.0:
        train_df = train_df.sample(frac=train_fraction, random_state=RANDOM_STATE).copy()

    if len(train_df) < 5:
        print(f"Skipping {exp_name}: too few training samples after fractioning.")
        return pd.DataFrame(), pd.DataFrame()

    train_part, val_part = train_test_split(train_df, test_size=validation_fraction, random_state=RANDOM_STATE)

    X_train_raw, y_train = prepare_xy(train_part)
    X_val_raw, y_val = prepare_xy(val_part)
    X_test_raw, y_test = prepare_xy(test_df)

    results = []
    prediction_dfs = []

    # 3GPP analytical baseline
    if "3GPP_Analytical" in models_to_run:
        model_name = "3GPP_Analytical"
        model_dir = make_dir(output_dir / "models" / model_name)
        start = time.perf_counter()
        y_train_base = analytical_3gpp_predict(train_part)
        offset = fit_calibrated_analytical_offset(y_train, y_train_base)
        train_time = time.perf_counter() - start
        start = time.perf_counter()
        y_pred = analytical_3gpp_predict(test_df) + offset
        pred_time = time.perf_counter() - start
        m = compute_metrics(y_test, y_pred)
        results.append({"Experiment": exp_name, "Experiment_Group": experiment_group or exp_name, "Model": model_name, "Train_Size": len(train_df), "Test_Size": len(test_df), "Train_Fraction": train_fraction, "Train_Time_sec": train_time, "Prediction_Time_sec": pred_time, "Prediction_Time_per_sample_ms": pred_time / max(len(test_df), 1) * 1000, **m})
        pred_df = test_df.copy()
        pred_df["Experiment"] = exp_name
        pred_df["Model"] = model_name
        pred_df["Original_RSRP"] = y_test.values
        pred_df["Predicted_RSRP"] = y_pred
        pred_df["Error"] = pred_df["Predicted_RSRP"] - pred_df["Original_RSRP"]
        prediction_dfs.append(pred_df)
        save_model_outputs(pred_df, y_test, y_pred, model_dir, exp_name, model_name)
        grouped_error_analysis(pred_df, model_dir, exp_name, model_name)

    # Sklearn models
    sklearn_models = build_sklearn_models()
    for model_name, estimator in sklearn_models.items():
        if model_name not in models_to_run:
            continue
        model_dir = make_dir(output_dir / "models" / model_name)
        pipe = Pipeline([("preprocess", make_sklearn_preprocessor()), ("model", estimator)])
        fitted, y_pred, train_time, pred_time = train_predict_sklearn(pipe, X_train_raw, y_train, X_test_raw)
        m = compute_metrics(y_test, y_pred)
        results.append({"Experiment": exp_name, "Experiment_Group": experiment_group or exp_name, "Model": model_name, "Train_Size": len(train_df), "Test_Size": len(test_df), "Train_Fraction": train_fraction, "Train_Time_sec": train_time, "Prediction_Time_sec": pred_time, "Prediction_Time_per_sample_ms": pred_time / max(len(test_df), 1) * 1000, **m})
        pred_df = test_df.copy()
        pred_df["Experiment"] = exp_name
        pred_df["Model"] = model_name
        pred_df["Original_RSRP"] = y_test.values
        pred_df["Predicted_RSRP"] = y_pred
        pred_df["Error"] = pred_df["Predicted_RSRP"] - pred_df["Original_RSRP"]
        prediction_dfs.append(pred_df)
        save_model_outputs(pred_df, y_test, y_pred, model_dir, exp_name, model_name)
        save_permutation_importance(fitted, X_test_raw, y_test, model_dir, exp_name, model_name)
        grouped_error_analysis(pred_df, model_dir, exp_name, model_name)

    # Stable DNN
    if "DNN" in models_to_run and TENSORFLOW_AVAILABLE:
        model_name = "DNN"
        model_dir = make_dir(output_dir / "models" / model_name)
        _, _, y_pred, train_time, pred_time, _ = train_predict_dnn(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, model_dir, exp_name)
        m = compute_metrics(y_test, y_pred)
        results.append({"Experiment": exp_name, "Experiment_Group": experiment_group or exp_name, "Model": model_name, "Train_Size": len(train_df), "Test_Size": len(test_df), "Train_Fraction": train_fraction, "Train_Time_sec": train_time, "Prediction_Time_sec": pred_time, "Prediction_Time_per_sample_ms": pred_time / max(len(test_df), 1) * 1000, **m})
        pred_df = test_df.copy()
        pred_df["Experiment"] = exp_name
        pred_df["Model"] = model_name
        pred_df["Original_RSRP"] = y_test.values
        pred_df["Predicted_RSRP"] = y_pred
        pred_df["Error"] = pred_df["Predicted_RSRP"] - pred_df["Original_RSRP"]
        prediction_dfs.append(pred_df)
        save_model_outputs(pred_df, y_test, y_pred, model_dir, exp_name, model_name)
        grouped_error_analysis(pred_df, model_dir, exp_name, model_name)

    # IINN
    if "IINN" in models_to_run and TENSORFLOW_AVAILABLE:
        model_name = "IINN"
        model_dir = make_dir(output_dir / "models" / model_name)
        _, y_pred, train_time, pred_time, _ = train_predict_iinn(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, model_dir, exp_name)
        m = compute_metrics(y_test, y_pred)
        results.append({"Experiment": exp_name, "Experiment_Group": experiment_group or exp_name, "Model": model_name, "Train_Size": len(train_df), "Test_Size": len(test_df), "Train_Fraction": train_fraction, "Train_Time_sec": train_time, "Prediction_Time_sec": pred_time, "Prediction_Time_per_sample_ms": pred_time / max(len(test_df), 1) * 1000, **m})
        pred_df = test_df.copy()
        pred_df["Experiment"] = exp_name
        pred_df["Model"] = model_name
        pred_df["Original_RSRP"] = y_test.values
        pred_df["Predicted_RSRP"] = y_pred
        pred_df["Error"] = pred_df["Predicted_RSRP"] - pred_df["Original_RSRP"]
        prediction_dfs.append(pred_df)
        save_model_outputs(pred_df, y_test, y_pred, model_dir, exp_name, model_name)
        grouped_error_analysis(pred_df, model_dir, exp_name, model_name)

    result_df = pd.DataFrame(results)
    pred_all = pd.concat(prediction_dfs, ignore_index=True) if prediction_dfs else pd.DataFrame()

    result_df.to_excel(exp_dir / f"{safe_name(exp_name)}_results.xlsx", index=False)
    if SAVE_ALL_PREDICTIONS and not pred_all.empty:
        pred_all.to_excel(exp_dir / f"{safe_name(exp_name)}_predictions.xlsx", index=False)

    return result_df, pred_all

# ============================================================
# SECTION 11: SPLIT BUILDERS
# ============================================================

def random_split_experiment(df, test_size=0.15):
    train_val, test = train_test_split(df, test_size=test_size, random_state=RANDOM_STATE)
    return train_val.copy(), test.copy()


def leave_one_site_splits(df: pd.DataFrame, site_col=SITE_COL):
    sites = sorted(df[site_col].dropna().unique())
    for site in sites:
        train = df[df[site_col] != site].copy()
        test = df[df[site_col] == site].copy()
        yield site, train, test


def sector_holdout_split(df: pd.DataFrame, sector_site_col=SITE_SECTOR_COL):
    sectors = sorted(df[sector_site_col].dropna().unique())
    test_sectors = sectors[2::5]
    if len(test_sectors) < 3:
        test_sectors = sectors[-3:]
    train = df[~df[sector_site_col].isin(test_sectors)].copy()
    test = df[df[sector_site_col].isin(test_sectors)].copy()
    return test_sectors, train, test

# ============================================================
# SECTION 12: EXPERIMENT SUITE E1-E17
# ============================================================

def run_experiment_suite(config):
    output_dir = config["output_dir"]
    models_to_run = config["models_to_run"]
    datasets = config["datasets"]

    all_results = []
    all_predictions = []

    def run(exp_name, train_df, test_df, group=None, train_fraction=None):
        res, pred = run_single_experiment(
            exp_name=exp_name,
            train_df=train_df,
            test_df=test_df,
            output_dir=output_dir,
            models_to_run=models_to_run,
            train_fraction=train_fraction,
            experiment_group=group,
        )
        if not res.empty:
            all_results.append(res)
        if not pred.empty:
            all_predictions.append(pred)

    ch33 = datasets["Chicago_3p3"]
    ch35 = datasets["Chicago_3p5"]
    br33 = datasets["Brussels_3p3"]
    br37 = datasets["Brussels_3p7"]

    # E1-E3 random ID
    for exp_name, df in [
        ("E1_Chicago_3p3_Random_ID", ch33),
        ("E2_Brussels_3p3_Random_ID", br33),
        ("E3_Brussels_3p7_Random_ID", br37),
    ]:
        train, test = random_split_experiment(df)
        run(exp_name, train, test, group="Random_ID")

    # E4-E10 OOD shifts
    run("E4_Chicago_Frequency_3p3_to_3p5", ch33, ch35, group="Frequency_OOD")
    run("E5_Chicago_Frequency_3p5_to_3p3", ch35, ch33, group="Frequency_OOD")
    run("E6_Brussels_Frequency_3p3_to_3p7", br33, br37, group="Frequency_OOD")
    run("E7_Brussels_Frequency_3p7_to_3p3", br37, br33, group="Frequency_OOD")
    run("E8_City_Chicago3p3_to_Brussels3p3", ch33, br33, group="City_OOD")
    run("E9_City_Brussels3p3_to_Chicago3p3", br33, ch33, group="City_OOD")
    run("E10_Strong_OOD_Chicago3p3_to_Brussels3p7", ch33, br37, group="Strong_OOD")

    # E11-E12 site holdout
    for city_name, df in [("Chicago_3p3", ch33), ("Brussels_3p3", br33)]:
        for held_site, train, test in leave_one_site_splits(df):
            run(f"E11_E12_SiteHoldout_{city_name}_Test_{held_site}", train, test, group=f"SiteHoldout_{city_name}")

    # E13-E14 sector holdout
    for city_name, df in [("Chicago_3p3", ch33), ("Brussels_3p3", br33)]:
        test_sectors, train, test = sector_holdout_split(df)
        run(f"E13_E14_SectorHoldout_{city_name}", train, test, group=f"SectorHoldout_{city_name}")

    # E15-E17 data scarcity
    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    ch33_train_full, ch33_test_fixed = random_split_experiment(ch33)
    for frac in fractions:
        run(f"E15_Scarcity_ID_Chicago3p3_frac_{frac}", ch33_train_full, ch33_test_fixed, group="Scarcity_ID", train_fraction=frac)
        run(f"E16_Scarcity_FreqOOD_Chicago3p3_to_3p5_frac_{frac}", ch33_train_full, ch35, group="Scarcity_Frequency_OOD", train_fraction=frac)
        run(f"E17_Scarcity_CityOOD_Chicago3p3_to_Brussels3p3_frac_{frac}", ch33_train_full, br33, group="Scarcity_City_OOD", train_fraction=frac)

    results_master = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    predictions_master = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    results_master = add_ood_degradation(results_master)
    aulc_df = compute_aulc(results_master, group_cols=["Model", "Experiment_Group"])
    site_summary = compute_site_fold_summary(results_master)
    save_master_outputs(results_master, predictions_master, aulc_df, site_summary, output_dir)

    return results_master, predictions_master, aulc_df, site_summary


def add_ood_degradation(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["OOD_Degradation_Percent"] = np.nan
    id_ref = results[results["Experiment"] == "E1_Chicago_3p3_Random_ID"]
    ref_by_model = id_ref.set_index("Model")["RMSE"].to_dict() if not id_ref.empty else {}
    for idx, row in results.iterrows():
        model = row["Model"]
        if row["Experiment_Group"] in ["Frequency_OOD", "City_OOD", "Strong_OOD", "Scarcity_Frequency_OOD", "Scarcity_City_OOD"]:
            ref = ref_by_model.get(model, np.nan)
            results.loc[idx, "OOD_Degradation_Percent"] = ood_degradation(ref, row["RMSE"])
    return results


def compute_site_fold_summary(results: pd.DataFrame) -> pd.DataFrame:
    site_rows = results[results["Experiment_Group"].astype(str).str.contains("SiteHoldout", na=False)].copy()
    if site_rows.empty:
        return pd.DataFrame()
    return site_rows.groupby(["Experiment_Group", "Model"]).agg(
        SiteFold_RMSE_Mean=("RMSE", "mean"),
        SiteFold_RMSE_Std=("RMSE", "std"),
        SiteFold_MAE_Mean=("MAE", "mean"),
        SiteFold_MAE_Std=("MAE", "std"),
        Folds=("Experiment", "nunique")
    ).reset_index()


def save_master_outputs(results_master, predictions_master, aulc_df, site_summary, output_dir: Path):
    excel_path = output_dir / "master_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        results_master.to_excel(writer, sheet_name="All_Results", index=False)
        if not aulc_df.empty:
            aulc_df.to_excel(writer, sheet_name="AULC", index=False)
        if not site_summary.empty:
            site_summary.to_excel(writer, sheet_name="SiteFold_Summary", index=False)
        for metric in ["RMSE", "MAE", "R2", "Mean_Bias_Error", "P95_Absolute_Error", "Train_Time_sec", "Prediction_Time_sec", "OOD_Degradation_Percent"]:
            if metric in results_master.columns:
                pivot = results_master.pivot_table(index="Experiment", columns="Model", values=metric, aggfunc="mean")
                pivot.to_excel(writer, sheet_name=f"Pivot_{metric[:20]}")

    if SAVE_ALL_PREDICTIONS and not predictions_master.empty:
        pred_dir = make_dir(output_dir / "master_predictions_csv_chunks")
        max_rows_per_file = 900_000
        n_rows = len(predictions_master)
        n_chunks = int(np.ceil(n_rows / max_rows_per_file))
        for i in range(n_chunks):
            start = i * max_rows_per_file
            end = min((i + 1) * max_rows_per_file, n_rows)
            chunk = predictions_master.iloc[start:end]
            chunk.to_csv(pred_dir / f"master_predictions_part_{i+1:03d}_of_{n_chunks:03d}.csv", index=False)
        predictions_master.to_csv(output_dir / "master_predictions_full.csv.gz", index=False, compression="gzip")
        sample_n = min(100_000, n_rows)
        pred_sample = predictions_master.sample(n=sample_n, random_state=RANDOM_STATE) if n_rows > sample_n else predictions_master
        pred_sample.to_excel(output_dir / "master_predictions_SAMPLE.xlsx", index=False)

    plots_dir = make_dir(output_dir / "master_plots")
    for metric in ["RMSE", "MAE", "R2", "Mean_Bias_Error", "P95_Absolute_Error", "OOD_Degradation_Percent", "Train_Time_sec", "Prediction_Time_sec"]:
        save_metric_barplot(results_master, metric, plots_dir / f"master_{metric}.png", f"Average {metric} by Model")

    results_master.to_csv(output_dir / "master_results.csv", index=False)
    print(f"\nSaved master results to: {excel_path}")

# ============================================================
# SECTION 13: MAIN PROGRAM
# ============================================================

def main():
    print("\nIINN Reviewer Experiment Runner - IINN + Stable DNN Version")
    print("=========================================================")

    output_dir = ask_path("Enter output folder path to save all results: ", must_exist=False)
    make_dir(output_dir)

    print("\nEnter Excel file paths:")
    ch33_path = ask_path("Chicago 3.3 GHz file path: ")
    ch35_path = ask_path("Chicago 3.5 GHz file path: ")
    br33_path = ask_path("Brussels 3.3 GHz file path: ")
    br37_path = ask_path("Brussels 3.7 GHz file path: ")

    print("\nLoading datasets...")
    datasets = {
        "Chicago_3p3": load_excel_dataset(ch33_path, "Chicago_3p3", "Chicago", 3300),
        "Chicago_3p5": load_excel_dataset(ch35_path, "Chicago_3p5", "Chicago", 3500),
        "Brussels_3p3": load_excel_dataset(br33_path, "Brussels_3p3", "Brussels", 3300),
        "Brussels_3p7": load_excel_dataset(br37_path, "Brussels_3p7", "Brussels", 3700),
    }

    for key in datasets:
        datasets[key] = add_site_sector_columns(datasets[key])
        validate_required_columns(datasets[key], key)

    print("\nFeature columns used for all trainable models:")
    for c in FEATURE_COLS:
        print(f"  - {c}")
    print(f"\nTarget column: {TARGET_COL}")

    models_to_run = [
        "3GPP_Analytical",
        "RandomForest",
        "ExtraTrees",
        "HistGradientBoosting",
        "XGBoost",
        "LightGBM",
        "DNN",
        "IINN",
    ]

    if not XGBOOST_AVAILABLE and "XGBoost" in models_to_run:
        models_to_run.remove("XGBoost")
    if not LIGHTGBM_AVAILABLE and "LightGBM" in models_to_run:
        models_to_run.remove("LightGBM")
    if not TENSORFLOW_AVAILABLE:
        if "DNN" in models_to_run:
            models_to_run.remove("DNN")
        if "IINN" in models_to_run:
            models_to_run.remove("IINN")

    print(f"\nModels to run: {models_to_run}")

    config = {
        "output_dir": output_dir,
        "datasets": datasets,
        "models_to_run": models_to_run,
    }

    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump({
            "output_dir": str(output_dir),
            "feature_cols": FEATURE_COLS,
            "target_col": TARGET_COL,
            "models_to_run": models_to_run,
        }, f, indent=2)

    run_experiment_suite(config)

    print("\nDone.")
    print(f"Master results: {output_dir / 'master_results.xlsx'}")
    print(f"Master predictions compressed: {output_dir / 'master_predictions_full.csv.gz'}")


if __name__ == "__main__":
    main()
