# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:40:44 2026

@author: 2687492Z
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks, constraints, regularizers
except ImportError as exc:
    raise RuntimeError(
        "TensorFlow is required. Install it in the environment where you run this script, "
        "for example: pip install tensorflow pandas openpyxl"
    ) from exc


DATA_3GPP = Path(r"C:\Users\2687492Z\IINN-Rebuttal\Dataset_File_with_IINN\3.3Ghz_Bruselles_Data.xlsx")
DATA_SPM = Path(r"C:\Users\2687492Z\IINN-Rebuttal\Dataset_File_with_IINN\3.3Ghz_Brussels_AlternativePropagation.xlsx")
OUT_DIR = Path(r"C:\Users\2687492Z\IINN-Rebuttal\Dataset_File_with_IINN\Physics-Feature MLP baseline")

TARGET = "RSRP"
RANDOM_SEED = 42
ENSEMBLE_SEEDS = [42, 7, 123]

# These two columns are not ordinary radio measurements; in your files they are
# propagation-model parameters. Keeping them can recreate model-label alignment
# bias, so the generalized IINN excludes them by default.
USE_PROPAGATION_MODEL_CONSTANTS = False

# Uses only SPM input features, never SPM labels. This is a transductive
# covariate-shift correction: train points that look more like the OOD feature
# distribution receive slightly higher weight.
USE_OOD_FEATURE_WEIGHTING = True

# Optional few-shot target-domain calibration. The IINN itself is still trained
# on 3GPP labels only; this only fits a two-parameter affine correction on a
# small SPM calibration split and evaluates on the remaining SPM rows.
USE_SPM_FEW_SHOT_CALIBRATION = True
SPM_CALIBRATION_FRACTION = 0.10
RUN_PHYSICS_FEATURE_MLP_BASELINE = True
RESIDUAL_BOUND_DB = 3.0
RESIDUAL_DOMINANCE_WARNING_RATIO = 0.30


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "experiment"


def circular_abs_diff_deg(a: pd.Series, b: pd.Series) -> pd.Series:
    diff = (a.astype(float) - b.astype(float) + 180.0) % 360.0 - 180.0
    return diff.abs()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column {TARGET!r} in {path}")
    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physically meaningful features without imposing a propagation equation."""
    x = df.copy()

    x["delta_x"] = x["User_X"].astype(float) - x["Transmitter_X"].astype(float)
    x["delta_y"] = x["User_Y"].astype(float) - x["Transmitter_Y"].astype(float)
    x["horizontal_distance"] = np.sqrt(x["delta_x"] ** 2 + x["delta_y"] ** 2)
    x["log10_distance"] = np.log10(np.maximum(x["3D_Distance"].astype(float), 1e-6))
    x["log10_horizontal_distance"] = np.log10(np.maximum(x["horizontal_distance"].astype(float), 1e-6))
    x["log10_frequency"] = np.log10(np.maximum(x["Frequency"].astype(float), 1e-6))

    x["azimuth_misalignment_deg"] = circular_abs_diff_deg(
        x["User_Azimuth"], x["Transmitter_Azimuth"]
    )
    x["downtilt_misalignment_deg"] = (
        x["User_Downtilt"].astype(float) - x["Transmitter_Downtilt"].astype(float)
    ).abs()

    x["height_difference"] = x["Transmitter_Height"].astype(float) - x["User_Height"].astype(float)
    x["height_ratio"] = x["Transmitter_Height"].astype(float) / np.maximum(
        x["User_Height"].astype(float), 1e-6
    )
    x["building_clearance"] = x["Transmitter_Height"].astype(float) - x["Building_Height"].astype(float)

    x["vertical_to_beamwidth"] = x["Vertical_Attenuation"].astype(float) / np.maximum(
        x["Half-power Vertical Beamwidth"].astype(float), 1e-6
    )
    x["horizontal_to_beamwidth"] = x["Horizontal_Attenuation"].astype(float) / np.maximum(
        x["Half-power Horizontal Beamwidth"].astype(float), 1e-6
    )

    return x


def split_train_val_test(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    train_end = int(train_frac * len(indices))
    val_end = int((train_frac + val_frac) * len(indices))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def split_calibration_test(df: pd.DataFrame, calibration_frac: float):
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    calibration_size = max(1, int(calibration_frac * len(indices)))
    calibration_idx = indices[:calibration_size]
    test_idx = indices[calibration_size:]
    return (
        df.iloc[calibration_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def build_feature_tables(train_df: pd.DataFrame, val_df: pd.DataFrame, iid_test_df: pd.DataFrame, ood_df: pd.DataFrame):
    train_df = add_domain_features(train_df)
    val_df = add_domain_features(val_df)
    iid_test_df = add_domain_features(iid_test_df)
    ood_df = add_domain_features(ood_df)

    drop_columns = [
        TARGET,
        "Transmitter",
        # Absolute coordinates are dangerous for OOD testing because the train and
        # alternative-propagation files can sample different map regions. Use
        # relative geometry and distance features instead.
        "User_X",
        "User_Y",
        "Transmitter_X",
        "Transmitter_Y",
    ]
    if not USE_PROPAGATION_MODEL_CONSTANTS:
        drop_columns.extend(["LoS_Constant", "NLoS_Constant"])
    categorical_columns = ["Clutter Class"]

    numeric_columns = [
        column
        for column in train_df.columns
        if column not in drop_columns + categorical_columns
        and pd.api.types.is_numeric_dtype(train_df[column])
    ]

    # The model should not be forced to rediscover these useful radio abstractions
    # from raw columns alone, but no coefficients or propagation-law constants are fixed.
    preferred_first = [
        "Max_Transmitter_Power",
        "log10_distance",
        "log10_horizontal_distance",
        "log10_frequency",
        "3D_Distance",
        "horizontal_distance",
        "delta_x",
        "delta_y",
        "Frequency",
        "azimuth_misalignment_deg",
        "downtilt_misalignment_deg",
        "Vertical_Attenuation",
        "Horizontal_Attenuation",
        "User_Height",
        "Transmitter_Height",
        "height_difference",
        "height_ratio",
        "Street_Width",
        "Building_Height",
        "building_clearance",
        "Efficiency",
        "Half-power Horizontal Beamwidth",
        "Half-power Vertical Beamwidth",
    ]
    if USE_PROPAGATION_MODEL_CONSTANTS:
        preferred_first.extend(["LoS_Constant", "NLoS_Constant"])
    numeric_columns = [c for c in preferred_first if c in numeric_columns] + [
        c for c in numeric_columns if c not in preferred_first
    ]

    train_num = train_df[numeric_columns].astype(np.float32)
    val_num = val_df[numeric_columns].astype(np.float32)
    iid_num = iid_test_df[numeric_columns].astype(np.float32)
    ood_num = ood_df[numeric_columns].astype(np.float32)

    train_cat_parts = []
    val_cat_parts = []
    iid_cat_parts = []
    ood_cat_parts = []
    category_metadata = {}

    for column in categorical_columns:
        categories = sorted(train_df[column].astype(str).fillna("missing").unique().tolist())
        category_metadata[column] = categories

        def encode(frame: pd.DataFrame) -> pd.DataFrame:
            values = frame[column].astype(str).fillna("missing")
            encoded = pd.DataFrame(index=frame.index)
            for category in categories:
                encoded[f"{column}={category}"] = (values == category).astype(np.float32)
            encoded[f"{column}=UNKNOWN"] = (~values.isin(categories)).astype(np.float32)
            return encoded

        train_cat_parts.append(encode(train_df))
        val_cat_parts.append(encode(val_df))
        iid_cat_parts.append(encode(iid_test_df))
        ood_cat_parts.append(encode(ood_df))

    train_x = pd.concat([train_num] + train_cat_parts, axis=1)
    val_x = pd.concat([val_num] + val_cat_parts, axis=1)
    iid_x = pd.concat([iid_num] + iid_cat_parts, axis=1)
    ood_x = pd.concat([ood_num] + ood_cat_parts, axis=1)

    train_x = train_x.replace([np.inf, -np.inf], np.nan).fillna(train_x.median(numeric_only=True))
    val_x = val_x.replace([np.inf, -np.inf], np.nan).fillna(train_x.median(numeric_only=True))
    iid_x = iid_x.replace([np.inf, -np.inf], np.nan).fillna(train_x.median(numeric_only=True))
    ood_x = ood_x.replace([np.inf, -np.inf], np.nan).fillna(train_x.median(numeric_only=True))

    metadata = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "category_metadata": category_metadata,
        "feature_columns": train_x.columns.tolist(),
    }

    return train_x, val_x, iid_x, ood_x, metadata


def infer_feature_roles(feature_columns: list[str]) -> dict[str, list[int]]:
    def matching(*tokens: str) -> list[int]:
        tokens_lower = [token.lower() for token in tokens]
        return [
            i
            for i, name in enumerate(feature_columns)
            if any(token in name.lower() for token in tokens_lower)
        ]

    tx = matching("max_transmitter_power", "tx_power", "ptx")
    distance_frequency = matching("distance", "frequency", "log10_distance", "log10_frequency")
    alignment = matching("azimuth", "downtilt", "attenuation", "beamwidth", "misalignment")
    height_environment = matching(
        "height",
        "street_width",
        "building",
        "clearance",
        "clutter class",
        "los_constant",
        "nlos_constant",
        "efficiency",
    )

    used = set(tx + distance_frequency + alignment + height_environment)
    residual = [i for i in range(len(feature_columns)) if i not in used]

    return {
        "tx": sorted(set(tx)),
        "distance_frequency": sorted(set(distance_frequency)),
        "alignment": sorted(set(alignment)),
        "height_environment": sorted(set(height_environment)),
        "residual": residual,
    }


class ClipLayer(tf.keras.layers.Layer):
    def __init__(self, min_value: float = -5.0, max_value: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_value, self.max_value)

    def get_config(self):
        config = super().get_config()
        config.update({"min_value": self.min_value, "max_value": self.max_value})
        return config


def slice_role(x, indices: list[int], name: str):
    if not indices:
        return None
    return tf.keras.layers.Lambda(
        lambda value, idx=indices: tf.gather(value, idx, axis=1),
        name=name,
    )(x)


def dense_stack(x, widths: list[int], name: str, l2: float, dropout: float = 0.0):
    reg = regularizers.l2(l2) if l2 else None
    y = x
    for i, width in enumerate(widths, start=1):
        y = tf.keras.layers.Dense(
            width,
            activation="swish",
            kernel_initializer="he_normal",
            kernel_regularizer=reg,
            name=f"{name}_dense_{i}",
        )(y)
        if dropout:
            y = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_{i}")(y)
    return y


def scalar_head(x, name: str, l2: float, zero_init: bool = True, activation: str = "linear"):
    reg = regularizers.l2(l2) if l2 else None
    return tf.keras.layers.Dense(
        1,
        activation=activation,
        kernel_initializer="zeros" if zero_init else "he_normal",
        bias_initializer="zeros",
        kernel_regularizer=reg,
        name=name,
    )(x)


def bounded_head(x, name: str, scale: float, l2: float):
    raw = scalar_head(x, f"{name}_raw", l2, activation="tanh")
    return tf.keras.layers.Lambda(lambda value, s=scale: value * s, name=name)(raw)


def build_generalized_iinn(
    feature_columns: list[str],
    l2: float = 1e-5,
    dropout: float = 0.05,
    residual_bound_db: float = RESIDUAL_BOUND_DB,
    target_std_db: float = 1.0,
):
    """Domain-informed, but not 3GPP-locked.

    Structure:
      RSRP = learned Tx contribution
             - learned propagation loss
             + learned alignment/gain term
             + learned environment/site term
             + learned residual interaction, explicitly bounded in dB so that
               the residual cannot silently become the dominant predictor.

    The branch split injects radio knowledge. The coefficients and functions are
    learned from labels, so the model can adapt to 3GPP, SPM, or measurements.
    """
    n_features = len(feature_columns)
    roles = infer_feature_roles(feature_columns)
    residual_scale = float(residual_bound_db) / max(float(target_std_db), 1e-6)

    inputs = tf.keras.layers.Input(shape=(n_features,), name="features")
    normalizer = tf.keras.layers.Normalization(axis=-1, name="normalizer")
    x_norm = normalizer(inputs)
    x = ClipLayer(-5.0, 5.0, name="clip_normalized_features")(x_norm)

    terms = []
    trace = {"roles": roles, "normalized": x_norm, "clipped_normalized": x}

    tx_x = slice_role(x, roles["tx"], "tx_slice")
    if tx_x is not None:
        tx_term = tf.keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer=tf.keras.initializers.Ones(),
            bias_initializer="zeros",
            name="tx_power_term",
        )(tx_x)
    else:
        tx_term = tf.keras.layers.Lambda(lambda value: value[:, :1] * 0.0, name="zero_tx_term")(x)
    terms.append(tx_term)
    trace["tx_power_term"] = tx_term

    df_x = slice_role(x, roles["distance_frequency"], "distance_frequency_slice")
    if df_x is not None:
        df_hidden = dense_stack(df_x, [32, 24], "propagation_loss", l2, dropout)
        flexible_loss = scalar_head(df_hidden, "flexible_path_loss", l2, activation="softplus")

        # A weak monotonic inductive bias: larger distance/frequency features should
        # be allowed to increase loss, but the slope is learned and not model-specific.
        monotone_inputs = tf.keras.layers.Lambda(
            lambda value: value + 5.0,
            name="nonnegative_distance_frequency_inputs",
        )(df_x)
        monotone_loss = tf.keras.layers.Dense(
            1,
            activation="linear",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Constant(0.5),
            kernel_constraint=constraints.NonNeg(),
            name="learned_monotone_distance_frequency_loss",
        )(monotone_inputs)
        path_loss = tf.keras.layers.Add(name="learned_path_loss")([flexible_loss, monotone_loss])
    else:
        path_loss = tf.keras.layers.Lambda(lambda value: value[:, :1] * 0.0, name="zero_path_loss")(x)

    negative_path_loss = tf.keras.layers.Lambda(lambda value: -value, name="negative_path_loss")(path_loss)
    terms.append(negative_path_loss)
    trace["path_loss"] = path_loss

    align_x = slice_role(x, roles["alignment"], "alignment_slice")
    if align_x is not None:
        align_hidden = dense_stack(align_x, [32, 16], "alignment_gain", l2, dropout)
        alignment_term = bounded_head(align_hidden, "alignment_term", scale=35.0, l2=l2)
        terms.append(alignment_term)
        trace["alignment_term"] = alignment_term

    env_x = slice_role(x, roles["height_environment"], "height_environment_slice")
    if env_x is not None:
        env_hidden = dense_stack(env_x, [32, 16], "environment_site", l2, dropout)
        environment_term = bounded_head(env_hidden, "environment_site_term", scale=45.0, l2=l2)
        terms.append(environment_term)
        trace["environment_site_term"] = environment_term

    residual_hidden = dense_stack(x, [64, 32], "residual_interaction", l2, dropout)
    residual_term = bounded_head(residual_hidden, "residual_interaction_term", scale=residual_scale, l2=l2)
    terms.append(residual_term)
    trace["residual_interaction_term"] = residual_term

    prediction = tf.keras.layers.Add(name="domain_informed_sum")(terms)
    output = tf.keras.layers.Dense(
        1,
        activation="linear",
        kernel_initializer=tf.keras.initializers.Ones(),
        bias_initializer="zeros",
        name="final_calibration",
    )(prediction)
    trace["prediction_before_calibration"] = prediction
    trace["output"] = output

    model = tf.keras.Model(inputs=inputs, outputs=output, name="Generalized_IINN")
    model.normalizer = normalizer
    model.trace_tensors = trace
    model.feature_columns = feature_columns
    model.residual_bound_db = float(residual_bound_db)
    return model


def build_physics_feature_mlp(n_features: int, l2: float = 1e-5, dropout: float = 0.20) -> tf.keras.Model:
    """Feature-engineering baseline with the same physics-derived inputs.

    This intentionally removes the IINN graph decomposition. It receives the
    same physically meaningful feature table as the Generalized IINN, but maps
    it through a conventional dense 128-64-32 MLP. This isolates whether the
    gain comes from IINN topology rather than feature engineering alone.
    """
    reg = regularizers.l2(l2) if l2 else None
    inputs = tf.keras.layers.Input(shape=(n_features,), name="physics_features")
    normalizer = tf.keras.layers.Normalization(axis=-1, name="normalizer")
    x = ClipLayer(-5.0, 5.0, name="clip_normalized_features")(normalizer(inputs))
    for width in [128, 64, 32]:
        x = tf.keras.layers.Dense(
            width,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=reg,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    output = tf.keras.layers.Dense(1, activation="linear", name="rsrp_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=output, name="Physics_Feature_MLP")
    model.normalizer = normalizer
    return model


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred.reshape(-1) - y_true.reshape(-1)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    mbe = float(np.mean(err))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true.reshape(-1) - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"RMSE_dB": rmse, "MAE_dB": mae, "MBE_dB": mbe, "R2": r2}


def compute_ood_feature_weights(train_x: pd.DataFrame, ood_x: pd.DataFrame) -> np.ndarray:
    """Weight 3GPP training rows by similarity to the unlabeled SPM feature domain."""
    train = train_x.to_numpy(np.float32)
    ood = ood_x.to_numpy(np.float32)

    center = np.nanmedian(train, axis=0)
    scale = np.nanpercentile(train, 75, axis=0) - np.nanpercentile(train, 25, axis=0)
    scale = np.where(scale < 1e-6, np.nanstd(train, axis=0), scale)
    scale = np.where(scale < 1e-6, 1.0, scale)

    train_z = np.clip((train - center) / scale, -5.0, 5.0)
    ood_z = np.clip((ood - center) / scale, -5.0, 5.0)
    ood_center = np.mean(ood_z, axis=0, keepdims=True)

    squared_distance = np.mean((train_z - ood_center) ** 2, axis=1)
    weights = np.exp(-0.35 * squared_distance)
    weights = weights / np.mean(weights)
    return np.clip(weights, 0.35, 3.0).astype(np.float32)


def fit_affine_calibrator(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Fit y_true ~= slope * y_pred + intercept on a calibration split."""
    pred = y_pred.reshape(-1).astype(np.float64)
    true = y_true.reshape(-1).astype(np.float64)
    pred_std = float(np.std(pred))
    if pred_std < 1e-8:
        slope = 1.0
        intercept = float(np.mean(true) - np.mean(pred))
    else:
        slope, intercept = np.polyfit(pred, true, deg=1)
        slope = float(np.clip(slope, 0.5, 1.8))
        intercept = float(intercept)
    return {"slope": slope, "intercept": intercept}


def apply_affine_calibrator(y_pred: np.ndarray, calibrator: dict[str, float]) -> np.ndarray:
    return calibrator["slope"] * y_pred.reshape(-1) + calibrator["intercept"]


def save_training_curve(history: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.sqrt(history["loss"]), label="Train RMSE")
    ax.plot(np.sqrt(history["val_loss"]), label="Validation RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (dB)")
    ax.set_title("Generalized IINN Training Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_ensemble_learning_curves(histories: list[pd.DataFrame], out_dir: Path) -> None:
    """Save learning-curve evidence that the neural model is learning from data."""
    import matplotlib.pyplot as plt

    if not histories:
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))
    summary_rows = []

    for i, history in enumerate(histories, start=1):
        seed = int(history["ensemble_seed"].iloc[0]) if "ensemble_seed" in history else i
        epochs = np.arange(1, len(history) + 1)
        train_rmse = np.sqrt(history["loss"].to_numpy(dtype=float))
        val_rmse = np.sqrt(history["val_loss"].to_numpy(dtype=float))

        ax.plot(
            epochs,
            train_rmse,
            color="#2563eb",
            alpha=0.22,
            linewidth=1.4,
            label="Train RMSE" if i == 1 else None,
        )
        ax.plot(
            epochs,
            val_rmse,
            color="#dc2626",
            alpha=0.22,
            linewidth=1.4,
            label="Validation RMSE" if i == 1 else None,
        )

        best_idx = int(np.argmin(val_rmse))
        summary_rows.append(
            {
                "ensemble_seed": seed,
                "epochs_ran": int(len(history)),
                "train_rmse_epoch_1": float(train_rmse[0]),
                "val_rmse_epoch_1": float(val_rmse[0]),
                "best_val_rmse": float(val_rmse[best_idx]),
                "best_val_epoch": int(best_idx + 1),
                "final_train_rmse": float(train_rmse[-1]),
                "final_val_rmse": float(val_rmse[-1]),
                "train_rmse_reduction": float(train_rmse[0] - train_rmse[-1]),
                "val_rmse_reduction": float(val_rmse[0] - np.min(val_rmse)),
            }
        )

    # Add a clearer mean curve over the common epoch range.
    min_epochs = min(len(h) for h in histories)
    if min_epochs > 1:
        train_stack = np.vstack([np.sqrt(h["loss"].to_numpy(dtype=float)[:min_epochs]) for h in histories])
        val_stack = np.vstack([np.sqrt(h["val_loss"].to_numpy(dtype=float)[:min_epochs]) for h in histories])
        epochs = np.arange(1, min_epochs + 1)
        ax.plot(epochs, train_stack.mean(axis=0), color="#2563eb", linewidth=2.8, label="Mean train RMSE")
        ax.plot(epochs, val_stack.mean(axis=0), color="#dc2626", linewidth=2.8, label="Mean validation RMSE")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE on standardized target")
    ax.set_title("Generalized IINN Learning Curves Across Ensemble Members")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "generalized_iinn_ensemble_learning_curves.png", dpi=220)
    plt.close(fig)

    pd.DataFrame(summary_rows).to_excel(out_dir / "generalized_iinn_learning_curve_summary.xlsx", index=False)


def save_calibration_fit_plot(
    y_cal_true: np.ndarray,
    y_cal_pred_uncalibrated: np.ndarray,
    calibrator: dict[str, float],
    out_dir: Path,
) -> None:
    """Visualize the two-parameter few-shot SPM affine calibrator."""
    import matplotlib.pyplot as plt

    if y_cal_true is None or y_cal_pred_uncalibrated is None or len(y_cal_true) == 0:
        return

    true = y_cal_true.reshape(-1)
    pred = y_cal_pred_uncalibrated.reshape(-1)
    pred_cal = apply_affine_calibrator(pred, calibrator)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(pred, true, s=10, alpha=0.25, color="#2563eb", label="SPM calibration samples")

    x_min = float(np.nanmin(pred))
    x_max = float(np.nanmax(pred))
    x_line = np.linspace(x_min, x_max, 200)
    y_line = apply_affine_calibrator(x_line, calibrator)
    ax.plot(
        x_line,
        y_line,
        color="#dc2626",
        linewidth=2.4,
        label=f"Affine fit: y={calibrator['slope']:.4f}x+{calibrator['intercept']:.2f}",
    )

    diag_min = float(min(np.nanmin(pred), np.nanmin(true), np.nanmin(pred_cal)))
    diag_max = float(max(np.nanmax(pred), np.nanmax(true), np.nanmax(pred_cal)))
    ax.plot([diag_min, diag_max], [diag_min, diag_max], color="#111111", linewidth=1.2, linestyle="--", label="Ideal")

    ax.set_xlabel("Uncalibrated IINN prediction on SPM calibration split (dB)")
    ax.set_ylabel("SPM label (dB)")
    ax.set_title("Few-Shot SPM Calibration Fit")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "few_shot_spm_calibration_fit.png", dpi=220)
    plt.close(fig)


def save_calibration_effect_plots(
    y_true: np.ndarray,
    y_pred_uncalibrated: np.ndarray,
    y_pred_calibrated: np.ndarray,
    out_dir: Path,
) -> None:
    """Save before/after plots showing that calibration fixes systematic bias."""
    import matplotlib.pyplot as plt

    true = y_true.reshape(-1)
    pred_uncal = y_pred_uncalibrated.reshape(-1)
    pred_cal = y_pred_calibrated.reshape(-1)
    err_uncal = pred_uncal - true
    err_cal = pred_cal - true

    uncal_metrics = metrics(true, pred_uncal)
    cal_metrics = metrics(true, pred_cal)

    # True-vs-predicted before/after.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)
    axis_min = float(min(np.nanmin(true), np.nanmin(pred_uncal), np.nanmin(pred_cal)))
    axis_max = float(max(np.nanmax(true), np.nanmax(pred_uncal), np.nanmax(pred_cal)))
    for ax, pred, title, m, color in [
        (axes[0], pred_uncal, "Before few-shot calibration", uncal_metrics, "#2563eb"),
        (axes[1], pred_cal, "After few-shot calibration", cal_metrics, "#059669"),
    ]:
        ax.scatter(true, pred, s=8, alpha=0.22, color=color)
        ax.plot([axis_min, axis_max], [axis_min, axis_max], color="#111111", linewidth=1.4)
        ax.set_title(f"{title}\nRMSE={m['RMSE_dB']:.3f} dB, MBE={m['MBE_dB']:.3f} dB")
        ax.set_xlabel("True SPM RSRP (dB)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Predicted RSRP (dB)")
    fig.suptitle("Calibration Effect on OOD SPM Predictions")
    fig.tight_layout()
    fig.savefig(out_dir / "few_shot_calibration_true_vs_predicted_before_after.png", dpi=220)
    plt.close(fig)

    # Error histogram before/after.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(
        float(min(np.nanpercentile(err_uncal, 0.5), np.nanpercentile(err_cal, 0.5))),
        float(max(np.nanpercentile(err_uncal, 99.5), np.nanpercentile(err_cal, 99.5))),
        60,
    )
    ax.hist(err_uncal, bins=bins, alpha=0.45, color="#2563eb", label=f"Uncalibrated MBE={uncal_metrics['MBE_dB']:.3f} dB")
    ax.hist(err_cal, bins=bins, alpha=0.55, color="#059669", label=f"Calibrated MBE={cal_metrics['MBE_dB']:.3f} dB")
    ax.axvline(0.0, color="#111111", linewidth=1.2)
    ax.axvline(uncal_metrics["MBE_dB"], color="#2563eb", linewidth=2.2, linestyle="--")
    ax.axvline(cal_metrics["MBE_dB"], color="#059669", linewidth=2.2, linestyle="--")
    ax.set_xlabel("Prediction error, pred - true (dB)")
    ax.set_ylabel("Count")
    ax.set_title("Calibration Reduces Systematic OOD Bias")
    ax.grid(alpha=0.20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "few_shot_calibration_error_histogram_before_after.png", dpi=220)
    plt.close(fig)

    # Metric summary bar chart.
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    labels = ["RMSE", "Abs. MBE"]
    before = [uncal_metrics["RMSE_dB"], abs(uncal_metrics["MBE_dB"])]
    after = [cal_metrics["RMSE_dB"], abs(cal_metrics["MBE_dB"])]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, before, width, color="#2563eb", label="Uncalibrated")
    ax.bar(x + width / 2, after, width, color="#059669", label="Calibrated")
    for xpos, vals in [(x - width / 2, before), (x + width / 2, after)]:
        for xx, vv in zip(xpos, vals):
            ax.text(xx, vv + 0.15, f"{vv:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("dB")
    ax.set_title("Few-Shot Calibration: Error Magnitude Before vs After")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "few_shot_calibration_metric_bars.png", dpi=220)
    plt.close(fig)


def make_trace_model(model: tf.keras.Model) -> tf.keras.Model:
    tensor_outputs = {
        key: value
        for key, value in model.trace_tensors.items()
        if not isinstance(value, dict)
    }
    return tf.keras.Model(model.inputs, tensor_outputs, name=f"{model.name}_trace")


def save_white_box_diagnostics(
    model: tf.keras.Model,
    x_df: pd.DataFrame,
    y_true_db: np.ndarray,
    y_pred_db: np.ndarray,
    y_mean: float,
    y_std: float,
    out_dir: Path,
    prefix: str,
) -> dict[str, float | bool]:
    """Save visual diagnostics showing how the generalized IINN forms predictions."""
    import matplotlib.pyplot as plt

    trace_model = make_trace_model(model)
    traces = trace_model.predict(x_df.to_numpy(np.float32), verbose=0)

    rows = pd.DataFrame(
        {
            "true_RSRP_dB": y_true_db.reshape(-1),
            "pred_RSRP_dB": y_pred_db.reshape(-1),
            "error_pred_minus_true_dB": y_pred_db.reshape(-1) - y_true_db.reshape(-1),
        }
    )

    component_map = {
        "tx_power_term": "Tx contribution",
        "path_loss": "Path loss contribution",
        "alignment_term": "Alignment contribution",
        "environment_site_term": "Environment/site contribution",
        "residual_interaction_term": "Residual interaction",
        "prediction_before_calibration": "Pre-calibration sum",
    }

    component_columns = []
    for key, label in component_map.items():
        if key not in traces:
            continue
        values = traces[key].reshape(-1)
        if key == "path_loss":
            values = -values
        column = f"{label} (relative dB)"
        rows[column] = values * y_std
        component_columns.append(column)

    if "output" in traces:
        rows["single_model_output_dB"] = traces["output"].reshape(-1) * y_std + y_mean

    rows.to_excel(out_dir / f"{prefix}_white_box_contributions.xlsx", index=False)

    audit_component_columns = [
        column for column in component_columns if not column.startswith("Pre-calibration sum")
    ]

    summary = (
        rows[audit_component_columns]
        .agg(["mean", "std", "min", "max"])
        .T.reset_index()
        .rename(columns={"index": "component"})
    )
    for column in audit_component_columns:
        values = rows[column].to_numpy(dtype=float)
        summary.loc[summary["component"] == column, "mean_abs_dB"] = float(np.mean(np.abs(values)))
        summary.loc[summary["component"] == column, "median_abs_dB"] = float(np.median(np.abs(values)))
        summary.loc[summary["component"] == column, "p95_abs_dB"] = float(np.quantile(np.abs(values), 0.95))
    total_abs = float(summary["mean_abs_dB"].sum()) if "mean_abs_dB" in summary else 0.0
    summary["mean_abs_share_percent"] = (
        100.0 * summary["mean_abs_dB"] / total_abs if total_abs > 0 else np.nan
    )
    summary.to_excel(out_dir / f"{prefix}_white_box_contribution_summary.xlsx", index=False)

    means = summary.set_index("component")["mean"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(means.index, means.values)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean contribution to RSRP (dB, relative)")
    ax.set_title(f"{prefix}: learned IINN component contributions")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_component_contribution_bar.png", dpi=180)
    plt.close(fig)

    abs_means = summary.set_index("component")["mean_abs_dB"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(abs_means.index, abs_means.values, color="#2563eb")
    ax.set_xlabel("Mean absolute contribution (dB)")
    ax.set_title(f"{prefix}: component magnitude audit")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_component_magnitude_audit.png", dpi=220)
    plt.close(fig)

    if audit_component_columns:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.boxplot(
            [rows[c].to_numpy(dtype=float) for c in audit_component_columns],
            labels=audit_component_columns,
            showfliers=False,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Component contribution (dB)")
        ax.set_title(f"{prefix}: component-level white-box decomposition")
        ax.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_component_boxplot.png", dpi=220)
        plt.close(fig)

    diagnostics = {
        "residual_bound_db": float(getattr(model, "residual_bound_db", RESIDUAL_BOUND_DB)),
        "residual_dominance_warning_ratio": float(RESIDUAL_DOMINANCE_WARNING_RATIO),
    }
    residual_label = "Residual interaction (relative dB)"
    physics_labels = [
        "Path loss contribution (relative dB)",
        "Alignment contribution (relative dB)",
        "Environment/site contribution (relative dB)",
    ]
    if residual_label in audit_component_columns:
        residual_abs = float(np.mean(np.abs(rows[residual_label].to_numpy(dtype=float))))
        physics_abs_values = [
            float(np.mean(np.abs(rows[label].to_numpy(dtype=float))))
            for label in physics_labels
            if label in audit_component_columns
        ]
        largest_physics_abs = max(physics_abs_values) if physics_abs_values else 0.0
        residual_max_abs = float(np.max(np.abs(rows[residual_label].to_numpy(dtype=float))))
        diagnostics.update(
            {
                "residual_mean_abs_dB": residual_abs,
                "largest_physics_component_mean_abs_dB": largest_physics_abs,
                "residual_to_largest_physics_ratio": residual_abs / max(largest_physics_abs, 1e-9),
                "residual_to_total_component_ratio": residual_abs / max(total_abs, 1e-9),
                "residual_max_abs_dB": residual_max_abs,
                "residual_within_declared_bound": bool(
                    residual_max_abs <= diagnostics["residual_bound_db"] + 1e-4
                ),
                "residual_dominates_warning": bool(
                    residual_abs / max(largest_physics_abs, 1e-9) > RESIDUAL_DOMINANCE_WARNING_RATIO
                ),
            }
        )
    pd.DataFrame([diagnostics]).to_excel(out_dir / f"{prefix}_component_audit.xlsx", index=False)

    err = y_pred_db.reshape(-1) - y_true_db.reshape(-1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true_db.reshape(-1), y_pred_db.reshape(-1), s=8, alpha=0.35)
    lo = float(min(np.min(y_true_db), np.min(y_pred_db)))
    hi = float(max(np.max(y_true_db), np.max(y_pred_db)))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("True RSRP (dB)")
    ax.set_ylabel("Predicted RSRP (dB)")
    ax.set_title(f"{prefix}: true vs predicted")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_true_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=50, alpha=0.85)
    ax.axvline(float(np.mean(err)), color="black", linewidth=1, label=f"MBE={np.mean(err):.2f} dB")
    ax.set_xlabel("Prediction error, pred - true (dB)")
    ax.set_ylabel("Count")
    ax.set_title(f"{prefix}: error distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_error_histogram.png", dpi=180)
    plt.close(fig)
    return diagnostics


def save_prediction_diagnostics(
    y_true_db: np.ndarray,
    y_pred_db: np.ndarray,
    out_dir: Path,
    prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    y_true = y_true_db.reshape(-1)
    y_pred = y_pred_db.reshape(-1)
    err = y_pred - y_true

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("True RSRP (dB)")
    ax.set_ylabel("Predicted RSRP (dB)")
    ax.set_title(f"{prefix}: true vs predicted")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_true_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=50, alpha=0.85)
    ax.axvline(float(np.mean(err)), color="black", linewidth=1, label=f"MBE={np.mean(err):.2f} dB")
    ax.set_xlabel("Prediction error, pred - true (dB)")
    ax.set_ylabel("Count")
    ax.set_title(f"{prefix}: error distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_error_histogram.png", dpi=180)
    plt.close(fig)


def train_physics_feature_mlp_baseline(
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    iid_test_x: pd.DataFrame,
    spm_cal_x: pd.DataFrame,
    spm_eval_x: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_iid: np.ndarray,
    y_spm_cal: np.ndarray,
    y_ood: np.ndarray,
    y_mean: float,
    y_std: float,
    sample_weight: np.ndarray | None,
    out_dir: Path,
) -> dict[str, object]:
    """Train the feature-engineering MLP baseline under the identical protocol."""
    iid_predictions = []
    cal_predictions = []
    ood_predictions = []
    histories = []
    cal_rows = len(spm_cal_x)

    for run_no, seed in enumerate(ENSEMBLE_SEEDS, start=1):
        print(f"\n--- Training Physics-Feature MLP baseline {run_no}/{len(ENSEMBLE_SEEDS)} seed={seed} ---")
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.keras.backend.clear_session()

        model = build_physics_feature_mlp(train_x.shape[1], l2=2e-5, dropout=0.20)
        model.normalizer.adapt(train_x.to_numpy(np.float32))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
            loss="mse",
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-6),
            callbacks.TerminateOnNaN(),
        ]
        history = model.fit(
            train_x.to_numpy(np.float32),
            y_train,
            sample_weight=sample_weight,
            validation_data=(val_x.to_numpy(np.float32), y_val),
            epochs=300,
            batch_size=64,
            verbose=1,
            callbacks=cb,
        )
        history_df = pd.DataFrame(history.history)
        history_df["ensemble_seed"] = seed
        histories.append(history_df)

        iid_z = model.predict(iid_test_x.to_numpy(np.float32), verbose=0).reshape(-1)
        ood_z = model.predict(spm_eval_x.to_numpy(np.float32), verbose=0).reshape(-1)
        iid_predictions.append(iid_z * y_std + y_mean)
        ood_predictions.append(ood_z * y_std + y_mean)
        if cal_rows:
            cal_z = model.predict(spm_cal_x.to_numpy(np.float32), verbose=0).reshape(-1)
            cal_predictions.append(cal_z * y_std + y_mean)

        model.save(out_dir / f"physics_feature_mlp_seed_{seed}.keras")

    history_df = pd.concat(histories, axis=0, ignore_index=True)
    history_df.to_excel(out_dir / "physics_feature_mlp_training_history.xlsx", index=False)

    iid_pred = np.mean(np.vstack(iid_predictions), axis=0)
    ood_pred = np.mean(np.vstack(ood_predictions), axis=0)
    calibrator = {"slope": 1.0, "intercept": 0.0}
    calibrated_ood_pred = ood_pred
    calibration_metrics = None
    if cal_predictions:
        cal_pred = np.mean(np.vstack(cal_predictions), axis=0)
        calibrator = fit_affine_calibrator(cal_pred, y_spm_cal)
        calibrated_ood_pred = apply_affine_calibrator(ood_pred, calibrator)
        calibration_metrics = metrics(y_spm_cal.reshape(-1), apply_affine_calibrator(cal_pred, calibrator))

    iid_metrics = metrics(y_iid.reshape(-1), iid_pred)
    ood_metrics = metrics(y_ood.reshape(-1), ood_pred)
    calibrated_ood_metrics = metrics(y_ood.reshape(-1), calibrated_ood_pred)
    degradation = ((ood_metrics["RMSE_dB"] - iid_metrics["RMSE_dB"]) / iid_metrics["RMSE_dB"]) * 100.0
    calibrated_degradation = (
        (calibrated_ood_metrics["RMSE_dB"] - iid_metrics["RMSE_dB"]) / iid_metrics["RMSE_dB"]
    ) * 100.0

    save_prediction_diagnostics(y_ood, ood_pred, out_dir, "Physics_Feature_MLP_OOD_SPM_uncalibrated")
    save_prediction_diagnostics(y_ood, calibrated_ood_pred, out_dir, "Physics_Feature_MLP_OOD_SPM_calibrated")

    pd.DataFrame(
        {
            "OOD_true_RSRP": y_ood.reshape(-1),
            "OOD_pred_RSRP_uncalibrated": ood_pred,
            "OOD_pred_RSRP_calibrated": calibrated_ood_pred,
            "OOD_error_pred_minus_true": ood_pred - y_ood.reshape(-1),
            "OOD_calibrated_error_pred_minus_true": calibrated_ood_pred - y_ood.reshape(-1),
        }
    ).to_excel(out_dir / "physics_feature_mlp_ood_predictions.xlsx", index=False)

    result = {
        "model": "Physics-Feature MLP",
        "description": "Same physics-derived inputs as Generalized IINN, but conventional 128-64-32 dense topology.",
        "IID_3GPP_test": iid_metrics,
        "OOD_SPM_test": ood_metrics,
        "OOD_SPM_test_calibrated": calibrated_ood_metrics,
        "OOD_degradation_percent": float(degradation),
        "OOD_degradation_percent_calibrated": float(calibrated_degradation),
        "SPM_calibration_split": calibration_metrics,
        "affine_calibrator": calibrator,
        "affine_calibrator_parameter_count": 2,
    }
    with open(out_dir / "physics_feature_mlp_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(
        [
            {"Split": "PhysicsFeatureMLP_IID_3GPP_test", **iid_metrics},
            {"Split": "PhysicsFeatureMLP_OOD_SPM_test_uncalibrated", **ood_metrics},
            {"Split": "PhysicsFeatureMLP_OOD_SPM_test_calibrated", **calibrated_ood_metrics},
            {"Split": "PhysicsFeatureMLP_OOD_degradation_percent", "RMSE_dB": degradation},
            {"Split": "PhysicsFeatureMLP_OOD_degradation_percent_calibrated", "RMSE_dB": calibrated_degradation},
        ]
    ).to_excel(out_dir / "physics_feature_mlp_metrics.xlsx", index=False)
    return result


def main() -> None:
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    atoll_3gpp = load_dataset(DATA_3GPP)
    spm_ood = load_dataset(DATA_SPM)
    if USE_SPM_FEW_SHOT_CALIBRATION:
        spm_cal_df, spm_eval_df = split_calibration_test(spm_ood, SPM_CALIBRATION_FRACTION)
    else:
        spm_cal_df = spm_ood.iloc[0:0].copy()
        spm_eval_df = spm_ood

    train_df, val_df, iid_test_df = split_train_val_test(atoll_3gpp)
    spm_feature_df = pd.concat([spm_cal_df, spm_eval_df], axis=0, ignore_index=True)
    train_x, val_x, iid_test_x, ood_x, feature_metadata = build_feature_tables(
        train_df, val_df, iid_test_df, spm_feature_df
    )
    cal_rows = len(spm_cal_df)
    spm_cal_x = ood_x.iloc[:cal_rows].reset_index(drop=True)
    spm_eval_x = ood_x.iloc[cal_rows:].reset_index(drop=True)

    y_train_raw = train_df[TARGET].to_numpy(np.float32).reshape(-1, 1)
    y_val_raw = val_df[TARGET].to_numpy(np.float32).reshape(-1, 1)
    y_iid = iid_test_df[TARGET].to_numpy(np.float32).reshape(-1, 1)
    y_spm_cal = spm_cal_df[TARGET].to_numpy(np.float32).reshape(-1, 1)
    y_ood = spm_eval_df[TARGET].to_numpy(np.float32).reshape(-1, 1)

    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw) + 1e-6)
    y_train = (y_train_raw - y_mean) / y_std
    y_val = (y_val_raw - y_mean) / y_std

    sample_weight = None
    if USE_OOD_FEATURE_WEIGHTING:
        sample_weight = compute_ood_feature_weights(train_x, ood_x)
        pd.DataFrame({"sample_weight": sample_weight}).to_excel(
            OUT_DIR / "generalized_iinn_ood_feature_weights.xlsx",
            index=False,
        )

    iid_predictions = []
    spm_cal_predictions = []
    ood_predictions = []
    histories = []
    last_model = None
    last_ood_pred = None
    train_start = time.perf_counter()

    for run_no, seed in enumerate(ENSEMBLE_SEEDS, start=1):
        print(f"\n--- Training generalized IINN ensemble member {run_no}/{len(ENSEMBLE_SEEDS)} seed={seed} ---")
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.keras.backend.clear_session()

        model = build_generalized_iinn(
            train_x.columns.tolist(),
            l2=2e-5,
            dropout=0.08,
            residual_bound_db=RESIDUAL_BOUND_DB,
            target_std_db=y_std,
        )
        model.normalizer.adapt(train_x.to_numpy(np.float32))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
            loss="mse",
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )

        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-6),
            callbacks.TerminateOnNaN(),
        ]

        history = model.fit(
            train_x.to_numpy(np.float32),
            y_train,
            sample_weight=sample_weight,
            validation_data=(val_x.to_numpy(np.float32), y_val),
            epochs=300,
            batch_size=64,
            verbose=1,
            callbacks=cb,
        )

        history_df = pd.DataFrame(history.history)
        history_df["ensemble_seed"] = seed
        histories.append(history_df)

        iid_pred_z = model.predict(iid_test_x.to_numpy(np.float32), verbose=0).reshape(-1)
        if cal_rows:
            spm_cal_pred_z = model.predict(spm_cal_x.to_numpy(np.float32), verbose=0).reshape(-1)
            spm_cal_predictions.append(spm_cal_pred_z * y_std + y_mean)
        ood_pred_z = model.predict(spm_eval_x.to_numpy(np.float32), verbose=0).reshape(-1)
        iid_predictions.append(iid_pred_z * y_std + y_mean)
        ood_pred_db_single = ood_pred_z * y_std + y_mean
        ood_predictions.append(ood_pred_db_single)
        last_model = model
        last_ood_pred = ood_pred_db_single

        model.save(OUT_DIR / f"generalized_iinn_model_seed_{seed}.keras")

    train_seconds = time.perf_counter() - train_start

    history_df = pd.concat(histories, axis=0, ignore_index=True)
    history_df.to_excel(OUT_DIR / "generalized_iinn_training_history.xlsx", index=False)
    save_training_curve(histories[-1], OUT_DIR / "generalized_iinn_training_curve_rmse_last_member.png")
    save_ensemble_learning_curves(histories, OUT_DIR)

    iid_pred = np.mean(np.vstack(iid_predictions), axis=0)
    ood_pred = np.mean(np.vstack(ood_predictions), axis=0)
    spm_cal_pred = np.mean(np.vstack(spm_cal_predictions), axis=0) if spm_cal_predictions else None

    calibrator = {"slope": 1.0, "intercept": 0.0}
    calibrated_ood_pred = ood_pred
    calibration_metrics = None
    if USE_SPM_FEW_SHOT_CALIBRATION and spm_cal_pred is not None and len(spm_cal_pred):
        calibrator = fit_affine_calibrator(spm_cal_pred, y_spm_cal)
        calibrated_ood_pred = apply_affine_calibrator(ood_pred, calibrator)
        calibration_metrics = metrics(y_spm_cal.reshape(-1), apply_affine_calibrator(spm_cal_pred, calibrator))
        save_calibration_fit_plot(y_spm_cal, spm_cal_pred, calibrator, OUT_DIR)

    component_audit = None
    if last_model is not None and last_ood_pred is not None:
        component_audit = save_white_box_diagnostics(
            last_model,
            spm_eval_x,
            y_ood,
            last_ood_pred,
            y_mean,
            y_std,
            OUT_DIR,
            "OOD_SPM_last_ensemble_member",
        )
    save_prediction_diagnostics(y_ood, ood_pred, OUT_DIR, "OOD_SPM_ensemble_uncalibrated")
    save_prediction_diagnostics(y_ood, calibrated_ood_pred, OUT_DIR, "OOD_SPM_ensemble_calibrated")
    save_calibration_effect_plots(y_ood, ood_pred, calibrated_ood_pred, OUT_DIR)

    physics_feature_mlp_results = None
    if RUN_PHYSICS_FEATURE_MLP_BASELINE:
        physics_feature_mlp_results = train_physics_feature_mlp_baseline(
            train_x=train_x,
            val_x=val_x,
            iid_test_x=iid_test_x,
            spm_cal_x=spm_cal_x,
            spm_eval_x=spm_eval_x,
            y_train=y_train,
            y_val=y_val,
            y_iid=y_iid,
            y_spm_cal=y_spm_cal,
            y_ood=y_ood,
            y_mean=y_mean,
            y_std=y_std,
            sample_weight=sample_weight,
            out_dir=OUT_DIR,
        )

    iid_metrics = metrics(y_iid.reshape(-1), iid_pred)
    ood_metrics = metrics(y_ood.reshape(-1), ood_pred)
    calibrated_ood_metrics = metrics(y_ood.reshape(-1), calibrated_ood_pred)
    degradation = ((ood_metrics["RMSE_dB"] - iid_metrics["RMSE_dB"]) / iid_metrics["RMSE_dB"]) * 100.0
    calibrated_degradation = (
        (calibrated_ood_metrics["RMSE_dB"] - iid_metrics["RMSE_dB"]) / iid_metrics["RMSE_dB"]
    ) * 100.0

    results = {
        "train_dataset": str(DATA_3GPP),
        "ood_dataset": str(DATA_SPM),
        "target": TARGET,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "iid_test_rows": int(len(iid_test_df)),
        "spm_calibration_rows": int(len(spm_cal_df)),
        "ood_test_rows": int(len(spm_eval_df)),
        "train_seconds": train_seconds,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "use_propagation_model_constants": USE_PROPAGATION_MODEL_CONSTANTS,
        "use_ood_feature_weighting": USE_OOD_FEATURE_WEIGHTING,
        "use_spm_few_shot_calibration": USE_SPM_FEW_SHOT_CALIBRATION,
        "spm_calibration_fraction": SPM_CALIBRATION_FRACTION if USE_SPM_FEW_SHOT_CALIBRATION else 0.0,
        "affine_calibrator": calibrator,
        "affine_calibrator_parameter_count": 2,
        "residual_bound_db": RESIDUAL_BOUND_DB,
        "component_audit": component_audit,
        "physics_feature_mlp_baseline": physics_feature_mlp_results,
        "target_mean": y_mean,
        "target_std": y_std,
        "IID_3GPP_test": iid_metrics,
        "OOD_SPM_test": ood_metrics,
        "SPM_calibration_split": calibration_metrics,
        "OOD_SPM_test_calibrated": calibrated_ood_metrics,
        "OOD_degradation_percent": float(degradation),
        "OOD_degradation_percent_calibrated": float(calibrated_degradation),
        "feature_roles": infer_feature_roles(train_x.columns.tolist()),
        "feature_metadata": feature_metadata,
    }

    with open(OUT_DIR / "generalized_iinn_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(
        [
            {"Split": "IID_3GPP_test", **iid_metrics},
            {"Split": "OOD_SPM_test_uncalibrated", **ood_metrics},
            {"Split": "OOD_SPM_test_calibrated", **calibrated_ood_metrics},
            {"Split": "OOD_degradation_percent", "RMSE_dB": degradation},
            {"Split": "OOD_degradation_percent_calibrated", "RMSE_dB": calibrated_degradation},
            *(
                [
                    {"Split": "PhysicsFeatureMLP_IID_3GPP_test", **physics_feature_mlp_results["IID_3GPP_test"]},
                    {
                        "Split": "PhysicsFeatureMLP_OOD_SPM_test_uncalibrated",
                        **physics_feature_mlp_results["OOD_SPM_test"],
                    },
                    {
                        "Split": "PhysicsFeatureMLP_OOD_SPM_test_calibrated",
                        **physics_feature_mlp_results["OOD_SPM_test_calibrated"],
                    },
                    {
                        "Split": "PhysicsFeatureMLP_OOD_degradation_percent",
                        "RMSE_dB": physics_feature_mlp_results["OOD_degradation_percent"],
                    },
                    {
                        "Split": "PhysicsFeatureMLP_OOD_degradation_percent_calibrated",
                        "RMSE_dB": physics_feature_mlp_results["OOD_degradation_percent_calibrated"],
                    },
                ]
                if physics_feature_mlp_results is not None
                else []
            ),
        ]
    ).to_excel(OUT_DIR / "generalized_iinn_metrics.xlsx", index=False)

    predictions = pd.DataFrame(
        {
            "OOD_true_RSRP": y_ood.reshape(-1),
            "OOD_pred_RSRP_uncalibrated": ood_pred,
            "OOD_pred_RSRP_calibrated": calibrated_ood_pred,
            "OOD_error_pred_minus_true": ood_pred - y_ood.reshape(-1),
            "OOD_calibrated_error_pred_minus_true": calibrated_ood_pred - y_ood.reshape(-1),
        }
    )
    predictions.to_excel(OUT_DIR / "generalized_iinn_ood_predictions.xlsx", index=False)

    print("\n=== Generalized IINN Results ===")
    print(f"IID 3GPP test RMSE: {iid_metrics['RMSE_dB']:.3f} dB")
    print(f"OOD SPM test RMSE uncalibrated: {ood_metrics['RMSE_dB']:.3f} dB")
    print(f"OOD degradation uncalibrated:   {degradation:.2f}%")
    print(f"OOD MBE uncalibrated:           {ood_metrics['MBE_dB']:.3f} dB")
    if USE_SPM_FEW_SHOT_CALIBRATION:
        print(
            "Few-shot SPM calibrator: "
            f"slope={calibrator['slope']:.4f}, intercept={calibrator['intercept']:.4f}, "
            f"calibration_rows={len(spm_cal_df)}"
        )
        print(f"OOD SPM test RMSE calibrated:   {calibrated_ood_metrics['RMSE_dB']:.3f} dB")
        print(f"OOD degradation calibrated:     {calibrated_degradation:.2f}%")
        print(f"OOD MBE calibrated:             {calibrated_ood_metrics['MBE_dB']:.3f} dB")
    print(f"Outputs saved to:   {OUT_DIR}")


if __name__ == "__main__":
    main()
