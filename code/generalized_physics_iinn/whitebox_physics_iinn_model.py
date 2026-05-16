from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks, constraints, regularizers
except ImportError as exc:
    raise RuntimeError(
        "TensorFlow is required. Install it with: pip install tensorflow pandas openpyxl matplotlib"
    ) from exc


DATA_3GPP = Path(r"C:\Users\2687492Z\IINN-Rebuttal\Dataset_File_with_IINN\3.3Ghz_Bruselles_Data.xlsx")
DATA_SPM = Path(r"C:\Users\2687492Z\IINN-Rebuttal\Dataset_File_with_IINN\3.3Ghz_Brussels_AlternativePropagation.xlsx")
OUT_DIR = Path(r"C:\Users\2687492Z\IINN-Rebuttal\Dataset_File_with_IINN\whitebox_physics_iinn_results")

TARGET = "RSRP"
RANDOM_SEED = 42
ENSEMBLE_SEEDS = [42, 7, 123]
SPM_CALIBRATION_FRACTION = 0.10
RESIDUAL_BOUND_DB = 3.0
RESIDUAL_DOMINANCE_WARNING_RATIO = 0.30


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} not found in {path}")
    return df


def circular_abs_diff_deg(a: pd.Series, b: pd.Series) -> pd.Series:
    diff = (a.astype(float) - b.astype(float) + 180.0) % 360.0 - 180.0
    return diff.abs()


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["dx_m"] = x["User_X"].astype(float) - x["Transmitter_X"].astype(float)
    x["dy_m"] = x["User_Y"].astype(float) - x["Transmitter_Y"].astype(float)
    x["horizontal_distance_m"] = np.sqrt(x["dx_m"] ** 2 + x["dy_m"] ** 2)
    x["log10_d3d"] = np.log10(np.maximum(x["3D_Distance"].astype(float), 1e-6))
    x["log10_d2d"] = np.log10(np.maximum(x["horizontal_distance_m"].astype(float), 1e-6))
    x["log10_frequency"] = np.log10(np.maximum(x["Frequency"].astype(float), 1e-6))

    x["azimuth_error_deg"] = circular_abs_diff_deg(x["User_Azimuth"], x["Transmitter_Azimuth"])
    x["tilt_error_deg"] = (x["User_Downtilt"].astype(float) - x["Transmitter_Downtilt"].astype(float)).abs()

    x["azimuth_error_over_hbw"] = x["azimuth_error_deg"] / np.maximum(
        x["Half-power Horizontal Beamwidth"].astype(float), 1e-6
    )
    x["tilt_error_over_vbw"] = x["tilt_error_deg"] / np.maximum(
        x["Half-power Vertical Beamwidth"].astype(float), 1e-6
    )

    x["height_difference_m"] = x["Transmitter_Height"].astype(float) - x["User_Height"].astype(float)
    x["height_ratio"] = x["Transmitter_Height"].astype(float) / np.maximum(
        x["User_Height"].astype(float), 1e-6
    )
    x["building_clearance_m"] = x["Transmitter_Height"].astype(float) - x["Building_Height"].astype(float)
    x["log10_street_width"] = np.log10(np.maximum(x["Street_Width"].astype(float), 1e-6))
    return x


def split_train_val_test(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    train_end = int(train_frac * len(idx))
    val_end = int((train_frac + val_frac) * len(idx))
    return (
        df.iloc[idx[:train_end]].reset_index(drop=True),
        df.iloc[idx[train_end:val_end]].reset_index(drop=True),
        df.iloc[idx[val_end:]].reset_index(drop=True),
    )


def split_calibration_test(df: pd.DataFrame, calibration_frac: float):
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_cal = max(1, int(calibration_frac * len(idx)))
    return (
        df.iloc[idx[:n_cal]].reset_index(drop=True),
        df.iloc[idx[n_cal:]].reset_index(drop=True),
    )


def encode_clutter(train_df: pd.DataFrame, *frames: pd.DataFrame):
    categories = sorted(train_df["Clutter Class"].astype(str).fillna("missing").unique().tolist())

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        values = frame["Clutter Class"].astype(str).fillna("missing")
        encoded = pd.DataFrame(index=frame.index)
        for category in categories:
            encoded[f"clutter={category}"] = (values == category).astype(np.float32)
        encoded["clutter=UNKNOWN"] = (~values.isin(categories)).astype(np.float32)
        return encoded

    return categories, [encode(frame) for frame in frames]


def build_tables(train_df, val_df, iid_df, spm_cal_df, spm_eval_df):
    train_df = add_physics_features(train_df)
    val_df = add_physics_features(val_df)
    iid_df = add_physics_features(iid_df)
    spm_cal_df = add_physics_features(spm_cal_df)
    spm_eval_df = add_physics_features(spm_eval_df)

    scalar_columns = [
        "Max_Transmitter_Power",
        "log10_d3d",
        "log10_d2d",
        "log10_frequency",
        "azimuth_error_over_hbw",
        "tilt_error_over_vbw",
        "Vertical_Attenuation",
        "Horizontal_Attenuation",
        "height_difference_m",
        "height_ratio",
        "building_clearance_m",
        "log10_street_width",
        "Efficiency",
    ]

    frames = [train_df, val_df, iid_df, spm_cal_df, spm_eval_df]
    scalar_parts = [frame[scalar_columns].astype(np.float32).reset_index(drop=True) for frame in frames]
    clutter_categories, clutter_parts = encode_clutter(train_df, *frames)

    x_parts = []
    for scalar, clutter in zip(scalar_parts, clutter_parts):
        merged = pd.concat([scalar, clutter.reset_index(drop=True)], axis=1)
        merged = merged.replace([np.inf, -np.inf], np.nan)
        x_parts.append(merged)

    med = x_parts[0].median(numeric_only=True)
    x_parts = [x.fillna(med).astype(np.float32) for x in x_parts]

    metadata = {
        "scalar_columns": scalar_columns,
        "clutter_categories": clutter_categories,
        "feature_columns": x_parts[0].columns.tolist(),
    }
    return (*x_parts, metadata)


class ClipLayer(tf.keras.layers.Layer):
    def __init__(self, min_value=-5.0, max_value=5.0, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_value, self.max_value)

    def get_config(self):
        config = super().get_config()
        config.update({"min_value": self.min_value, "max_value": self.max_value})
        return config


def gather(x, feature_columns: list[str], columns: list[str], name: str):
    idx = [feature_columns.index(col) for col in columns if col in feature_columns]
    return tf.keras.layers.Lambda(lambda value, i=idx: tf.gather(value, i, axis=1), name=name)(x)


def bounded_linear(x, units: int, scale: float, name: str):
    raw = tf.keras.layers.Dense(
        units,
        activation="tanh",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name=f"{name}_raw",
    )(x)
    return tf.keras.layers.Lambda(lambda value, s=scale: value * s, name=name)(raw)


def build_whitebox_physics_iinn(
    feature_columns: list[str],
    l2: float = 1e-5,
    residual_bound_db: float = RESIDUAL_BOUND_DB,
    target_std_db: float = 1.0,
):
    """Completely decomposed physics-grounded IINN.

    The model is intentionally formula-like:

        RSRP = C
             + T(P_tx)
             - L_d,f(d, f)
             + G_ant(theta_h/B_h, theta_v/B_v, A_h, A_v)
             + H_site(heights, street, efficiency)
             + C_clutter(clutter class)
             + R_small(selected interactions), where R_small is explicitly
               bounded in dB to prevent the residual branch from becoming the
               primary prediction mechanism.

    Every term is exposed as a named output. The learned parameters adapt to
    3GPP, SPM, or measurement labels without hardcoding any one model equation.
    """
    reg = regularizers.l2(l2) if l2 else None
    residual_scale = float(residual_bound_db) / max(float(target_std_db), 1e-6)
    inputs = tf.keras.layers.Input(shape=(len(feature_columns),), name="features")
    normalizer = tf.keras.layers.Normalization(axis=-1, name="normalizer")
    z = ClipLayer(-5.0, 5.0, name="clipped_normalized_features")(normalizer(inputs))

    tx = gather(z, feature_columns, ["Max_Transmitter_Power"], "tx_input")
    distance = gather(z, feature_columns, ["log10_d3d", "log10_d2d", "log10_frequency"], "distance_frequency_input")
    alignment = gather(
        z,
        feature_columns,
        [
            "azimuth_error_over_hbw",
            "tilt_error_over_vbw",
            "Vertical_Attenuation",
            "Horizontal_Attenuation",
        ],
        "alignment_input",
    )
    site = gather(
        z,
        feature_columns,
        [
            "height_difference_m",
            "height_ratio",
            "building_clearance_m",
            "log10_street_width",
            "Efficiency",
        ],
        "site_input",
    )
    clutter_cols = [c for c in feature_columns if c.startswith("clutter=")]
    clutter = gather(inputs, feature_columns, clutter_cols, "clutter_one_hot_input")

    intercept = tf.keras.layers.Dense(
        1,
        activation="linear",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="intercept_term",
    )(tf.keras.layers.Lambda(lambda value: value[:, :1] * 0.0 + 1.0, name="constant_one")(inputs))

    tx_term = tf.keras.layers.Dense(
        1,
        activation="linear",
        kernel_initializer=tf.keras.initializers.Constant(0.25),
        bias_initializer="zeros",
        kernel_regularizer=reg,
        name="tx_power_term",
    )(tx)

    monotone_loss = tf.keras.layers.Dense(
        1,
        activation="linear",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(0.8),
        kernel_constraint=constraints.NonNeg(),
        kernel_regularizer=reg,
        name="monotone_distance_frequency_loss",
    )(tf.keras.layers.Lambda(lambda value: value + 5.0, name="nonnegative_distance_frequency")(distance))

    loss_curvature = bounded_linear(
        tf.keras.layers.Concatenate(name="path_loss_curvature_features")(
            [
                distance,
                tf.keras.layers.Multiply(name="logd_logf_interaction")(
                    [
                        tf.keras.layers.Lambda(lambda value: value[:, 0:1], name="logd_for_interaction")(distance),
                        tf.keras.layers.Lambda(lambda value: value[:, 2:3], name="logf_for_interaction")(distance),
                    ]
                ),
            ]
        ),
        units=1,
        scale=20.0,
        name="learned_path_loss_curvature",
    )
    path_loss = tf.keras.layers.Add(name="path_loss_term")([monotone_loss, loss_curvature])
    negative_path_loss = tf.keras.layers.Lambda(lambda value: -value, name="negative_path_loss_term")(path_loss)

    alignment_features = tf.keras.layers.Concatenate(name="alignment_formula_features")(
        [
            alignment,
            tf.keras.layers.Lambda(lambda value: tf.square(value[:, 0:1]), name="horizontal_alignment_square")(alignment),
            tf.keras.layers.Lambda(lambda value: tf.square(value[:, 1:2]), name="vertical_alignment_square")(alignment),
        ]
    )
    alignment_term = bounded_linear(alignment_features, 1, 35.0, "antenna_alignment_term")

    site_features = tf.keras.layers.Concatenate(name="site_formula_features")(
        [
            site,
            tf.keras.layers.Lambda(lambda value: value[:, 0:1] * value[:, 2:3], name="height_clearance_interaction")(site),
        ]
    )
    site_term = bounded_linear(site_features, 1, 45.0, "site_environment_term")

    clutter_term = bounded_linear(clutter, 1, 25.0, "clutter_offset_term")

    residual_inputs = tf.keras.layers.Concatenate(name="small_residual_inputs")(
        [distance, alignment, site]
    )
    residual_term = bounded_linear(residual_inputs, 1, residual_scale, "small_residual_term")

    physics_sum = tf.keras.layers.Add(name="whitebox_physics_sum")(
        [
            intercept,
            tx_term,
            negative_path_loss,
            alignment_term,
            site_term,
            clutter_term,
            residual_term,
        ]
    )
    output = tf.keras.layers.Dense(
        1,
        activation="linear",
        kernel_initializer=tf.keras.initializers.Ones(),
        bias_initializer="zeros",
        name="global_affine_calibration",
    )(physics_sum)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="WhiteBox_Physics_IINN")
    model.normalizer = normalizer
    model.feature_columns = feature_columns
    model.residual_bound_db = float(residual_bound_db)
    model.component_tensors = {
        "intercept_term": intercept,
        "tx_power_term": tx_term,
        "path_loss_term": path_loss,
        "negative_path_loss_term": negative_path_loss,
        "antenna_alignment_term": alignment_term,
        "site_environment_term": site_term,
        "clutter_offset_term": clutter_term,
        "small_residual_term": residual_term,
        "whitebox_physics_sum": physics_sum,
        "output": output,
    }
    return model


def component_model(model: tf.keras.Model) -> tf.keras.Model:
    return tf.keras.Model(model.inputs, model.component_tensors, name=f"{model.name}_components")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred.reshape(-1) - y_true.reshape(-1)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true.reshape(-1) - np.mean(y_true)) ** 2))
    return {
        "RMSE_dB": float(np.sqrt(np.mean(err**2))),
        "MAE_dB": float(np.mean(np.abs(err))),
        "MBE_dB": float(np.mean(err)),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def fit_affine_calibrator(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(y_pred.reshape(-1).astype(float), y_true.reshape(-1).astype(float), deg=1)
    return {"slope": float(np.clip(slope, 0.5, 1.8)), "intercept": float(intercept)}


def apply_calibrator(y_pred: np.ndarray, cal: dict[str, float]) -> np.ndarray:
    return cal["slope"] * y_pred.reshape(-1) + cal["intercept"]


def save_architecture_figure(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#E8F1FF", ec="#1F2937", fs=11, weight="normal"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            linewidth=1.2,
            facecolor=fc,
            edgecolor=ec,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)

    ax.text(7, 7.55, "Generalized White-Box Physics-IINN", ha="center", fontsize=20, weight="bold")
    ax.text(
        7,
        7.15,
        r"$\hat{P}_{rx}=C+T(P_{tx})-L(d,f)+G_{ant}(\Delta\theta/B)+H_{site}+C_{clutter}+R_{small}$",
        ha="center",
        fontsize=15,
    )

    branches = [
        ("Tx power\n$T(P_{tx})$", "Max Tx power", 0.7, "#DCFCE7"),
        ("Path loss\n$L(d,f)$", "log-distance\nlog-frequency", 2.8, "#FEE2E2"),
        ("Antenna alignment\n$G_{ant}$", "azimuth/tilt error\nbeamwidth, attenuation", 4.9, "#E0F2FE"),
        ("Site/environment\n$H_{site}$", "heights, street width\nbuilding clearance", 7.0, "#FEF3C7"),
        ("Clutter offset\n$C_{clutter}$", "one-hot clutter class", 9.1, "#F3E8FF"),
        ("Small residual\n$R_{small}$", "bounded interactions", 11.2, "#E5E7EB"),
    ]

    for title, inp, x, color in branches:
        box(x, 5.9, 1.8, 0.55, inp, fc="#FFFFFF", fs=9)
        box(x, 4.8, 1.8, 0.8, title, fc=color, fs=10, weight="bold")
        ax.annotate("", xy=(x + 0.9, 4.8), xytext=(x + 0.9, 5.9), arrowprops=dict(arrowstyle="->", lw=1.4))
        ax.annotate("", xy=(7, 3.65), xytext=(x + 0.9, 4.8), arrowprops=dict(arrowstyle="->", lw=1.2))

    box(5.4, 3.05, 3.2, 0.75, "Add named physical components", fc="#DBEAFE", fs=12, weight="bold")
    box(5.4, 1.9, 3.2, 0.65, "Transparent affine calibration\n(optional few-shot SPM)", fc="#F8FAFC", fs=11)
    box(5.4, 0.9, 3.2, 0.65, "Predicted RSRP", fc="#DCFCE7", fs=13, weight="bold")
    ax.annotate("", xy=(7, 1.9), xytext=(7, 3.05), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(7, 0.9 + 0.65), xytext=(7, 1.9), arrowprops=dict(arrowstyle="->", lw=1.5))

    box(0.7, 0.5, 3.2, 1.3, "White-box property:\neach branch has a physical name,\na sign/scale constraint, and exported\nper-sample contribution.", fc="#FFF7ED", fs=10)
    box(10.1, 0.5, 3.2, 1.3, "Generalization property:\nno fixed 3GPP equations, no LOS/NLOS\nmax rule, no simulator constants.", fc="#FFF7ED", fs=10)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_prediction_plots(y_true, y_pred, out_dir: Path, prefix: str):
    import matplotlib.pyplot as plt

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    err = y_pred - y_true

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    ax.set_xlabel("True RSRP (dB)")
    ax.set_ylabel("Predicted RSRP (dB)")
    ax.set_title(f"{prefix}: true vs predicted")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_true_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=50, alpha=0.85)
    ax.axvline(np.mean(err), color="black", lw=1, label=f"MBE={np.mean(err):.2f} dB")
    ax.set_xlabel("Prediction error, pred - true (dB)")
    ax.set_ylabel("Count")
    ax.set_title(f"{prefix}: error distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_error_histogram.png", dpi=180)
    plt.close(fig)


def save_component_outputs(model, x_df, y_mean, y_std, out_dir: Path, prefix: str):
    import matplotlib.pyplot as plt

    outputs = component_model(model).predict(x_df.to_numpy(np.float32), verbose=0)
    table = pd.DataFrame(index=x_df.index)
    summary_rows = []
    for name, values in outputs.items():
        values_db = values.reshape(-1) * y_std
        if name == "path_loss_term":
            values_db = -values_db
        if name == "output":
            values_db = values.reshape(-1) * y_std + y_mean
        table[name] = values_db
        if name != "output":
            summary_rows.append(
                {
                    "component": name,
                    "mean_dB": float(np.mean(values_db)),
                    "mean_abs_dB": float(np.mean(np.abs(values_db))),
                    "median_abs_dB": float(np.median(np.abs(values_db))),
                    "p95_abs_dB": float(np.quantile(np.abs(values_db), 0.95)),
                    "std_dB": float(np.std(values_db)),
                    "min_dB": float(np.min(values_db)),
                    "max_dB": float(np.max(values_db)),
                }
            )
    table.to_excel(out_dir / f"{prefix}_component_outputs.xlsx", index=False)
    summary = pd.DataFrame(summary_rows)
    manuscript_components = [
        "tx_power_term",
        "path_loss_term",
        "antenna_alignment_term",
        "site_environment_term",
        "clutter_offset_term",
        "small_residual_term",
    ]
    if not summary.empty:
        component_mask = summary["component"].isin(manuscript_components)
        total_abs = float(summary.loc[component_mask, "mean_abs_dB"].sum())
        summary["mean_abs_share_percent"] = np.where(
            component_mask & (total_abs > 0),
            100.0 * summary["mean_abs_dB"] / total_abs,
            np.nan,
        )
    summary.to_excel(out_dir / f"{prefix}_component_summary.xlsx", index=False)

    plot_summary = summary[summary["component"].isin(manuscript_components)].copy()
    plot_summary = plot_summary.sort_values("mean_dB")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(plot_summary["component"], plot_summary["mean_dB"])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean component contribution to RSRP (dB)")
    ax.set_title(f"{prefix}: white-box component decomposition")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_component_decomposition.png", dpi=180)
    plt.close(fig)

    abs_summary = plot_summary.sort_values("mean_abs_dB", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(abs_summary["component"], abs_summary["mean_abs_dB"], color="#2563eb")
    ax.set_xlabel("Mean absolute contribution (dB)")
    ax.set_title(f"{prefix}: component magnitude audit")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_component_magnitude_audit.png", dpi=220)
    plt.close(fig)

    boxplot_columns = [c for c in manuscript_components if c in table.columns]
    if boxplot_columns:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.boxplot([table[c].to_numpy() for c in boxplot_columns], labels=boxplot_columns, showfliers=False)
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
    if not plot_summary.empty and "small_residual_term" in set(plot_summary["component"]):
        residual_abs = float(
            plot_summary.loc[plot_summary["component"] == "small_residual_term", "mean_abs_dB"].iloc[0]
        )
        physics_abs = plot_summary.loc[
            plot_summary["component"].isin(
                ["path_loss_term", "antenna_alignment_term", "site_environment_term", "clutter_offset_term"]
            ),
            "mean_abs_dB",
        ]
        largest_physics_abs = float(physics_abs.max()) if len(physics_abs) else 0.0
        total_named_abs = float(plot_summary["mean_abs_dB"].sum())
        residual_max_abs = float(np.max(np.abs(table["small_residual_term"])))
        diagnostics.update(
            {
                "residual_mean_abs_dB": residual_abs,
                "largest_physics_component_mean_abs_dB": largest_physics_abs,
                "residual_to_largest_physics_ratio": residual_abs / max(largest_physics_abs, 1e-9),
                "residual_to_total_named_component_ratio": residual_abs / max(total_named_abs, 1e-9),
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
    return diagnostics


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_architecture_figure(OUT_DIR / "whitebox_physics_iinn_architecture.png")

    atoll = load_dataset(DATA_3GPP)
    spm = load_dataset(DATA_SPM)
    spm_cal_df, spm_eval_df = split_calibration_test(spm, SPM_CALIBRATION_FRACTION)
    train_df, val_df, iid_df = split_train_val_test(atoll)

    train_x, val_x, iid_x, spm_cal_x, spm_eval_x, metadata = build_tables(
        train_df, val_df, iid_df, spm_cal_df, spm_eval_df
    )

    y_train_raw = train_df[TARGET].to_numpy(np.float32).reshape(-1, 1)
    y_val_raw = val_df[TARGET].to_numpy(np.float32).reshape(-1, 1)
    y_iid = iid_df[TARGET].to_numpy(np.float32).reshape(-1)
    y_spm_cal = spm_cal_df[TARGET].to_numpy(np.float32).reshape(-1)
    y_spm_eval = spm_eval_df[TARGET].to_numpy(np.float32).reshape(-1)

    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw) + 1e-6)
    y_train = (y_train_raw - y_mean) / y_std
    y_val = (y_val_raw - y_mean) / y_std

    iid_preds = []
    spm_cal_preds = []
    spm_eval_preds = []
    histories = []
    last_model = None
    start = time.perf_counter()

    for run_no, seed in enumerate(ENSEMBLE_SEEDS, start=1):
        print(f"\n--- Training white-box Physics-IINN {run_no}/{len(ENSEMBLE_SEEDS)} seed={seed} ---")
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.keras.backend.clear_session()

        model = build_whitebox_physics_iinn(
            train_x.columns.tolist(),
            l2=2e-5,
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
            validation_data=(val_x.to_numpy(np.float32), y_val),
            epochs=300,
            batch_size=64,
            verbose=1,
            callbacks=cb,
        )
        history_df = pd.DataFrame(history.history)
        history_df["seed"] = seed
        histories.append(history_df)

        iid_preds.append(model.predict(iid_x.to_numpy(np.float32), verbose=0).reshape(-1) * y_std + y_mean)
        spm_cal_preds.append(model.predict(spm_cal_x.to_numpy(np.float32), verbose=0).reshape(-1) * y_std + y_mean)
        spm_eval_preds.append(model.predict(spm_eval_x.to_numpy(np.float32), verbose=0).reshape(-1) * y_std + y_mean)
        model.save(OUT_DIR / f"whitebox_physics_iinn_seed_{seed}.keras")
        last_model = model

    train_seconds = time.perf_counter() - start
    iid_pred = np.mean(np.vstack(iid_preds), axis=0)
    spm_cal_pred = np.mean(np.vstack(spm_cal_preds), axis=0)
    spm_eval_pred = np.mean(np.vstack(spm_eval_preds), axis=0)

    calibrator = fit_affine_calibrator(spm_cal_pred, y_spm_cal)
    spm_eval_pred_cal = apply_calibrator(spm_eval_pred, calibrator)

    iid_metrics = metrics(y_iid, iid_pred)
    spm_uncal_metrics = metrics(y_spm_eval, spm_eval_pred)
    spm_cal_metrics = metrics(y_spm_eval, spm_eval_pred_cal)
    uncal_degradation = 100.0 * (spm_uncal_metrics["RMSE_dB"] - iid_metrics["RMSE_dB"]) / iid_metrics["RMSE_dB"]
    cal_degradation = 100.0 * (spm_cal_metrics["RMSE_dB"] - iid_metrics["RMSE_dB"]) / iid_metrics["RMSE_dB"]

    pd.concat(histories, axis=0, ignore_index=True).to_excel(OUT_DIR / "training_history.xlsx", index=False)
    save_prediction_plots(y_spm_eval, spm_eval_pred, OUT_DIR, "SPM_OOD_uncalibrated")
    save_prediction_plots(y_spm_eval, spm_eval_pred_cal, OUT_DIR, "SPM_OOD_calibrated")
    component_audit = None
    if last_model is not None:
        component_audit = save_component_outputs(
            last_model, spm_eval_x, y_mean, y_std, OUT_DIR, "SPM_OOD_last_member"
        )

    pd.DataFrame(
        [
            {"split": "IID_3GPP", **iid_metrics},
            {"split": "SPM_OOD_uncalibrated", **spm_uncal_metrics, "degradation_percent": uncal_degradation},
            {"split": "SPM_OOD_calibrated", **spm_cal_metrics, "degradation_percent": cal_degradation},
        ]
    ).to_excel(OUT_DIR / "metrics.xlsx", index=False)

    pd.DataFrame(
        {
            "true_RSRP": y_spm_eval,
            "pred_uncalibrated": spm_eval_pred,
            "pred_calibrated": spm_eval_pred_cal,
            "error_uncalibrated": spm_eval_pred - y_spm_eval,
            "error_calibrated": spm_eval_pred_cal - y_spm_eval,
        }
    ).to_excel(OUT_DIR / "spm_ood_predictions.xlsx", index=False)

    results = {
        "model": "WhiteBox_Physics_IINN",
        "formula": "RSRP = C + T(Ptx) - L(d,f) + G_ant + H_site + C_clutter + R_small",
        "train_dataset": str(DATA_3GPP),
        "spm_dataset": str(DATA_SPM),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "iid_rows": int(len(iid_df)),
        "spm_calibration_rows": int(len(spm_cal_df)),
        "spm_eval_rows": int(len(spm_eval_df)),
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "calibrator": calibrator,
        "calibrator_parameter_count": 2,
        "residual_bound_db": RESIDUAL_BOUND_DB,
        "component_audit": component_audit,
        "target_mean": y_mean,
        "target_std": y_std,
        "train_seconds": train_seconds,
        "IID_3GPP": iid_metrics,
        "SPM_OOD_uncalibrated": spm_uncal_metrics,
        "SPM_OOD_calibrated": spm_cal_metrics,
        "OOD_degradation_uncalibrated_percent": float(uncal_degradation),
        "OOD_degradation_calibrated_percent": float(cal_degradation),
        "metadata": metadata,
    }
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== White-Box Physics-IINN Results ===")
    print(f"IID 3GPP RMSE:              {iid_metrics['RMSE_dB']:.3f} dB")
    print(f"SPM OOD RMSE uncalibrated:  {spm_uncal_metrics['RMSE_dB']:.3f} dB")
    print(f"SPM OOD MBE uncalibrated:   {spm_uncal_metrics['MBE_dB']:.3f} dB")
    print(f"Few-shot calibrator: slope={calibrator['slope']:.4f}, intercept={calibrator['intercept']:.4f}")
    print(f"SPM OOD RMSE calibrated:    {spm_cal_metrics['RMSE_dB']:.3f} dB")
    print(f"SPM OOD MBE calibrated:     {spm_cal_metrics['MBE_dB']:.3f} dB")
    print(f"Outputs saved to:           {OUT_DIR}")


if __name__ == "__main__":
    main()
