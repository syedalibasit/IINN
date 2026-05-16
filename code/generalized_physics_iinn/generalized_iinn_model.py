from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks, constraints, regularizers

    TENSORFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    tf = None
    callbacks = None
    constraints = None
    regularizers = None
    TENSORFLOW_AVAILABLE = False


EPSILON = 1e-6


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "experiment"


def _clean_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "feature"


def _as_frame(X: pd.DataFrame | np.ndarray, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy().astype(np.float32)

    values = np.asarray(X)
    if columns is None:
        columns = [f"x{i}" for i in range(values.shape[1])]
    return pd.DataFrame(values, columns=list(columns)).astype(np.float32)


def _as_target(y: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
    values = y.values if hasattr(y, "values") else np.asarray(y)
    return values.astype(np.float32).reshape(-1, 1)


def _find_first(columns: list[str], candidates: Iterable[str]) -> int | None:
    lowered = {column.lower(): index for index, column in enumerate(columns)}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _find_many(columns: list[str], tokens: Iterable[str]) -> list[int]:
    tokens = [token.lower() for token in tokens]
    found = []
    for index, column in enumerate(columns):
        name = column.lower()
        if any(token in name for token in tokens):
            found.append(index)
    return found


def _slice_columns(x, indices: list[int], name: str):
    if not indices:
        return None
    return tf.keras.layers.Lambda(
        lambda value, idx=indices: tf.gather(value, idx, axis=1),
        name=name,
    )(x)


def _dense_stack(x, widths: list[int], name: str, l2: float, dropout: float = 0.0):
    reg = regularizers.l2(l2) if l2 else None
    y = x
    for layer_no, width in enumerate(widths, start=1):
        y = tf.keras.layers.Dense(
            width,
            activation="swish",
            kernel_initializer="he_normal",
            kernel_regularizer=reg,
            name=f"{name}_dense_{layer_no}",
        )(y)
        if dropout:
            y = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_{layer_no}")(y)
    return y


def _zero_head(x, name: str, l2: float):
    reg = regularizers.l2(l2) if l2 else None
    return tf.keras.layers.Dense(
        1,
        activation="linear",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        kernel_regularizer=reg,
        name=name,
    )(x)


def _log10_positive(x, name: str):
    return tf.keras.layers.Lambda(
        lambda value: tf.math.log(tf.maximum(value, EPSILON)) / tf.math.log(tf.constant(10.0, value.dtype)),
        name=name,
    )(x)


def infer_iinn_feature_roles(feature_names: Iterable[str]) -> dict[str, list[int] | int | None]:
    """Infer broad radio-domain feature roles from column names.

    These roles are used only to decide which learned branch sees which inputs.
    They do not impose a 3GPP, SPM, COST, or Atoll equation.
    """
    columns = list(feature_names)
    tx_power = _find_first(columns, ["Tx_Pwr", "TxPower", "TransmitPower", "Pt", "Ptx"])
    distance = _find_first(columns, ["Distance", "distance", "d", "Link_Distance"])
    frequency = _find_first(columns, ["Frequency", "frequency", "freq", "Carrier_Frequency"])

    angle = sorted(
        set(
            _find_many(columns, ["azimuth", "tilt", "bearing", "downtilt"])
        )
    )
    height = sorted(
        set(
            _find_many(columns, ["height", "hbs", "hue", "building", "clutter"])
        )
    )
    environment = sorted(
        set(
            _find_many(columns, ["width", "street", "building", "clutter", "terrain", "morphology", "los", "nlos"])
        )
    )

    excluded = {index for index in [tx_power, distance, frequency] if index is not None}
    excluded.update(angle)
    excluded.update(height)
    excluded.update(environment)
    residual = [index for index in range(len(columns)) if index not in excluded]

    return {
        "tx_power": tx_power,
        "distance": distance,
        "frequency": frequency,
        "angle": angle,
        "height": height,
        "environment": environment,
        "residual": residual,
    }


def build_iinn_with_trace(
    feature_names: Iterable[str],
    *,
    branch_width: int = 24,
    residual_width: int = 32,
    l2: float = 1e-5,
    dropout: float = 0.05,
):
    """Build a generalized domain-informed neural model.

    The model keeps the radio-link structure:

        received power = learned Tx contribution
                         - learned propagation loss
                         + learned antenna/alignment effect
                         + learned site/environment effect
                         + small residual interaction

    Unlike the original IINN, every term is learned from data. There are no
    fixed 3GPP constants, no LOS/NLOS max equation, and no hard-coded path-loss
    coefficients. This allows the same architecture to train on labels generated
    by 3GPP, SPM, ray tracing, measurements, or another propagation model.
    """
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is not available; IINN cannot run.")

    feature_names = list(feature_names)
    if not feature_names:
        raise ValueError("feature_names cannot be empty.")

    roles = infer_iinn_feature_roles(feature_names)
    n_features = len(feature_names)
    trace = {"feature_roles": roles}

    inputs = tf.keras.layers.Input(shape=(n_features,), name="features")
    normalizer = tf.keras.layers.Normalization(axis=-1, name="feature_normalization")
    x_norm = normalizer(inputs)
    trace["normalized_features"] = x_norm

    terms = []

    tx_index = roles["tx_power"]
    if tx_index is not None:
        tx = _slice_columns(x_norm, [tx_index], "tx_power_slice")
        tx_term = tf.keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer=tf.keras.initializers.Ones(),
            bias_initializer="zeros",
            name="learned_tx_power_term",
        )(tx)
    else:
        tx_term = tf.keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="learned_intercept_without_tx_power",
        )(tf.keras.layers.Lambda(lambda value: value[:, :1] * 0.0, name="zero_reference")(x_norm))
    terms.append(tx_term)
    trace["tx_power_term"] = tx_term

    propagation_inputs = []
    monotone_inputs = []
    distance_index = roles["distance"]
    frequency_index = roles["frequency"]

    if distance_index is not None:
        raw_distance = _slice_columns(inputs, [distance_index], "distance_raw_slice")
        log_distance = _log10_positive(raw_distance, "log10_distance")
        propagation_inputs.append(log_distance)
        monotone_inputs.append(log_distance)
        trace["log10_distance"] = log_distance

    if frequency_index is not None:
        raw_frequency = _slice_columns(inputs, [frequency_index], "frequency_raw_slice")
        log_frequency = _log10_positive(raw_frequency, "log10_frequency")
        propagation_inputs.append(log_frequency)
        monotone_inputs.append(log_frequency)
        trace["log10_frequency"] = log_frequency

    height_x = _slice_columns(x_norm, roles["height"], "height_feature_slice")
    environment_x = _slice_columns(x_norm, roles["environment"], "environment_feature_slice")
    if height_x is not None:
        propagation_inputs.append(height_x)
        trace["height_features"] = height_x
    if environment_x is not None:
        propagation_inputs.append(environment_x)
        trace["environment_features"] = environment_x

    if propagation_inputs:
        if len(propagation_inputs) == 1:
            propagation_x = propagation_inputs[0]
        else:
            propagation_x = tf.keras.layers.Concatenate(name="propagation_feature_stack")(propagation_inputs)
        propagation_hidden = _dense_stack(
            propagation_x,
            [branch_width, branch_width],
            "learned_propagation",
            l2,
            dropout,
        )
        learned_loss = tf.keras.layers.Dense(
            1,
            activation="softplus",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            kernel_regularizer=regularizers.l2(l2) if l2 else None,
            name="learned_positive_path_loss",
        )(propagation_hidden)
    else:
        learned_loss = tf.keras.layers.Lambda(lambda value: value[:, :1] * 0.0, name="zero_path_loss")(x_norm)

    if monotone_inputs:
        monotone_x = monotone_inputs[0] if len(monotone_inputs) == 1 else tf.keras.layers.Concatenate(
            name="monotone_path_trend_inputs"
        )(monotone_inputs)
        monotone_loss = tf.keras.layers.Dense(
            1,
            activation="linear",
            use_bias=False,
            kernel_constraint=constraints.NonNeg(),
            kernel_initializer=tf.keras.initializers.Constant(1.0),
            name="monotone_distance_frequency_trend",
        )(monotone_x)
        learned_loss = tf.keras.layers.Add(name="total_learned_path_loss")([learned_loss, monotone_loss])
        trace["monotone_distance_frequency_trend"] = monotone_loss

    negative_loss = tf.keras.layers.Lambda(lambda value: -value, name="negative_path_loss")(learned_loss)
    terms.append(negative_loss)
    trace["learned_path_loss"] = learned_loss

    angle_x = _slice_columns(x_norm, roles["angle"], "angle_feature_slice")
    if angle_x is not None:
        angle_hidden = _dense_stack(angle_x, [branch_width, max(4, branch_width // 2)], "learned_alignment", l2)
        alignment_term = _zero_head(angle_hidden, "learned_alignment_term", l2)
        terms.append(alignment_term)
        trace["alignment_term"] = alignment_term

    site_indices = sorted(set(roles["height"]) | set(roles["environment"]))
    site_x = _slice_columns(x_norm, site_indices, "site_context_slice")
    if site_x is not None:
        site_hidden = _dense_stack(site_x, [branch_width, max(4, branch_width // 2)], "learned_site_context", l2)
        site_term = _zero_head(site_hidden, "learned_site_context_term", l2)
        terms.append(site_term)
        trace["site_context_term"] = site_term

    residual_hidden = _dense_stack(x_norm, [residual_width, max(4, residual_width // 2)], "small_residual", l2, dropout)
    residual_term = _zero_head(residual_hidden, "small_residual_interaction_term", l2)
    terms.append(residual_term)
    trace["residual_interaction_term"] = residual_term

    prediction = tf.keras.layers.Add(name="domain_informed_prediction")(terms)
    calibrated = tf.keras.layers.Dense(
        1,
        activation="linear",
        kernel_initializer=tf.keras.initializers.Ones(),
        bias_initializer="zeros",
        name="final_calibration",
    )(prediction)
    trace["prediction_before_calibration"] = prediction
    trace["output"] = calibrated

    model = tf.keras.Model(inputs=inputs, outputs=calibrated, name="Generalized_IINN")
    model.normalizer = normalizer
    model.feature_names = feature_names
    model.trace_tensors = trace
    return model, trace


def make_trace_model(model: tf.keras.Model) -> tf.keras.Model:
    if not hasattr(model, "trace_tensors"):
        raise ValueError("The supplied model does not expose trace_tensors.")
    tensor_outputs = {
        key: value
        for key, value in model.trace_tensors.items()
        if not isinstance(value, dict)
    }
    return tf.keras.Model(model.inputs, tensor_outputs, name=f"{model.name}_trace")


def save_training_curve(history_df: pd.DataFrame, path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history_df.index + 1, history_df["loss"], label="train")
        if "val_loss" in history_df:
            ax.plot(history_df.index + 1, history_df["val_loss"], label="validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
    except Exception:
        pass


def train_predict_iinn(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    out_model_dir: Path,
    exp_name: str,
    *,
    branch_width: int = 24,
    residual_width: int = 32,
    learning_rate: float = 1e-3,
    epochs: int = 250,
    batch_size: int = 64,
    l2: float = 1e-5,
    dropout: float = 0.05,
):
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is not available; IINN cannot run.")

    out_model_dir = Path(out_model_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)

    X_train_df = _as_frame(X_train)
    feature_names = list(X_train_df.columns)
    X_val_df = _as_frame(X_val, feature_names)[feature_names]
    X_test_df = _as_frame(X_test, feature_names)[feature_names]

    X_train_np = X_train_df.to_numpy(np.float32)
    X_val_np = X_val_df.to_numpy(np.float32)
    X_test_np = X_test_df.to_numpy(np.float32)
    y_train_np = _as_target(y_train)
    y_val_np = _as_target(y_val)

    tf.keras.backend.clear_session()
    model, trace = build_iinn_with_trace(
        feature_names,
        branch_width=branch_width,
        residual_width=residual_width,
        l2=l2,
        dropout=dropout,
    )
    model.normalizer.adapt(X_train_np)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
        callbacks.TerminateOnNaN(),
    ]

    start = time.perf_counter()
    history = model.fit(
        X_train_np,
        y_train_np,
        validation_data=(X_val_np, y_val_np),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=cb,
    )
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test_np, verbose=0).ravel()
    pred_time = time.perf_counter() - start

    experiment = safe_name(exp_name)
    history_df = pd.DataFrame(history.history)
    history_df.to_excel(out_model_dir / f"{experiment}_Generalized_IINN_training_history.xlsx", index=False)
    save_training_curve(
        history_df,
        out_model_dir / f"{experiment}_Generalized_IINN_training_curve.png",
        f"Generalized IINN Training Curve - {exp_name}",
    )

    rows = []
    for layer in model.layers:
        for weight_index, weight in enumerate(layer.get_weights()):
            rows.append(
                {
                    "Layer": layer.name,
                    "Weight_Index": weight_index,
                    "Shape": str(weight.shape),
                    "Mean": float(np.mean(weight)),
                    "Std": float(np.std(weight)),
                    "Min": float(np.min(weight)),
                    "Max": float(np.max(weight)),
                    "Trainable": layer.trainable,
                }
            )
    if rows:
        pd.DataFrame(rows).to_excel(
            out_model_dir / f"{experiment}_Generalized_IINN_weights_summary.xlsx",
            index=False,
        )

    role_rows = []
    for role, indices in trace["feature_roles"].items():
        if indices is None:
            continue
        if isinstance(indices, int):
            indices = [indices]
        for index in indices:
            role_rows.append({"Role": role, "Feature_Index": index, "Feature": feature_names[index]})
    pd.DataFrame(role_rows).to_excel(
        out_model_dir / f"{experiment}_Generalized_IINN_feature_roles.xlsx",
        index=False,
    )

    return model, y_pred, train_time, pred_time, history_df
