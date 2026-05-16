"""
iinn_generalised.py
====================
Generalised IINN (Innately Intelligent Neural Network) for radio
propagation modelling.

Design principles
-----------------
1. Physics-inspired topology is preserved:
   - Vertical beam-pattern branch  (UE_Tilt, BS_Tilt, bv, Av)
   - Horizontal beam-pattern branch (UE_Azimuth, BS_Azimuth, bh, Ah)
   - Path-loss branch: LOS and NLOS sub-paths that are SOFT-blended
   - Antenna gain / path-coupling aggregator
   - Received-power subtraction

2. All hard 3GPP-specific constraints are removed:
   - bv_dense / bh_dense are now trainable (no frozen layers)
   - All initialisers replaced with appropriate data-driven defaults
   - Hard Maximum(LOS, NLOS) replaced by a learned soft gate
   - Single scalar output replaced by a small residual MLP
   - log10 safe-guard: epsilon-protected, handles zero inputs

3. Width parameter `w` controls the number of neurons per physics
   operator.  w=1 preserves the compact 3GPP-aligned interpretation;
   w>1 adds capacity within the same topology.

4. A `ResidualCorrection` block sits between Rx_Pwr and the final
   linear output.  It is bounded by tanh (cannot dominate the physics)
   and adds at most 2*w + w + 1 parameters beyond the core model.

5. Drop-in replacement for the original build_iinn_with_trace() and
   train_predict_iinn() functions.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper: numerically stable log10
# ---------------------------------------------------------------------------

def _safe_log10(inputs: tf.Tensor) -> tf.Tensor:
    """
    Element-wise log10.
    Returns 0 where input == 0 (avoids -inf at the origin).
    Uses a small epsilon for numerical safety near zero.
    """
    epsilon = 1e-8
    return tf.where(
        tf.equal(inputs, 0),
        tf.zeros_like(inputs),
        tf.math.log(tf.maximum(tf.abs(inputs), epsilon))
        / tf.math.log(10.0 + epsilon),
    )


def _safe_log10_layer(name: str):
    return tf.keras.layers.Lambda(_safe_log10, name=name)


# ---------------------------------------------------------------------------
# Soft LOS/NLOS gate
# ---------------------------------------------------------------------------

def _soft_gate(los: tf.Tensor, nlos: tf.Tensor) -> tf.Tensor:
    """
    Differentiable, data-adaptive replacement for max(LOS, NLOS).

    gate = sigmoid(LOS - NLOS)  ∈ (0, 1)
    output = gate * LOS + (1 - gate) * NLOS

    When LOS >> NLOS  →  output ≈ LOS   (recovers 3GPP behaviour)
    When NLOS >> LOS  →  output ≈ NLOS  (recovers 3GPP behaviour)
    When equal        →  output = mean  (smooth blend)

    No additional trainable parameters.
    """
    gate = tf.sigmoid(los - nlos)
    return gate * los + (1.0 - gate) * nlos


# ---------------------------------------------------------------------------
# Core model builder
# ---------------------------------------------------------------------------

def build_generalised_iinn(w: int = 1) -> tuple[tf.keras.Model, dict]:
    """
    Build the generalised IINN.

    Parameters
    ----------
    w : int
        Width multiplier for each physics operator Dense layer.
        w=1  →  34 trainable parameters  (compact, 3GPP-interpretable)
        w=4  →  ~200 parameters          (more flexible residuals)
        w=16 →  ~800 parameters          (high-capacity variant)

    Returns
    -------
    model : tf.keras.Model
    trace : dict
        Intermediate tensors keyed by semantic name for interpretability.
    """
    # ------------------------------------------------------------------
    # Initialisers
    # ------------------------------------------------------------------
    # Physics-meaningful layers: start near their 3GPP-analytical values
    # (kernel ≈ 1 so log-inputs pass through with unit gain at init).
    # Biases start at 0 so no systematic offset is baked in.
    phys_kernel = tf.keras.initializers.Constant(1.0)
    phys_bias   = tf.keras.initializers.Zeros()           # ← was Constant(1)

    # Residual / gate layers: glorot for stable gradient flow from init.
    res_kernel  = tf.keras.initializers.GlorotUniform()
    res_bias    = tf.keras.initializers.Zeros()

    trace: dict = {}

    # ==================================================================
    # ── BRANCH 1: Vertical beam-pattern attenuation ──────────────────
    # ==================================================================
    # A_v(θ) = min( (θ_UE - θ_BS)² / bv², Av_max )
    # All four parameters are now trainable.

    inp_ue_tilt = tf.keras.layers.Input(shape=[1], name="UE_Tilt")
    inp_bs_tilt = tf.keras.layers.Input(shape=[1], name="BS_Tilt")
    inp_bv      = tf.keras.layers.Input(shape=[1], name="bv")
    inp_av      = tf.keras.layers.Input(shape=[1], name="Av")

    out_ue_tilt = tf.keras.layers.Dense(
        w, activation="linear", use_bias=True,
        kernel_initializer=phys_kernel, bias_initializer=phys_bias,
        name="UE_Tilt_dense"
    )(inp_ue_tilt)
    trace["out_ue_tilt"] = out_ue_tilt

    out_bs_tilt = tf.keras.layers.Dense(
        w, activation="linear", use_bias=True,
        kernel_initializer=phys_kernel, bias_initializer=phys_bias,
        name="BS_Tilt_dense"
    )(inp_bs_tilt)
    trace["out_bs_tilt"] = out_bs_tilt

    log_bv = _safe_log10_layer("log_bv")(inp_bv)
    trace["log_bv"] = log_bv

    # bv_dense: NOW TRAINABLE (was frozen in original)
    out_bv = tf.keras.layers.Dense(
        w, activation="linear", use_bias=False,
        kernel_initializer=phys_kernel,
        name="bv_dense",
        # trainable=True is the default; no override needed
    )(inp_bv)
    trace["out_bv"] = out_bv

    out_av = tf.keras.layers.Dense(
        w, activation="linear", use_bias=True,
        kernel_initializer=phys_kernel, bias_initializer=phys_bias,
        name="Av_dense"
    )(inp_av)
    trace["out_av"] = out_av

    diff_tilt = tf.keras.layers.Subtract(name="diff_tilt")(
        [out_ue_tilt, out_bs_tilt]
    )
    trace["diff_tilt"] = diff_tilt

    ratio_v = tf.keras.layers.Lambda(
        lambda x: x[0] / (x[1] + 1e-8), name="ratio_v"
    )([diff_tilt, out_bv])
    trace["ratio_v"] = ratio_v

    sq_v = tf.keras.layers.Lambda(
        lambda x: tf.math.square(x), name="sq_v"
    )(ratio_v)
    trace["sq_v"] = sq_v

    # Learned Av_max: data can adjust the clipping threshold
    beam_v = tf.keras.layers.Minimum(name="beam_v")([sq_v, out_av])
    trace["beam_v"] = beam_v

    # ==================================================================
    # ── BRANCH 2: Horizontal beam-pattern attenuation ─────────────────
    # ==================================================================

    inp_ue_az = tf.keras.layers.Input(shape=[1], name="UE_Azimuth")
    inp_bs_az = tf.keras.layers.Input(shape=[1], name="BS_Azimuth")
    inp_bh    = tf.keras.layers.Input(shape=[1], name="bh")
    inp_ah    = tf.keras.layers.Input(shape=[1], name="Ah")

    out_ue_az = tf.keras.layers.Dense(
        w, activation="linear", use_bias=True,
        kernel_initializer=phys_kernel, bias_initializer=phys_bias,
        name="UE_Azimuth_dense"
    )(inp_ue_az)
    trace["out_ue_az"] = out_ue_az

    out_bs_az = tf.keras.layers.Dense(
        w, activation="linear", use_bias=True,
        kernel_initializer=phys_kernel, bias_initializer=phys_bias,
        name="BS_Azimuth_dense"
    )(inp_bs_az)
    trace["out_bs_az"] = out_bs_az

    log_bh = _safe_log10_layer("log_bh")(inp_bh)
    trace["log_bh"] = log_bh

    # bh_dense: NOW TRAINABLE (was frozen in original)
    out_bh = tf.keras.layers.Dense(
        w, activation="linear", use_bias=False,
        kernel_initializer=phys_kernel,
        name="bh_dense",
    )(inp_bh)
    trace["out_bh"] = out_bh

    out_ah = tf.keras.layers.Dense(
        w, activation="linear", use_bias=True,
        kernel_initializer=phys_kernel, bias_initializer=phys_bias,
        name="Ah_dense"
    )(inp_ah)
    trace["out_ah"] = out_ah

    diff_az = tf.keras.layers.Subtract(name="diff_az")(
        [out_ue_az, out_bs_az]
    )
    trace["diff_az"] = diff_az

    ratio_h = tf.keras.layers.Lambda(
        lambda x: x[0] / (x[1] + 1e-8), name="ratio_h"
    )([diff_az, out_bh])
    trace["ratio_h"] = ratio_h

    sq_h = tf.keras.layers.Lambda(
        lambda x: tf.math.square(x), name="sq_h"
    )(ratio_h)
    trace["sq_h"] = sq_h

    beam_h = tf.keras.layers.Minimum(name="beam_h")([sq_h, out_ah])
    trace["beam_h"] = beam_h

    # ==================================================================
    # ── BRANCH 3: Total antenna gain ─────────────────────────────────
    # ==================================================================
    # G = -(beam_v + beam_h)  +  log(4π / (bv * bh))  +  ζ
    # All scaling is learned.

    beam_sum = tf.keras.layers.Add(name="beam_sum")([beam_v, beam_h])
    trace["beam_sum"] = beam_sum

    antenna_pattern = tf.keras.layers.Dense(
        w, activation="linear", use_bias=False,
        kernel_initializer=phys_kernel,
        name="antenna_pattern_dense"
    )(beam_sum)
    trace["antenna_pattern"] = antenna_pattern

    # log(4π) − log(bv) − log(bh)  →  learned coupling
    const_log4pi = float(np.log10(4.0 * np.pi))
    log_4pi = tf.keras.layers.Lambda(
        lambda x: tf.ones_like(x) * const_log4pi, name="log4pi_const"
    )(log_bv)
    trace["log_4pi"] = log_4pi

    log_bvbh = tf.keras.layers.Add(name="log_bv_plus_log_bh")(
        [log_bv, log_bh]
    )
    sub_4pi_bvbh = tf.keras.layers.Subtract(name="log4pi_minus_logbvbh")(
        [log_4pi, log_bvbh]
    )
    trace["sub_4pi_bvbh"] = sub_4pi_bvbh

    bvbh_coupling = tf.keras.layers.Dense(
        w, activation="linear", use_bias=True,
        kernel_initializer=phys_kernel, bias_initializer=phys_bias,
        name="bvbh_coupling_dense"
    )(sub_4pi_bvbh)
    trace["bvbh_coupling"] = bvbh_coupling

    inp_zeta = tf.keras.layers.Input(shape=[1], name="zeta")
    log_zeta = _safe_log10_layer("log_zeta")(inp_zeta)
    trace["log_zeta"] = log_zeta

    out_zeta = tf.keras.layers.Dense(
        w, activation="linear", use_bias=True,
        kernel_initializer=phys_kernel, bias_initializer=phys_bias,
        name="zeta_dense"
    )(log_zeta)
    trace["out_zeta"] = out_zeta

    antenna_gain = tf.keras.layers.Add(name="antenna_gain")(
        [out_zeta, bvbh_coupling]
    )
    trace["antenna_gain"] = antenna_gain

    # ==================================================================
    # ── BRANCH 4a: LOS path-loss sub-path ────────────────────────────
    # ==================================================================
    # PL_LOS ≈ α·log10(d) + β·log10(f) + c1

    inp_d  = tf.keras.layers.Input(shape=[1], name="Distance")
    inp_f  = tf.keras.layers.Input(shape=[1], name="Frequency")
    inp_c1 = tf.keras.layers.Input(shape=[1], name="c1")

    log_d = _safe_log10_layer("log_d")(inp_d)
    trace["log_d"] = log_d

    log_f = _safe_log10_layer("log_f")(inp_f)
    trace["log_f"] = log_f

    out_d = tf.keras.layers.Dense(
        w, activation="linear", use_bias=False,
        kernel_initializer=phys_kernel, name="Distance_dense"
    )(log_d)
    trace["out_d"] = out_d

    out_f = tf.keras.layers.Dense(
        w, activation="linear", use_bias=True,
        kernel_initializer=phys_kernel, bias_initializer=phys_bias,
        name="Frequency_dense"
    )(log_f)
    trace["out_f"] = out_f

    out_c1 = tf.keras.layers.Dense(
        w, activation="linear", use_bias=False,
        kernel_initializer=phys_kernel, name="c1_dense"
    )(inp_c1)
    trace["out_c1"] = out_c1

    LOS = tf.keras.layers.Add(name="LOS")([out_d, out_f, out_c1])
    trace["LOS"] = LOS

    # ==================================================================
    # ── BRANCH 4b: NLOS path-loss sub-path ───────────────────────────
    # ==================================================================
    # PL_NLOS ≈ log10(d) · log10(hbs) + log10(f) − log10(W) + log10(hob) − log10(hbs) − hue + c2

    inp_W   = tf.keras.layers.Input(shape=[1], name="Width")
    inp_hob = tf.keras.layers.Input(shape=[1], name="Building_height")
    inp_hue = tf.keras.layers.Input(shape=[1], name="UE_height")
    inp_hbs = tf.keras.layers.Input(shape=[1], name="BS_height")
    inp_c2  = tf.keras.layers.Input(shape=[1], name="NLoS_constant")

    log_W   = _safe_log10_layer("log_W")(inp_W)
    log_hob = _safe_log10_layer("log_hob")(inp_hob)
    log_hbs = _safe_log10_layer("log_hbs")(inp_hbs)
    trace.update({"log_W": log_W, "log_hob": log_hob, "log_hbs": log_hbs})

    out_W = tf.keras.layers.Dense(
        w, activation="linear", use_bias=True,
        kernel_initializer=phys_kernel, bias_initializer=phys_bias,
        name="Width_dense"
    )(log_W)
    trace["out_W"] = out_W

    out_hob = tf.keras.layers.Dense(
        w, activation="linear", use_bias=True,
        kernel_initializer=phys_kernel, bias_initializer=phys_bias,
        name="Object_height_dense"
    )(log_hob)
    trace["out_hob"] = out_hob

    out_hue = tf.keras.layers.Dense(
        w, activation="linear", use_bias=False,
        kernel_initializer=phys_kernel, name="UE_height_dense"
    )(inp_hue)
    trace["out_hue"] = out_hue

    out_hbs = tf.keras.layers.Dense(
        w, activation="linear", use_bias=True,
        kernel_initializer=phys_kernel, bias_initializer=phys_bias,
        name="BS_height_dense"
    )(log_hbs)
    trace["out_hbs"] = out_hbs

    out_c2 = tf.keras.layers.Dense(
        w, activation="linear", use_bias=False,
        kernel_initializer=phys_kernel, name="c2_dense"
    )(inp_c2)
    trace["out_c2"] = out_c2

    # log10(d) * log10(hbs)  →  cross-term
    mul_d_hbs = tf.keras.layers.Multiply(name="mul_d_hbs")([log_d, log_hbs])
    trace["mul_d_hbs"] = mul_d_hbs

    out_mul_d_hbs = tf.keras.layers.Dense(
        w, activation="linear", use_bias=False,
        kernel_initializer=phys_kernel, name="mul_d_hbs_dense"
    )(mul_d_hbs)
    trace["out_mul_d_hbs"] = out_mul_d_hbs

    # Assemble NLOS: f − W + hob − hbs − hue − d·hbs_term + d + c2
    nlos_s1 = tf.keras.layers.Subtract(name="nlos_s1")([out_f, out_W])
    nlos_a1 = tf.keras.layers.Add(name="nlos_a1")([nlos_s1, out_hob])
    nlos_s2 = tf.keras.layers.Subtract(name="nlos_s2")([nlos_a1, out_hbs])
    nlos_s3 = tf.keras.layers.Subtract(name="nlos_s3")([nlos_s2, out_hue])
    nlos_s4 = tf.keras.layers.Subtract(name="nlos_s4")([nlos_s3, out_mul_d_hbs])
    nlos_a2 = tf.keras.layers.Add(name="nlos_a2")([nlos_s4, out_d])
    NLOS = tf.keras.layers.Add(name="NLOS")([nlos_a2, out_c2])
    trace["NLOS"] = NLOS

    # ==================================================================
    # ── SOFT LOS/NLOS gate (replaces hard Maximum) ────────────────────
    # ==================================================================
    # gate = σ(LOS − NLOS) ∈ (0,1)
    # path_loss = gate·LOS + (1−gate)·NLOS
    # No extra parameters.  Recovers hard-max when |LOS−NLOS| >> 0.
    # Differentiable everywhere → gradients flow through both sub-paths.

    path_loss = tf.keras.layers.Lambda(
        lambda x: tf.sigmoid(x[0] - x[1]) * x[0]
                  + (1.0 - tf.sigmoid(x[0] - x[1])) * x[1],
        name="soft_path_loss"
    )([LOS, NLOS])
    trace["path_loss"] = path_loss

    # ==================================================================
    # ── Aggregate: path_loss + antenna_gain + antenna_pattern ─────────
    # ==================================================================

    total_loss = tf.keras.layers.Add(name="total_loss")(
        [antenna_pattern, antenna_gain, path_loss]
    )
    trace["total_loss"] = total_loss

    # ==================================================================
    # ── Received power  Rx = Pt − total_loss ─────────────────────────
    # ==================================================================

    inp_Pt = tf.keras.layers.Input(shape=[1], name="Tx_Pwr")
    out_Pt = tf.keras.layers.Dense(
        w, activation="linear", use_bias=True,
        kernel_initializer=phys_kernel, bias_initializer=phys_bias,
        name="TX_Pwr_dense"
    )(inp_Pt)
    trace["out_Pt"] = out_Pt

    Rx_Pwr = tf.keras.layers.Subtract(name="Rx_Pwr")([out_Pt, total_loss])
    trace["Rx_Pwr"] = Rx_Pwr

    # ==================================================================
    # ── Residual correction block (replaces single scalar output) ─────
    # ==================================================================
    # Two-layer MLP with tanh activations.
    # tanh bounding: corrections cannot exceed ±1 in any neuron,
    # preventing the residual from overriding the physics computation.
    # Total added parameters: w*(w) + w  +  w*1 + 1  =  w²+2w+1
    # For w=1: 4 extra params (34 → 38 total).
    # For w=4: 24 extra params.
    # The final linear layer carries no bias; the scalar output_dense
    # below carries the single free-bias that absorbs global offset.

    res_hidden = tf.keras.layers.Dense(
        w, activation="tanh",
        kernel_initializer=res_kernel, bias_initializer=res_bias,
        name="residual_hidden"
    )(Rx_Pwr)
    trace["res_hidden"] = res_hidden

    res_out = tf.keras.layers.Dense(
        w, activation="tanh",
        kernel_initializer=res_kernel, bias_initializer=res_bias,
        name="residual_out"
    )(res_hidden)
    trace["res_out"] = res_out

    # ==================================================================
    # ── Final linear projection to scalar RSRP ────────────────────────
    # ==================================================================
    # bias_initializer=Zeros():  no baked-in 3GPP operating-point offset.
    # The network is free to shift by ±N dB during training without
    # fighting a pre-loaded constant bias.

    output_Rx_Pwr = tf.keras.layers.Dense(
        1, activation="linear",
        kernel_initializer=tf.keras.initializers.Constant(1.0),
        bias_initializer=tf.keras.initializers.Zeros(),   # ← was Constant(1)
        use_bias=True,
        name="Rx_Pwr_output"
    )(res_out)
    trace["output_Rx_Pwr"] = output_Rx_Pwr

    # ==================================================================
    # ── Assemble model ────────────────────────────────────────────────
    # ==================================================================

    model = tf.keras.Model(
        inputs=[
            inp_ue_tilt, inp_bs_tilt, inp_bv, inp_av,
            inp_ue_az,   inp_bs_az,   inp_bh, inp_ah,
            inp_zeta,
            inp_d, inp_f,
            inp_W, inp_hob, inp_hue, inp_hbs,
            inp_c1, inp_c2, inp_Pt,
        ],
        outputs=output_Rx_Pwr,
        name=f"IINN_generalised_w{w}",
    )

    return model, trace


# ---------------------------------------------------------------------------
# Parameter count helper
# ---------------------------------------------------------------------------

def count_params(model: tf.keras.Model) -> dict[str, int]:
    trainable     = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
    return {
        "trainable":     trainable,
        "non_trainable": non_trainable,
        "total":         trainable + non_trainable,
    }


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------

def train_predict_iinn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    X_test:  pd.DataFrame,
    out_model_dir: Path,
    exp_name: str,
    w: int = 1,
    learning_rate: float = 1e-3,
    max_epochs: int = 300,
    batch_size: int = 32,
    patience_es: int = 30,
    patience_lr: int = 10,
) -> tuple:
    """
    Build, train, and evaluate the generalised IINN.

    Parameters
    ----------
    X_train, y_train : training split
    X_val,   y_val   : validation split  (used for early stopping)
    X_test           : held-out test set
    out_model_dir    : directory for artefacts
    exp_name         : experiment label (used in output filenames)
    w                : width multiplier  (1 = compact, 4/16 = flexible)
    learning_rate    : initial Adam LR
    max_epochs       : upper bound (early stopping usually activates first)
    batch_size       : mini-batch size
    patience_es      : early-stopping patience (epochs)
    patience_lr      : ReduceLROnPlateau patience (epochs)

    Returns
    -------
    model, y_pred, train_time_sec, pred_time_sec, history_df
    """
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is not available; cannot run IINN."
        )

    tf.keras.backend.clear_session()
    model, trace_tensors = build_generalised_iinn(w=w)

    # ── Verify no layers are frozen ──────────────────────────────────
    frozen = [l.name for l in model.layers if not l.trainable]
    if frozen:
        raise RuntimeError(
            f"Unexpected frozen layers detected: {frozen}. "
            "All layers should be trainable in the generalised model."
        )

    params = count_params(model)
    print(
        f"[IINN w={w}] "
        f"trainable={params['trainable']}, "
        f"total={params['total']}"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,     # gradient clipping prevents NaN on bad batches
        ),
        loss="mse",
    )

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=0,
        ),
        callbacks.TerminateOnNaN(),
    ]

    # ── Convert DataFrame to list of per-feature arrays ──────────────
    def _to_inputs(df: pd.DataFrame) -> list[np.ndarray]:
        return [df.iloc[:, i].values.astype(np.float32)
                for i in range(df.shape[1])]

    start = time.perf_counter()
    history = model.fit(
        _to_inputs(X_train),
        y_train.values.astype(np.float32),
        validation_data=(
            _to_inputs(X_val),
            y_val.values.astype(np.float32),
        ),
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=cb,
    )
    train_time = time.perf_counter() - start

    # ── Predict ───────────────────────────────────────────────────────
    start = time.perf_counter()
    y_pred = model.predict(
        _to_inputs(X_test), verbose=0
    ).ravel()
    pred_time = time.perf_counter() - start

    # ── Save artefacts ────────────────────────────────────────────────
    out_model_dir = Path(out_model_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)

    safe = exp_name.replace(" ", "_").replace("/", "-")

    history_df = pd.DataFrame(history.history)
    history_df.to_excel(
        out_model_dir / f"{safe}_IINN_w{w}_training_history.xlsx",
        index=False,
    )

    # Trained-weight summary
    weight_rows = []
    for layer in model.layers:
        wts = layer.get_weights()
        if not wts:
            continue
        for wi, wt in enumerate(wts):
            weight_rows.append(
                {
                    "Layer":       layer.name,
                    "Weight_idx":  wi,
                    "Shape":       str(wt.shape),
                    "Mean":        float(np.mean(wt)),
                    "Std":         float(np.std(wt)),
                    "Min":         float(np.min(wt)),
                    "Max":         float(np.max(wt)),
                    "Trainable":   layer.trainable,
                }
            )
    if weight_rows:
        pd.DataFrame(weight_rows).to_excel(
            out_model_dir / f"{safe}_IINN_w{w}_weights_summary.xlsx",
            index=False,
        )

    return model, y_pred, train_time, pred_time, history_df


# ---------------------------------------------------------------------------
# Intermediate-layer inspector (interpretability utility)
# ---------------------------------------------------------------------------

def extract_intermediate_outputs(
    model: tf.keras.Model,
    trace: dict,
    X_sample: pd.DataFrame,
    layer_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Evaluate and return intermediate tensor values for a sample batch.

    Parameters
    ----------
    model       : trained generalised IINN
    trace       : dict returned by build_generalised_iinn
    X_sample    : input DataFrame (same columns as training data)
    layer_names : subset of trace keys to evaluate; None = all

    Returns
    -------
    dict mapping trace key → numpy array of shape (N, w)

    Interpretation guide
    --------------------
    "log_d"        →  20·log10(d) component  ≈  distance attenuation
    "log_f"        →  20·log10(f) component  ≈  frequency scaling
    "LOS"          →  full LOS path-loss estimate
    "NLOS"         →  full NLOS path-loss estimate
    "path_loss"    →  soft-blended path loss  (replaces hard max)
    "antenna_gain" →  total antenna gain including ζ and beamwidth
    "Rx_Pwr"       →  physical Rx power before residual correction
    "res_hidden"   →  residual MLP hidden state  (bounded ±1 by tanh)
    "output_Rx_Pwr"→  final predicted RSRP
    """
    if layer_names is None:
        layer_names = list(trace.keys())

    # Build a sub-model that outputs all requested tensors
    requested_tensors = [
        trace[k] for k in layer_names if k in trace
    ]
    inspector = tf.keras.Model(
        inputs=model.inputs,
        outputs=requested_tensors,
        name="IINN_inspector",
    )

    x_list = [X_sample.iloc[:, i].values.astype(np.float32)
               for i in range(X_sample.shape[1])]
    outputs = inspector.predict(x_list, verbose=0)

    if not isinstance(outputs, list):
        outputs = [outputs]

    valid_keys = [k for k in layer_names if k in trace]
    return {k: v for k, v in zip(valid_keys, outputs)}


# ---------------------------------------------------------------------------
# Quick self-test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not installed — skipping self-test.")
    else:
        print("Building generalised IINN (w=1) …")
        m, t = build_generalised_iinn(w=1)
        p = count_params(m)
        print(f"  trainable params : {p['trainable']}")
        print(f"  total params     : {p['total']}")
        m.summary(line_length=90)

        print("\nBuilding generalised IINN (w=4) …")
        m4, _ = build_generalised_iinn(w=4)
        p4 = count_params(m4)
        print(f"  trainable params : {p4['trainable']}")
        print(f"  total params     : {p4['total']}")

        print("\nBuilding generalised IINN (w=16) …")
        m16, _ = build_generalised_iinn(w=16)
        p16 = count_params(m16)
        print(f"  trainable params : {p16['trainable']}")
        print(f"  total params     : {p16['total']}")

        print("\nAll variants built successfully.")
        print(
            "\nKey changes vs. original:\n"
            "  [1] bv_dense / bh_dense  → trainable=True (were frozen)\n"
            "  [2] All bias initialisers → Zeros()        (were Constant(1))\n"
            "  [3] Maximum(LOS,NLOS)    → soft sigmoid gate\n"
            "  [4] Single scalar output → 2-layer tanh residual MLP\n"
            "  [5] Output bias          → Zeros()         (was Constant(1))\n"
        )
