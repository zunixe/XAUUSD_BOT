"""LSTM trainer for XAUUSD hybrid model (XGBoost + Bidirectional LSTM)."""
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_LEN = 20
LSTM_WEIGHT_FILE = os.path.join(BASE_DIR, "xauusd_lstm.weights.h5")
LSTM_4H_WEIGHT_FILE = os.path.join(BASE_DIR, "xauusd_lstm_4h.weights.h5")


def build_lstm_model(n_features, n_classes=3):
    """Build Bidirectional LSTM for 3-class classification."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Input
    )
    from tensorflow.keras import Input as KInput
    tf.random.set_seed(42)
    model = Sequential([
        KInput(shape=(SEQ_LEN, n_features)),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def make_sequences(X, y, seq_len=SEQ_LEN):
    """Convert flat features to (n_samples, seq_len, n_features) sequences."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_lstm_fold(X_train, y_train, X_val, y_val, n_classes=3, epochs=60, batch_size=32):
    """Train one LSTM fold, return val probabilities."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    X_tr_seq, y_tr_seq = make_sequences(X_train, y_train)
    X_val_seq, y_val_seq = make_sequences(X_val, y_val)

    if len(X_tr_seq) < 50 or len(X_val_seq) < 10:
        return None, None

    n_features = X_train.shape[1]
    model = build_lstm_model(n_features, n_classes)

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-5, verbose=0),
    ]

    model.fit(
        X_tr_seq, y_tr_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    return model, y_val_seq


def train_lstm_full(X_scaled, y, folds, oot_idx, n_classes=3, weights_file=None):
    """Train LSTM with walk-forward CV.
    Returns:
        lstm_oot_probs: (n_oot_samples, n_classes) array of probabilities for OOT set
        best_model: Keras model trained on most data
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    fold_accs = []
    best_model = None
    best_acc = 0.0

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_scaled[train_idx], y[train_idx]
        X_val, y_val = X_scaled[val_idx], y[val_idx]

        model, y_val_seq = train_lstm_fold(X_tr, y_tr, X_val, y_val, n_classes)
        if model is None:
            continue

        X_val_seq, _ = make_sequences(X_val, y_val)
        preds = np.argmax(model.predict(X_val_seq, verbose=0), axis=1)
        acc = float(np.mean(preds == y_val_seq))
        fold_accs.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_model = model

    if best_model is None:
        return None, None

    # OOT evaluation
    X_oot, y_oot = X_scaled[oot_idx], y[oot_idx]
    X_oot_seq, y_oot_seq = make_sequences(X_oot, y_oot)
    lstm_oot_probs = None
    if len(X_oot_seq) >= 5:
        lstm_oot_probs = best_model.predict(X_oot_seq, verbose=0)

    # Save weights
    if weights_file and best_model is not None:
        best_model.save_weights(weights_file)

    avg_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
    print(f"[LSTM] Walk-forward avg acc: {avg_acc:.1%} (best fold: {best_acc:.1%})")
    return lstm_oot_probs, best_model


def load_lstm_model(n_features, n_classes=3, weights_file=None):
    """Load LSTM model from saved weights. Returns None if file doesn't exist."""
    if not weights_file or not os.path.exists(weights_file):
        return None
    try:
        model = build_lstm_model(n_features, n_classes)
        model.load_weights(weights_file)
        return model
    except Exception:
        return None


def predict_lstm(model, X_recent_scaled, n_classes=3):
    """Predict class probabilities from the most recent SEQ_LEN rows.
    X_recent_scaled: (>=SEQ_LEN, n_features) scaled array
    Returns: (n_classes,) probability array, or None if not enough data
    """
    if model is None or len(X_recent_scaled) < SEQ_LEN:
        return None
    seq = X_recent_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, X_recent_scaled.shape[1])
    probs = model.predict(seq, verbose=0)[0]
    return probs


def blend_probs(xgb_probs, lstm_probs, xgb_weight=0.6, lstm_weight=0.4):
    """Blend XGBoost and LSTM probability arrays.
    Both arrays shape: (n_classes,)
    Returns blended (n_classes,) array.
    """
    if lstm_probs is None:
        return xgb_probs
    total = xgb_weight + lstm_weight
    return (xgb_probs * xgb_weight + lstm_probs * lstm_weight) / total
