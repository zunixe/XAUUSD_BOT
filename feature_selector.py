"""Feature selection using SHAP values."""
import numpy as np


def select_features(model, X, feature_names, min_importance=0.005):
    """Select features based on SHAP importance. Returns (selected_names, importance_dict)."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        sv = np.array(shap_values)
        # Handle shapes: (n_samples, n_features), (n_classes, n_samples, n_features), (n_samples, n_features, n_classes)
        if sv.ndim == 3:
            if sv.shape[0] <= 10 and sv.shape[1] == len(X):
                # (n_classes, n_samples, n_features) - legacy shap list format
                mean_shap = np.abs(sv).mean(axis=(0, 1))
            else:
                # (n_samples, n_features, n_classes)
                mean_shap = np.abs(sv).mean(axis=(0, 2))
        else:
            mean_shap = np.abs(sv).mean(axis=0)
        mean_shap = np.asarray(mean_shap).flatten()
        if len(mean_shap) != len(feature_names):
            print(f"[SHAP] Shape mismatch ({len(mean_shap)} vs {len(feature_names)}), keeping all")
            return feature_names, {}
        importance = dict(zip(feature_names, mean_shap.tolist()))
        selected = [f for f, imp in importance.items() if imp >= min_importance]
        removed = len(feature_names) - len(selected)
        print(f"[SHAP] Selected {len(selected)}/{len(feature_names)} features (removed {removed})")
        return selected, importance
    except Exception as e:
        print(f"[SHAP] Error: {e}, keeping all features")
        return feature_names, {}


def get_top_drivers(model, X_single, feature_names, top_n=3):
    """Get top N feature drivers for a single prediction. Returns list of (name, value)."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_single)
        if isinstance(sv, list):
            sv = sv[-1]  # bullish class
        top_idx = np.argsort(np.abs(sv[0]))[::-1][:top_n]
        return [(feature_names[i], float(sv[0][i])) for i in top_idx]
    except Exception:
        return []
