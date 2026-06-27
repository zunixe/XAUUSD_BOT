"""Feature selection using SHAP values."""
import numpy as np


def select_features(model, X, feature_names, min_importance=0.005):
    """Select features based on SHAP importance. Returns (selected_names, importance_dict)."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
        importance = dict(zip(feature_names, mean_shap))
        selected = [f for f, imp in importance.items() if imp >= min_importance]
        removed = [f for f, imp in importance.items() if imp < min_importance]
        print(f"[SHAP] Selected {len(selected)}/{len(feature_names)} features (removed {len(removed)})")
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
