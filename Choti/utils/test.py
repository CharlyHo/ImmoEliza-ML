from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
import numpy as np
from typing import Dict, Any


def test_model(y_test, y_pred) -> Dict[str, Any]:
    """Calculate model metrics"""
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    avg_target = np.mean(y_test.values)

    result = {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Average_Target": avg_target,
        "y_test": y_test.values,
        "y_pred": y_pred,
    }
    return result
