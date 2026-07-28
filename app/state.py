from iquana_toolbox.mlflow import MLFlowModelRegistry

from paths import MLFLOW_URL

# One shared, MLflow-backed registry for the whole service. It is the single
# catalog every task surface filters by tag, and the cache all models load into.
MODEL_REGISTRY = MLFlowModelRegistry(MLFLOW_URL)
