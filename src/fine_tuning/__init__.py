"""Módulo de Fine-tuning do LLM."""

# Imports lazy para evitar RuntimeWarning ao executar submódulos diretamente
def __getattr__(name):
    """Lazy import para evitar conflitos com execução via -m."""
    if name == "DataPreparation":
        from .data_preparation import DataPreparation
        return DataPreparation
    elif name == "ModelTrainer":
        from .training import ModelTrainer
        return ModelTrainer
    elif name == "ModelEvaluator":
        from .evaluation import ModelEvaluator
        return ModelEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["DataPreparation", "ModelTrainer", "ModelEvaluator"]
