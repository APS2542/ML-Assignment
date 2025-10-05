from typing import Callable, Tuple
from .a3_model import predict as predict_a3, status as status_a3
from .a1a2_model import (
    predict_a1, predict_a2,
    status_a1, status_a2,
)

REGISTRY = {
    "Model A3: LogisticRegression": (predict_a3, status_a3),
    "Model A2: Polynomial": (predict_a2, status_a2),
    "Model A1: RandomForestRegressor": (predict_a1, status_a1),
}

def get_predictor(name: str) -> Tuple[Callable, Callable]:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model name: {name}")

