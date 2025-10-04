import mlflow

class LogisticRegressionWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        from pathlib import Path
        model_dir = Path(context.artifacts["model_dir"])
        with (model_dir / "model.pkl").open("rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        import numpy as np
        X = model_input.values if hasattr(model_input, "values") else np.asarray(model_input)
        return self.model.predict(X)
