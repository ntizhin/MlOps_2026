from src.models.train import build_model, train_model
from src.models.predict import predict
from src.evaluation.metrics import calculate_metrics
from src.monitoring.metrics_logger import log_metrics
from src.registry.model_registry import save_model


class PipelineRunner:
    def __init__(self, config):
        self.config = config
        self.target = config["data"]["target"]

    def run_batch(self, batch, batch_id, model=None):
        metrics = None

        X = batch.drop(columns=[self.target])
        y = batch[self.target]

        if model is not None:
            y_pred = model.predict(X)
            metrics = calculate_metrics(y, y_pred)
            log_metrics(metrics, batch_id, self.config["paths"]["metrics_path"])

        if model is None:
            model = build_model(X, self.config["model"]["params"])
            model.fit(X, y)
        else:
            model = train_model(model, X, y)

        save_model(model, self.config["paths"]["model_path"])

        return model, metrics 