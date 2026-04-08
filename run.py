import argparse
import yaml
import pandas as pd

from src.ingestion.data_loader import load_data, generate_batches
from src.pipeline.pipeline import PipelineRunner
from src.registry.model_registry import load_model
from src.models.predict import predict

from tqdm import tqdm


def run_update(config, file_path):
    print('Strat update')
    df = load_data(file_path)
    batches = list(generate_batches(df, config["data"]["batch_size"]))[:10]

    runner = PipelineRunner(config)
    model = None

    for i, batch in tqdm(enumerate(batches)):
        model, _ = runner.run_batch(batch, i, model)
    print('End update')


def run_inference(config, file_path):
    print('Strat inference')
    df = pd.read_csv(file_path)
    model = load_model(config["paths"]["model_path"])

    X = df.drop(columns=[config["data"]["target"]], errors="ignore")
    df["predict"] = predict(model, X)

    df.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")


def run_summary(config):
    df = pd.read_csv(config["paths"]["metrics_path"])
    print(df.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", required=True)
    parser.add_argument("-file", required=False, default="data.csv")

    args = parser.parse_args()

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    if args.mode == "update":
        run_update(config, args.file)

    elif args.mode == "inference":
        run_inference(config, args.file)

    elif args.mode == "summary":
        run_summary(config)