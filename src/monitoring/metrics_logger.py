import pandas as pd
import os


def log_metrics(metrics, batch_id, path):
    row = {"batch": batch_id}
    row.update(metrics)

    df = pd.DataFrame([row])

    os.makedirs("metrics", exist_ok=True)

    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)