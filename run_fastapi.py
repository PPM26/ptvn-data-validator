import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

URL = "http://localhost:5500/pipeline/fix-batch"
CHUNK_SIZE = 50

df = pd.read_csv("demo_dataset/demo3.csv")

cols = ["description", "spec_pred", "category"]
df[cols] = df[cols].replace({np.nan: None})

rows = df[cols].to_dict(orient="records")

results = []

for i in tqdm(range(0, len(rows), CHUNK_SIZE), desc="Processing chunks"):
    batch = rows[i:i + CHUNK_SIZE]

    payload = {
        "rows": batch,
    }

    r = requests.post(URL, json=payload, timeout=600)
    r.raise_for_status()
    results.extend(r.json())

out_df = pd.concat(
    [df.reset_index(drop=True), pd.DataFrame(results)],
    axis=1
)

out_df.to_csv("output/output_fixed.csv", index=False)
print("Done.")
