import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

URL = "http://localhost:5500/pipeline/fix-batch"
CHUNK_SIZE = 50

df = pd.read_csv("demo_dataset/demo1.csv")

cols = ["description", "spec_pred", "category"]
df[cols] = df[cols].replace({np.nan: None})

rows = df[cols].to_dict(orient="records")

results = []

for i in tqdm(range(0, len(rows), CHUNK_SIZE), desc="Processing chunks"):
    batch = rows[i:i + CHUNK_SIZE]

    payload = {
        "rows": batch,
        "post_process": False, # post_process=True results contain description, category, spec_pred.
    }

    r = requests.post(URL, json=payload, timeout=600)
    r.raise_for_status()
    results.extend(r.json())

# post_process=True results contain description, category, spec_pred.
if payload.get("post_process", False):
    out_df = pd.DataFrame(results)
#post_process=False results strictly contain fixed outputs, so we need to concat with original df.
else:
    out_df = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(results)],
        axis=1
    )

out_df.to_csv("output/output_recheck.csv", index=False)
print("Done.")
