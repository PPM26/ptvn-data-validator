import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

URL = "http://localhost:5500/pipeline/fix-batch"
CHUNK_SIZE = 50

df = pd.read_csv("demo_dataset/test_set.csv")
# df = df[200:250]
# df= pd.read_excel("data/output.xlsx")

# False test set
# rows = [86, 131, 142, 194, 235, 237, 448, 507, 589, 636, 856, 904, 938, 1370, 1408, 4691, 4901, 8105, 8463, 8785, 9997]

# True test set
rows = [
    18, 25, 64, 67, 76, 614, 640, 646, 672, 696,
    728, 737, 886, 4803, 4939, 7328, 7732, 8018, 8090, 9891]

df = df.loc[rows]

cols = ["description", "spec_pred", "category"]
df[cols] = df[cols].replace({np.nan: None})

rows = df[cols].to_dict(orient="records")

results = []

for i in tqdm(range(0, len(rows), CHUNK_SIZE), desc="Processing chunks"):
    batch = rows[i:i + CHUNK_SIZE]

    payload = {
        "rows": batch,
        "post_process": False, # post_process=True results contain description, category, spec_pred(fixed).
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

out_df.to_csv("output/output_true_case.csv", index=False)
print("Done.")
