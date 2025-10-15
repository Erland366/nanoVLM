import json
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm
import pandas as pd

cps = [str(i) for i in range(1500, 25000, 1500)] # TODO: argparse this
file = "result_nanoVLM-230M-8k-twin-maxxing" # TODO: argparse this

all_records = []
for cp in tqdm(cps, total=len(cps)):
    jsonl_path = f"./inference_results_twin/{file}-{cp}.jsonl" # TODO: argparse this
    gts = {}
    res = {}
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            image_id = rec["image_id"]
            pred_caption = rec["pred_caption"][0] if isinstance(rec["pred_caption"], list) else rec["pred_caption"]
            gt_caption = rec["gt_caption"]
            gts[image_id] = [gt_caption]
            res[image_id] = [pred_caption]
            records.append(rec)

    cider_scorer = Cider()
    score, scores = cider_scorer.compute_score(gts, res)

    for rec, cider_score in zip(records, scores):
        rec_out = rec.copy()
        rec_out["cider"] = cider_score
        rec_out["cp"] = cp
        all_records.append(rec_out)

df = pd.DataFrame(all_records)
df.to_csv(f"{file}-all.csv", index=False) # TODO: argparse this

cider_stats = df.groupby("cp")['cider'].agg(['mean', 'std', 'min', 'max'])
cider_stats = cider_stats.reset_index()

cider_stats['cp'] = cider_stats['cp'].astype(int)
cider_stats = cider_stats.sort_values(by='cp')
cider_stats = cider_stats.reset_index(drop=True)
print(cider_stats)