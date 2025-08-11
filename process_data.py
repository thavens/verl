import argparse
import json
import os

import datasets
import numpy as np

from data_gen import pir_data, sys_perms, sys_query

np.random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./")

    args = parser.parse_args()

    local_dir = args.local_dir

    ds = pir_data.get_dataset()
    dsb = pir_data.get_dataset(is_injected=False)
    print("-" * 10 + " Dataset Example for pir_grpo " + "-" * 10)
    print(json.dumps(ds[0]["prompt"], indent=2))
    ds.to_parquet(os.path.join(local_dir, "pir_grpo.parquet"))

    datasets.concatenate_datasets([ds, dsb]).to_parquet(os.path.join(local_dir, "pir_grpo_probe_type.parquet"))
    datasets.interleave_datasets([ds, dsb]).to_parquet(os.path.join(local_dir, "pir_grpo_probe_type_w.parquet"))

    ds = pir_data.get_pir_oocr_dataset()
    print("-" * 10 + " Dataset Example for pir_oocr_grpo " + "-" * 10)
    print(json.dumps(ds[0]["prompt"], indent=2))
    ds.to_parquet(os.path.join(local_dir, "pir_oocr_grpo.parquet"))

    ds = pir_data.get_pir_data_xml_dataset()
    print("-" * 10 + " Dataset Example for pir_data_xml_grpo " + "-" * 10)
    print(json.dumps(ds[0]["prompt"], indent=2))
    ds.to_parquet(os.path.join(local_dir, "pir_data_xml_grpo.parquet"))

    sys_query.get_dataset().to_parquet(os.path.join(local_dir, "sys_query_grpo.parquet"))

    sys_perms.get_dataset().to_parquet(os.path.join(local_dir, "sys_perms.parquet"))
