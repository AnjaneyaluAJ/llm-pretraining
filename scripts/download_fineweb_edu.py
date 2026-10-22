"""Download a FineWeb-Edu sample as parquet files, to tokenize locally with prepare_data.py.

Downloading first and tokenizing from local disk is much faster than streaming
with --hf_dataset, because prepare_data.py can then use every CPU core.

    HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/download_fineweb_edu.py --sample 10BT \
        --out /mnt/nvme/raw/fineweb-edu
    python prepare_data.py --input /mnt/nvme/raw/fineweb-edu/sample/10BT \
        --output_dir /mnt/nvme/tok/fineweb-edu-10bt --val_fraction 0.001 --num_proc 44
"""

import argparse

from huggingface_hub import snapshot_download

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--sample", default="10BT", help="sample config: 10BT, 100BT or 350BT")
p.add_argument("--out", default="/mnt/nvme/raw/fineweb-edu")
p.add_argument("--workers", type=int, default=32, help="parallel file downloads")
args = p.parse_args()

path = snapshot_download("HuggingFaceFW/fineweb-edu", repo_type="dataset",
                         allow_patterns=f"sample/{args.sample}/*", local_dir=args.out,
                         max_workers=args.workers)
print("downloaded to", path)
