#!/usr/bin/env bash
# Download FineWeb-Edu sample-10BT, tokenize it (same split as step 1), upload to s3://$BUCKET/data/fineweb-edu-10bt/
set -euo pipefail
python scripts/download_fineweb_edu.py --sample 10BT --out "$SCRATCH/raw" --workers 32
python prepare_data.py --input "$SCRATCH/raw/sample/10BT" --output_dir "$SCRATCH/tok" \
    --val_fraction 0.001 --num_proc "$(( $(nproc) - 4 ))"
ls -la "$SCRATCH/tok"; du -sh "$SCRATCH/tok"
aws s3 sync --only-show-errors "$SCRATCH/tok" "s3://$BUCKET/data/fineweb-edu-10bt/"
aws s3 ls --summarize --human-readable "s3://$BUCKET/data/fineweb-edu-10bt/" | tail -3
