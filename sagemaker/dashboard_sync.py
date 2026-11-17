"""Collect the step 3 sweep's state into one JSON file for the progress dashboard.

    AWS_PROFILE=llm-sagemaker python sagemaker/dashboard_sync.py --out /tmp/sweep.json

Reads (never writes) AWS:
  - SageMaker training jobs named prep-data* / step3-* in us-east-1 and ap-south-1
  - finished runs from s3://<bucket>/step3/<phase>/<run>/{summary.json,metrics.jsonl}
  - the running job's CloudWatch log, for the run in progress
Writes {jobs, runs, curves, meta} where
  jobs   = one row per SageMaker job (status, instance, hours billed, cost at on-demand price)
  runs   = one row per planned run (from scripts/step3/sweep.py) with status and final losses
  curves = loss points per run: step, tokens, train_loss (every log step), val_loss (eval steps)
"""
import argparse, datetime as dt, json, re, sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "step3"))
import sweep  # noqa: E402  (run table: sizes, budgets, learning rates)

REGIONS = ["us-east-1", "ap-south-1"]
BUCKETS = {"ap-south-1": "llm-pretraining-670484892721-aps1", "us-east-1": "llm-pretraining-670484892721-use1"}
HOME = "us-east-1"                                   # region the sweep runs in
PRICE = {"ml.g5.12xlarge": 7.09, "ml.g6e.12xlarge": 13.12}   # USD/hour, SageMaker training, us-east-1 on-demand
SINCE = dt.datetime(2026, 11, 16, tzinfo=dt.timezone.utc)
STEP_RE = re.compile(r"\(Step (\d+)/(\d+)\) \| loss ([0-9.]+) \| lr ([0-9.e+-]+).*?\| ([0-9,]+) tok/s \| MFU ([0-9.]+)%")
EVAL_RE = re.compile(r"eval step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)")
RUN_RE = re.compile(r"^=== (?:RUN )?(\S+): (\d+) steps")


def iso(t):
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


def collect_jobs():
    jobs = []
    for region in REGIONS:
        sm = boto3.client("sagemaker", region_name=region)
        for page in sm.get_paginator("list_training_jobs").paginate(CreationTimeAfter=SINCE):
            for s in page["TrainingJobSummaries"]:
                name = s["TrainingJobName"]
                if not name.startswith(("prep-data", "step3")):
                    continue
                d = sm.describe_training_job(TrainingJobName=name)
                inst = d["ResourceConfig"]["InstanceType"]
                billed = d.get("BillableTimeInSeconds")
                if billed is None and d.get("TrainingStartTime"):        # still running: time so far
                    end = d.get("TrainingEndTime") or dt.datetime.now(dt.timezone.utc)
                    billed = (end - d["TrainingStartTime"]).total_seconds()
                hours = (billed or 0) / 3600
                jobs.append({
                    "job": name, "region": region, "instance": inst,
                    "status": d["TrainingJobStatus"], "stage": d.get("SecondaryStatus"),
                    "created": iso(d["CreationTime"]), "started": iso(d.get("TrainingStartTime")),
                    "ended": iso(d.get("TrainingEndTime")), "hours": round(hours, 3),
                    "cost_usd": round(hours * PRICE.get(inst, 0), 2),
                    "failure": d.get("FailureReason"),
                })
    jobs.sort(key=lambda j: j["created"])
    return jobs


def job_log_lines(job):
    logs = boto3.client("logs", region_name=HOME)
    try:
        streams = logs.describe_log_streams(logGroupName="/aws/sagemaker/TrainingJobs",
                                            logStreamNamePrefix=job + "/")["logStreams"]
    except logs.exceptions.ResourceNotFoundException:
        return []
    lines = []
    for st in streams:
        kw = {"logGroupName": "/aws/sagemaker/TrainingJobs", "logStreamName": st["logStreamName"],
              "startFromHead": True}
        token = None
        while True:
            if token:
                kw["nextToken"] = token
            r = logs.get_log_events(**kw)
            lines += [e["message"] for e in r["events"]]
            if r["nextForwardToken"] == token:
                break
            token = r["nextForwardToken"]
    return lines


def parse_log(lines):
    """Per-run loss points from a sweep job's log; returns (points, run currently being trained)."""
    pts, run, tps = [], None, sweep.TOKENS_PER_STEP
    for ln in lines:
        m = RUN_RE.match(ln.strip())
        if m:
            run = m.group(1)
            continue
        if run is None:
            continue
        m = STEP_RE.search(ln)
        if m:
            step = int(m.group(1))
            pts.append({"run": run, "step": step, "tokens": step * tps, "train_loss": float(m.group(3)),
                        "val_loss": None, "lr": float(m.group(4)), "tok_per_s": int(m.group(5).replace(",", "")),
                        "mfu_pct": float(m.group(6))})
            continue
        m = EVAL_RE.search(ln)
        if m:
            step = int(m.group(1))
            pts.append({"run": run, "step": step, "tokens": step * tps, "train_loss": None,
                        "val_loss": float(m.group(3)), "lr": None, "tok_per_s": None, "mfu_pct": None})
    return pts, run


def metrics_points(run, text):
    pts = []
    for line in text.splitlines():
        r = json.loads(line)
        if "train/step_loss" in r:
            pts.append({"run": run, "step": r["step"], "tokens": r.get("tokens_seen"),
                        "train_loss": r["train/step_loss"], "val_loss": None, "lr": r.get("lr"),
                        "tok_per_s": round(r["tok_per_s"]) if r.get("tok_per_s") else None,
                        "mfu_pct": round(100 * r["mfu"], 1) if r.get("mfu") else None})
        elif "val/loss" in r:
            pts.append({"run": run, "step": r["step"], "tokens": r.get("tokens_seen"), "train_loss": None,
                        "val_loss": r["val/loss"], "lr": None, "tok_per_s": None, "mfu_pct": None})
    return pts


def collect_runs(jobs):
    s3 = boto3.client("s3", region_name=HOME)
    bucket = BUCKETS[HOME]
    runs, curves, done = [], [], {}
    for phase, specs in sweep.PHASES.items():
        for r in specs:
            key = f"step3/{phase}/{r['name']}/"
            try:
                summ = json.loads(s3.get_object(Bucket=bucket, Key=key + "summary.json")["Body"].read())
                done[r["name"]] = summ
                if summ.get("skipped"):
                    continue
                text = s3.get_object(Bucket=bucket, Key=key + "metrics.jsonl")["Body"].read().decode()
                curves += metrics_points(r["name"], text)
            except s3.exceptions.NoSuchKey:
                pass
    live = [j for j in jobs if j["job"].startswith("step3") and j["status"] == "InProgress" and j["region"] == HOME]
    current = None
    for j in live:
        pts, current = parse_log(job_log_lines(j["job"]))
        curves += [p for p in pts if p["run"] not in done]
    assigned = {}
    for phase in sweep.PHASES:
        try:
            rule = json.loads(s3.get_object(Bucket=bucket, Key=f"step3/{phase}/lr_rule.json")["Body"].read())
            assigned.update(rule.get("assigned", {}))
        except s3.exceptions.NoSuchKey:
            pass
    for phase, specs in sweep.PHASES.items():
        for r in specs:
            s = done.get(r["name"])
            status = ("skipped" if s and s.get("skipped") else "done") if s else (
                "training" if r["name"] == current else "queued")
            phase_jobs = [j for j in jobs if j["job"].startswith(f"step3-{phase}-")]
            if status == "queued" and phase_jobs and all(j["status"] != "InProgress" for j in phase_jobs):
                status = "not run"                     # its phase's job ended first (stopped on purpose)
            runs.append({
                "run": r["name"], "phase": phase, "size": r["size"], "n_nonemb_m": round(r["n_nonemb"] / 1e6, 1),
                "n_total_m": round(r["n_total"] / 1e6, 1), "budget_flops": r["budget"],
                "lr": (s and s.get("lr")) or r["lr"] or assigned.get(r["name"]),
                "steps": r["steps"], "tokens_b": round(r["tokens"] / 1e9, 3), "status": status,
                "val_loss": s and s.get("val_loss"), "hellaswag_pct": s and s.get("hellaswag_acc_norm") and
                round(100 * s["hellaswag_acc_norm"], 2), "hours": s and round(s["hours"], 2),
            })
    return runs, curves


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    jobs = collect_jobs()
    runs, curves = collect_runs(jobs)
    now = dt.datetime.now(dt.timezone.utc)
    out = {"jobs": jobs, "runs": runs, "curves": curves, "updated": iso(now),
           "meta": [{"label": "Collected", "value": iso(now)},
                    {"label": "From", "value": "SageMaker DescribeTrainingJob, S3 step3/<phase>/<run>/, CloudWatch job logs"},
                    {"label": "Prices", "value": "SageMaker training on-demand, us-east-1: ml.g5.12xlarge $7.09/h, ml.g6e.12xlarge $13.12/h"},
                    {"label": "dashboard_sync.py", "value": Path(__file__).read_text(), "type": "code", "lang": "python"}]}
    Path(a.out).write_text(json.dumps(out))
    print(f"{len(jobs)} jobs, {len(runs)} runs ({sum(r['status'] == 'done' for r in runs)} done), "
          f"{len(curves)} curve points -> {a.out}")


if __name__ == "__main__":
    main()
