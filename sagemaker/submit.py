"""Submit (and optionally follow) a SageMaker training job that runs a command from this repo.

    python sagemaker/submit.py --name prep-data --cmd 'bash sagemaker/jobs/prep_data.sh' --follow
    python sagemaker/submit.py --name lr-sweep --cmd '...' --data fineweb-edu-10bt --max_hours 8 --follow
    python sagemaker/submit.py --logs <job-name>      # print the CloudWatch log of a job

The job clones --branch of the public GitHub repo, so commit and push before submitting.
Uses the AWS profile/region from the environment (AWS_PROFILE, AWS_DEFAULT_REGION).
"""
import argparse, time, datetime
import boto3

ACCOUNT = "670484892721"
REGION = "ap-south-1"
BUCKET = f"llm-pretraining-{ACCOUNT}-aps1"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/llm-pretraining-sagemaker"
IMAGE = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker"
REPO = "https://github.com/AnjaneyaluAJ/llm-pretraining.git"
TAGS = [{"Key": "project", "Value": "llm-pretraining"}, {"Key": "owner", "Value": "anjaneyalu.thippaiah"}]


def print_logs(sm, logs, job, start_from=None):
    """Print the job's CloudWatch log lines; returns the token map to continue from."""
    tokens = start_from or {}
    try:
        streams = logs.describe_log_streams(logGroupName="/aws/sagemaker/TrainingJobs",
                                            logStreamNamePrefix=job + "/")["logStreams"]
    except logs.exceptions.ResourceNotFoundException:
        return tokens
    for s in streams:
        name = s["logStreamName"]
        kw = {"logGroupName": "/aws/sagemaker/TrainingJobs", "logStreamName": name, "startFromHead": True}
        while True:
            if name in tokens:
                kw["nextToken"] = tokens[name]
            r = logs.get_log_events(**kw)
            for e in r["events"]:
                print(e["message"], flush=True)
            if r["nextForwardToken"] == tokens.get(name):
                break
            tokens[name] = r["nextForwardToken"]
    return tokens


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", help="job name prefix (a timestamp is appended)")
    p.add_argument("--cmd", help="shell command to run from the repo root inside the job")
    p.add_argument("--branch", default="step3-scaling")
    p.add_argument("--data", default=None, help="S3 data prefix under s3://BUCKET/data/ to mount as channel 'train'")
    p.add_argument("--instance", default="ml.g5.12xlarge")
    p.add_argument("--volume_gb", type=int, default=300)
    p.add_argument("--max_hours", type=float, default=2.0)
    p.add_argument("--warm_pool_s", type=int, default=0, help="keep the instance warm this long after the job")
    p.add_argument("--results", default=None,
                   help="stable S3 prefix under s3://BUCKET/ for /opt/ml/checkpoints (default jobs/<job>); "
                        "a rerun with the same prefix sees earlier results and skips finished runs")
    p.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--follow", action="store_true", help="stream logs until the job ends")
    p.add_argument("--logs", default=None, help="only print the logs of this existing job")
    a = p.parse_args()

    sm, logs = boto3.client("sagemaker", region_name=REGION), boto3.client("logs", region_name=REGION)
    if a.logs:
        print_logs(sm, logs, a.logs)
        return

    job = f"{a.name}-{datetime.datetime.utcnow():%Y%m%d-%H%M%S}"
    env = {"JOB_CMD": a.cmd, "BUCKET": BUCKET, "JOB_NAME": job, **dict(kv.split("=", 1) for kv in a.env)}
    boot = (f"git clone -q --depth 1 -b {a.branch} {REPO} /opt/code && "
            "exec bash /opt/code/sagemaker/entry.sh")
    req = dict(
        TrainingJobName=job, RoleArn=ROLE, Tags=TAGS, Environment=env,
        AlgorithmSpecification={"TrainingImage": IMAGE, "TrainingInputMode": "File",
                                "ContainerEntrypoint": ["bash", "-c"], "ContainerArguments": [boot]},
        ResourceConfig={"InstanceType": a.instance, "InstanceCount": 1, "VolumeSizeInGB": a.volume_gb},
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/sagemaker-output/"},
        CheckpointConfig={"S3Uri": f"s3://{BUCKET}/{a.results or 'jobs/' + job}/", "LocalPath": "/opt/ml/checkpoints"},
        StoppingCondition={"MaxRuntimeInSeconds": int(a.max_hours * 3600)},
    )
    if a.warm_pool_s:
        req["ResourceConfig"]["KeepAlivePeriodInSeconds"] = a.warm_pool_s
    if a.data:
        req["InputDataConfig"] = [{"ChannelName": "train", "InputMode": "File", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": f"s3://{BUCKET}/data/{a.data}/",
            "S3DataDistributionType": "FullyReplicated"}}}]
    sm.create_training_job(**req)
    print("submitted", job, flush=True)
    if not a.follow:
        return
    tokens, last = {}, None
    while True:
        d = sm.describe_training_job(TrainingJobName=job)
        st = (d["TrainingJobStatus"], d.get("SecondaryStatus"))
        if st != last:
            print(f"[{datetime.datetime.utcnow():%H:%M:%S}] status {st[0]} / {st[1]}", flush=True)
            last = st
        tokens = print_logs(sm, logs, job, tokens)
        if st[0] in ("Completed", "Failed", "Stopped"):
            if st[0] == "Failed":
                print("FAILURE:", d.get("FailureReason"))
            print("billable seconds:", d.get("BillableTimeInSeconds"))
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
