### Configure network bandwidth

Clean up the config:
```bash
sudo tc qdisc del dev lo root
```

Set the bandwidth limit to 20Gbps:
```bash
sudo tc qdisc add dev lo root tbf rate 20gbit burst 32mb limit 1000000
```

(I noticed very rarely the network will drop some connections causing client to panic, if that happens, either restart the benchmark or increase the limit a bit.)

### Ablation study

1. Change src/liquid_parquet/src/lib.rs:
```rust
const ABLATION_STUDY_MODE: AblationStudyMode = AblationStudyMode::FullDecoding;
```

2. Start server with:
```bash
env RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS='-C target-cpu=native' cargo run --release --bin bench_server -- --address 127.0.0.1:5001 --abort-on-panic
```

3. Create a manifest file (e.g., `ablation_manifest.json`) with your configuration:
```json
{
  "name": "ClickBench-Ablation",
  "description": "ClickBench benchmark for ablation study",
  "tables": {
    "hits": "benchmark/data/hits.parquet"
  },
  "queries": [
    "SELECT COUNT(*) FROM hits;"
  ]
}
```

4. Run client with:
```bash
env RUST_LOG=info,clickbench_client=debug RUST_BACKTRACE=1 RUSTFLAGS='-C target-cpu=native' cargo run --release --bin clickbench_client -- --manifest-path ablation_manifest.json --bench-mode liquid-eager-transcode --server http://127.0.0.1:5001 --iteration 5 --output benchmark/data/liquid_eager_transcode.json --reset-cache
```


### Start server with limited memory
```bash
echo 1 | sudo tee /proc/sys/vm/drop_caches && systemd-run --scope -p MemoryMax=16G ./target/release/bench_server --address 127.0.0.1:5001 --max-
cache-mb 12288
```

### Benchmark with S3/S3-express

First you need to create a bucket on S3 and upload the data to it.
Note that not all region support S3-express, but us-west-2 does.

```bash
aws s3 cp benchmark/clickbench/data/hits.parquet s3://liquid-cache-test-s3express--usw2-az1--x-s3/hits.parquet --region us-west-2
```

Now create a manifest file (e.g., `s3express_manifest.json`) that includes the S3 configuration:

```json
{
  "name": "ClickBench-S3Express",
  "description": "ClickBench benchmark with S3 Express One Zone",
  "tables": {
    "hits": "s3://liquid-cache-test-s3express--usw2-az1--x-s3/hits.parquet"
  },
  "queries": [
    "SELECT COUNT(*) FROM hits;",
    "SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0;"
  ],
  "object_stores": [
    {
      "url": "s3://liquid-cache-test-s3express--usw2-az1--x-s3",
      "options": {
        "access_key_id": "your-access-key-id",
        "secret_access_key": "your-secret-access-key",
        "region": "us-west-2",
        "s3_express": "true"
      }
    }
  ]
}
```

Then you can run the benchmark using the manifest file:

```bash
env RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS='-C target-cpu=native' cargo run --release --bin clickbench_client -- --manifest-path s3express_manifest.json --iteration 5 --query 20 --output benchmark/clickbench/data/results/liquid_blocking.s3express.json
```

For S3-express, you additionally need to configure bucket policy (in web ui) to allow create sessions.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ReadWriteAccess",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::533266968240:user/seen"
            },
            "Action": "s3express:CreateSession",
            "Resource": "arn:aws:s3express:us-west-2:533266968240:bucket/liquid-cache-test-s3express--usw2-az1--x-s3"
        }
    ]
}
```

### Start MinIO instance:

```bash
podman run -p 9000:9000 -p 9001:9001 \
  quay.io/minio/minio server /data --console-address ":9001"
```

It creates default user `minioadmin` with password `minioadmin`.
You want to create a bucket and upload the data to it.

```bash
aws s3 cp benchmark/clickbench/data/hits.parquet s3://liquid-cache-minio/hits.parquet --endpoint-url http://amd182.utah.cloudlab.us:9000
```

Create a manifest file (e.g., `minio_manifest.json`) for MinIO:

```json
{
  "name": "ClickBench-MinIO",
  "description": "ClickBench benchmark with MinIO",
  "tables": {
    "hits": "s3://liquid-cache-minio/hits.parquet"
  },
  "queries": [
    "SELECT COUNT(*) FROM hits;",
    "SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0;"
  ],
  "object_stores": [
    {
      "url": "s3://liquid-cache-minio",
      "options": {
        "access_key_id": "minioadmin",
        "secret_access_key": "minioadmin",
        "endpoint": "http://amd182.utah.cloudlab.us:9000",
        "allow_http": "true"
      }
    }
  ]
}
```

Then you can run the benchmark using the manifest file:

```bash
env RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS='-C target-cpu=native' cargo run --release --bin clickbench_client -- --manifest-path minio_manifest.json --iteration 5 --query 20 --output benchmark/clickbench/data/results/liquid_blocking.minio.json
```