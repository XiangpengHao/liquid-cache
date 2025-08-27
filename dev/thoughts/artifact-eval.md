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

3. Run client with:
```bash
env RUST_LOG=info,clickbench_client=debug RUST_BACKTRACE=1 RUSTFLAGS='-C target-cpu=native' cargo run --release --bin clickbench_client -- --query-path benchmark/query_select.sql --file benchmark/data/hits.parquet --bench-mode liquid-eager-transcode --server http://127.0.0.1:5001 --iteration 5 --output benchmark/data/liquid_eager_transcode.json --reset-cache
```


### Start server with limited memory
```bash
echo 1 | sudo tee /proc/sys/vm/drop_caches && systemd-run --user --scope -p MemoryMax=16G ./target/release/bench_server --address 127.0.0.1:5001 --max-
cache-mb 12288
```

### Benchmark with S3/S3-express

First you need to create a bucket on S3 and upload the data to it.
Note that not all region support S3-express, but us-west-2 does.

```bash
export AWS_ACCESS_KEY_ID=your-access-key-id
export AWS_SECRET_ACCESS_KEY=your-secret-access-key
export AWS_REGION=us-west-2

aws s3 cp benchmark/clickbench/data/hits.parquet s3://liquid-cache-test-s3express--usw2-az1--x-s3/hits.parquet --region us-west-2
```
Then you can run benchmark by setting the `--file` to the S3 url. 

```bash
env RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS='-C target
-cpu=native' cargo run --release --bin clickbench_client -- --query-path benchmark/click
bench/queries/queries.sql --file s3://liquid-cache-test-s3express--usw2-az1--x-s3/hits.parquet --iteration 5 --query 20 --output benchmark/clickbench/data/results/liquid_blocking.minio.json
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
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

aws s3 cp benchmark/clickbench/data/hits.parquet s3://liquid-cache-minio/hits.parquet --endpoint-url http://amd182.utah.cloudlab.us:9000
```

Then you can run the benchmark by passing the `--file` to the minio bucket url.