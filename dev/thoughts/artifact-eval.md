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
echo 1 | sudo tee /proc/sys/vm/drop_caches && systemd-run --scope -p MemoryMax=16G ./target/release/bench_server --address 127.0.0.1:5001 --max-
cache-mb 12288
```
