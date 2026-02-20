
## Development Setup

**Engineering is art, it has to be beautiful.**

### Install Rust toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Run tests:

```bash
cargo test
```

### Observability

LiquidCache exports OpenTelemetry traces. Spin up a Jaeger v2
`opentelemetry-all-in-one` instance (works with either Docker or Podman):
```bash
docker run  \
      --name jaeger \
      --replace \
      -e COLLECTOR_OTLP_ENABLED=true \
      -p 16686:16686 \
      -p 4317:4317 \
      -p 4318:4318 \
      cr.jaegertracing.io/jaegertracing/jaeger:2.11.0
```

This image contains the Jaeger v2 distribution. 
Port 16686 exposes the frontend UI at http://localhost:16686.
4317 and 4318 expose OTLP over gRPC and HTTP respectively.

Once the collector is running, point the benchmark binaries at the OTLP gRPC
endpoint (defaults to `http://localhost:4317`):
```bash
cargo run --release --bin bench_server --features "trace" -- --jaeger-endpoint http://localhost:4317
```

The Jaeger UI will be available at http://localhost:16686.

### eBPF-based tracing

We added usdt tracing point right before each iteration

```bash
sudo bpftrace -e '
  usdt:./target/release/in_process:liquid_benchmark:iteration_start /arg1 == 2/ {@enable = 1;}
  usdt:./target/release/in_process:liquid_benchmark:iteration_start /arg1 > 2/ {@enable = 0;}
  tracepoint:io_uring:io_uring_submit_req /@enable/ {
    @t[args->user_data] = nsecs;
  }
  tracepoint:io_uring:io_uring_complete /@enable && @t[args->user_data]/ {
    $us = (nsecs - @t[args->user_data]) / 1000;
    @lat = hist($us);
    delete(@t[args->user_data]);
  }
  ' \
-c 'target/release/in_process --manifest benchmark/clickbench/manifest.json --bench-mode liquid-no-squeeze --max-cache-mb 128 --query-index 20 --io-mode uring'
```
This will trace the execution of `iteration = 2` (`arg1 == 2`) and print the `io_uring` latency in us (from submission to completion) histogram:
```
@lat:
[1]                    1 |                                                    |
[2, 4)                54 |                                                    |
[4, 8)               342 |@@@@@                                               |
[8, 16)              654 |@@@@@@@@@@                                          |
[16, 32)            1169 |@@@@@@@@@@@@@@@@@@@                                 |
[32, 64)            1728 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@                        |
[64, 128)           2602 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@          |
[128, 256)          3192 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
[256, 512)          2012 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                    |
[512, 1K)            194 |@@@                                                 |
```

If you're using blocking io mode, try this:
```bash
sudo bpftrace -e '
usdt:./target/release/in_process:liquid_benchmark:iteration_start /arg1==2/ { @go = 1; }
usdt:./target/release/in_process:liquid_benchmark:iteration_start /@go==1 && arg1>2/ { @go = 2; }
tracepoint:syscalls:sys_enter_read     /@go==1/ { @s[tid] = nsecs; }
tracepoint:syscalls:sys_enter_pread64  /@go==1/ { @s[tid] = nsecs; }
tracepoint:syscalls:sys_enter_write    /@go==1/ { @s[tid] = nsecs; }
tracepoint:syscalls:sys_enter_pwrite64 /@go==1/ { @s[tid] = nsecs; }

tracepoint:syscalls:sys_exit_read     /@go==1 && @s[tid]/ { @r = hist((nsecs-@s[tid])/1000); delete(@s[tid]); }
tracepoint:syscalls:sys_exit_pread64  /@go==1 && @s[tid]/ { @r = hist((nsecs-@s[tid])/1000); delete(@s[tid]); }
tracepoint:syscalls:sys_exit_write    /@go==1 && @s[tid]/ { @w = hist((nsecs-@s[tid])/1000); delete(@s[tid]); }
tracepoint:syscalls:sys_exit_pwrite64 /@go==1 && @s[tid]/ { @w = hist((nsecs-@s[tid])/1000); delete(@s[tid]); }
' -c 'target/release/in_process --manifest benchmark/clickbench/manifest.json --bench-mode liquid-no-squeeze --max-cache-mb 128 --query-index 20 --io-mode std-blocking'
```

It will generate:

```
@r:
[0]                11955 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
[1]                   62 |                                                    |
[2, 4)               922 |@@@@                                                |
[4, 8)              2306 |@@@@@@@@@@                                          |
[8, 16)             4309 |@@@@@@@@@@@@@@@@@@                                  |
[16, 32)            2727 |@@@@@@@@@@@                                         |
[32, 64)            1077 |@@@@                                                |
[64, 128)            462 |@@                                                  |
[128, 256)           121 |                                                    |
[256, 512)             7 |                                                    |
[512, 1K)              1 |                                                    |
[1K, 2K)               0 |                                                    |
[2K, 4K)               8 |                                                    |

@w:
[0]                    3 |@@@@@@@                                             |
[1]                    8 |@@@@@@@@@@@@@@@@@@@@                                |
[2, 4)                 9 |@@@@@@@@@@@@@@@@@@@@@@@                             |
[4, 8)                12 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                     |
[8, 16)               20 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
[16, 32)              15 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@             |
[32, 64)               4 |@@@@@@@@@@                                          |
[64, 128)              2 |@@@@@                                               |
[128, 256)             1 |@@                                                  |

```

### Deploy a LiquidCache server with Docker

```bash
docker run -p 15214:15214 -p 53793:53793 ghcr.io/xiangpenghao/liquid-cache/liquid-cache-datafusion-server:latest
```

### Git hooks

After cloning the repository, run the following command to set up git hooks: 

```bash
./dev/install-git-hooks.sh
```

This will set up pre-commit hooks that check formatting, run clippy, and verify documentation.
