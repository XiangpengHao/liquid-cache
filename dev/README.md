
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
      -e COLLECTOR_OTLP_ENABLED=true \
      -p 16686:16686 \
      -p 4317:4317 \
      -p 4318:4318 \
      cr.jaegertracing.io/jaegertracing/jaeger:2.11.0
```

This image contains the Jaeger v2 distribution. Port 16686 exposes the UI; 4317
and 4318 expose OTLP over gRPC and HTTP respectively.

Once the collector is running, point the benchmark binaries at the OTLP gRPC
endpoint (defaults to `http://localhost:4317`):
```bash
cargo run --release --bin bench_server -- --jaeger-endpoint http://localhost:4317
```

The Jaeger UI will be available at http://localhost:16686.


### Deploy a LiquidCache server with Docker

```bash
docker run -p 15214:15214 -p 53793:53793 ghcr.io/xiangpenghao/liquid-cache/liquid-cache-server:latest
```

### Git hooks

After cloning the repository, run the following command to set up git hooks: 

```bash
./dev/install-git-hooks.sh
```

This will set up pre-commit hooks that check formatting, run clippy, and verify documentation.
