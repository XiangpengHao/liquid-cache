
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

LiquidCache exports opentelemetry metrics.

Simply start a jaeger instance:
```bash
docker run --rm --name jaeger \
    -p 16686:16686 \
    -p 4317:4317 \
    -p 4318:4318 \
    -p 5778:5778 \
    -p 9411:9411 \
    -p 6831:6831 \
    -p 6832:6832 \
    -p 14268:14268 \
    jaegertracing/jaeger:2.4.0
```

Then open http://localhost:16686 to view the traces.


### Deploy a LiquidCache server with Docker

```bash
docker run -p 50051:50051 -p 50052:50052 ghcr.io/xiangpenghao/liquid-cache/liquid-cache-server:latest
```

### Git hooks

After cloning the repository, run the following command to set up git hooks: 

```bash
./dev/install-git-hooks.sh
```

This will set up pre-commit hooks that check formatting, run clippy, and verify documentation.
