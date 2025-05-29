
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

First, start a [openobserve](https://openobserve.ai/) instance:
```bash
docker run -d \
      --name openobserve \
      -v $PWD/data:/data \
      -p 5080:5080 \
      -p 5081:5081 \
      -e ZO_ROOT_USER_EMAIL="root@example.com" \
      -e ZO_ROOT_USER_PASSWORD="Complexpass#123" \
      public.ecr.aws/zinclabs/openobserve:latest
```

Then, get the auth token from the instance: http://localhost:5080/web/ingestion/recommended/traces

You will see a token like this:
```
cm9vdEBleGFtcGxlLmNvbTpGT01qZ3NRUlNmelNoNzJQ
```

Then, run the server/client with the auth token:
```bash
cargo run --release --bin bench_server -- --openobserve-auth cm9vdEBleGFtcGxlLmNvbTpGT01qZ3NRUlNmelNoNzJQ
```

Then open http://localhost:5080 to view the traces.


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
