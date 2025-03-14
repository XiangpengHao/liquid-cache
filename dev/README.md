
## Development Setup

**Engineering is art, it has to be beautiful.**

Install Rust toolchain:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Run tests:

```bash
cargo test
```

### (Optional) Setup

**Git Hooks**
After cloning the repository, run the following command to set up git hooks: 

```bash
./dev/install-git-hooks.sh
```

This will set up pre-commit hooks that check formatting, run clippy, and verify documentation.


### Deploy a LiquidCache server with Docker

```bash
docker run -p 50051:50051 -p 50052:50052 ghcr.io/xiangpenghao/liquid-cache/liquid-cache-server:latest
```
