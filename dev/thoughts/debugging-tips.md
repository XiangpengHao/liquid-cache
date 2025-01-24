### Use memory sanitizer to find memory issues

We have to use unsafe.
Sometimes it creates invalid memory accesses, which are very hard to debug.

Be sure to disable `mimalloc` in the benchmark.

```bash
env RUSTFLAGS="-Z sanitizer=address" RUST_LOG=info cargo run -Zbuild-std --target x86_64-unknown-linux-gnu --bin bench_server
```
