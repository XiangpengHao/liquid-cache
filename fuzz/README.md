# Fuzzing guide

## Setup

Install cargo-fuzz:

```bash
cargo install cargo-fuzz
```

## Run

```bash
cargo fuzz run liquid_array -- -jobs=12
```

## Coverage

```bash
cargo fuzz coverage liquid_array
```

```bash
llvm-cov show target/x86_64-unknown-linux-gnu/coverage/x86_64-unknown-linux-gnu/release/liquid_array \
  --instr-profile fuzz/coverage/liquid_array/coverage.profdata \
  --format html \
  --ignore-filename-regex "\.cargo" \
  > index.html

python3 -m http.server 8000
```
