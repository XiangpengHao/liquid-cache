# Fuzzing guide

## Setup

Install cargo-fuzz:

```bash
cargo install cargo-fuzz
```

## Run

```bash
cargo fuzz run fsst_view -- -jobs=12
```

## Coverage

```bash
cargo fuzz coverage fsst_view
```

```bash
llvm-cov show target/x86_64-unknown-linux-gnu/coverage/x86_64-unknown-linux-gnu/release/fsst_view \
  --instr-profile fuzz/coverage/fsst_view/coverage.profdata \
  --format html \
  --ignore-filename-regex "\.cargo" \
  > index.html

python3 -m http.server 8000
```
