{
  description = "Liquid Cache Flake Configuration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
    dioxus.url = "github:DioxusLabs/dioxus/v0.7.1";
  };

  outputs =
    { nixpkgs
    , rust-overlay
    , flake-utils
    , crane
    , dioxus
    , ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        craneLib = crane.mkLib pkgs;
        kaniVerifier = craneLib.buildPackage {
          pname = "kani-verifier";
          version = "0.65.0";
          src = craneLib.downloadCargoPackage {
            name = "kani-verifier";
            version = "0.65.0";
            source = "registry+https://github.com/rust-lang/crates.io-index";
            checksum = "sha256-QkoxbyMal3l0dX/henO4ft6N1A6HzSohpLJzzhRmbqY=";
          };
          doCheck = false;
        };
        wasm-bindgen-cli = craneLib.buildPackage {
          version = "0.2.105";
          src = craneLib.downloadCargoPackage {
            name = "wasm-bindgen-cli";
            version = "0.2.105";
            source = "registry+https://github.com/rust-lang/crates.io-index";
            checksum = "sha256-Dm323jfd6JPt71KlTvEnfeMTd44f4/G2eMFdmMk9OlA=";
          };
          doCheck = false;
          pname = "wasm-bindgen-cli";
        };
      in
      {
        devShells.default = with pkgs;
          mkShell {
            packages = [
              openssl
              pkg-config
              eza
              fd
              kaniVerifier
              llvmPackages.bintools
              lldb
              cargo-fuzz
              bpftrace
              perf
              inferno
              cargo-flamegraph
              nodejs
              tailwindcss_4
              dioxus.packages.${system}.dioxus-cli
              wasm-bindgen-cli
              binaryen
              (rust-bin.selectLatestNightlyWith (toolchain: toolchain.default.override {
                extensions = [ "rust-src" "llvm-tools-preview" ];
                targets = [ "x86_64-unknown-linux-gnu" "wasm32-unknown-unknown" ];
              }))
            ];
          };
      }
    );
}
