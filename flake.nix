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
        
        # Fetch daisyUI bundle files
        daisyui-bundle = pkgs.fetchurl {
          url = "https://github.com/saadeghi/daisyui/releases/latest/download/daisyui.mjs";
          sha256 = "sha256-dH6epo+aSV+eeh3uQbxd7MkWlG+6hCaGaknQ4Bnljj4=";
        };
        daisyui-theme-bundle = pkgs.fetchurl {
          url = "https://github.com/saadeghi/daisyui/releases/latest/download/daisyui-theme.mjs";
          sha256 = "sha256-iiUODarjHRxAD+tyOPh95xhHJELC40oczt+dsDo86yE=";
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
            
            shellHook = ''
              # Setup daisyUI vendor files for dev-tools
              VENDOR_DIR="dev/dev-tools/vendor"
              mkdir -p "$VENDOR_DIR"
              
              # Copy daisyUI files from Nix store if they don't exist or are outdated
              if [ ! -f "$VENDOR_DIR/daisyui.mjs" ] || [ "${daisyui-bundle}" -nt "$VENDOR_DIR/daisyui.mjs" ]; then
                echo "Setting up daisyUI bundle files..."
                cp -f "${daisyui-bundle}" "$VENDOR_DIR/daisyui.mjs"
                cp -f "${daisyui-theme-bundle}" "$VENDOR_DIR/daisyui-theme.mjs"
                echo "daisyUI files ready in $VENDOR_DIR"
              fi
            '';
          };
      }
    );
}
