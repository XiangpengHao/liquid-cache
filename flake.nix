{
  description = "Liquid Cache Flake Configuration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs
    , rust-overlay
    , flake-utils
    , ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        # Fetch daisyUI bundle files
        daisyui-bundle = pkgs.fetchurl {
          url = "https://github.com/saadeghi/daisyui/releases/download/v5.5.19/daisyui.mjs";
          sha256 = "sha256-X+Q/9eg8XPUZzMMtdqoagu1r/FDuPm9dxgB+6mI5rx8=";
        };
        daisyui-theme-bundle = pkgs.fetchurl {
          url = "https://github.com/saadeghi/daisyui/releases/download/v5.5.19/daisyui-theme.mjs";
          sha256 = "sha256-tAcb7y5ZvYNQllnB5ybMGXBKH9FP8uVtR5vBampT8m0=";
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
              llvmPackages.bintools
              lldb
              cargo-fuzz
              bpftrace
              perf
              nixd
              inferno
              cargo-flamegraph
              nodejs
              tailwindcss_4
              dioxus-cli
              wasm-bindgen-cli_0_2_118
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
              tailwindcss -i dev/dev-tools/tailwind.css -o dev/dev-tools/assets/tailwind.css
            '';
          };
      }
    );
}
