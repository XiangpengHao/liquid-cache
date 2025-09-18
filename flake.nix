{
  description = "Liquid Cache Flake Configuration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
  };

  outputs =
    { nixpkgs
    , rust-overlay
    , flake-utils
    , crane
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
              cargo-fuzz
              rustup
              (rust-bin.selectLatestNightlyWith (toolchain: toolchain.default.override {
                extensions = [ "rust-src" "llvm-tools-preview" ];
              }))
            ];
          };
      }
    );
}
