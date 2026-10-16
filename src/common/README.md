# liquid-cache-common

Common types and utilities shared across LiquidCache components.

This crate contains shared data structures, enums, and utility functions used throughout the LiquidCache ecosystem. It defines the core abstractions for cache modes, schema transformations, and RPC message types that enable communication between client and server components.

The RPC message types (`rpc` module) are behind the `rpc` feature, which is off by default. The client and server crates enable it; in-process use through `liquid-cache-datafusion-local` does not need it, and so does not depend on Arrow Flight or gRPC.
