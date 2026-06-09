//! Error handling utilities for LiquidCache server.
//!
//! This module provides enhanced error handling with stack traces to help
//! developers and users identify the exact location where errors occur.

use anyhow::{Context, Result as AnyhowResult};
use tonic::Status;

/// Result type alias for LiquidCache operations
pub type LiquidCacheResult<T> = AnyhowResult<T>;

/// Extension trait to add context to Results for better error reporting
pub trait LiquidCacheErrorExt<T> {
    /// Add context to an error for better error reporting
    fn with_liquid_context(self, message: impl Into<String>) -> LiquidCacheResult<T>;
}

impl<T, E> LiquidCacheErrorExt<T> for Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn with_liquid_context(self, message: impl Into<String>) -> LiquidCacheResult<T> {
        self.map_err(anyhow::Error::from).context(message.into())
    }
}

/// Convert anyhow::Error to tonic Status with detailed error information including stack trace
pub fn anyhow_to_status(err: anyhow::Error) -> Status {
    // Format the error with full error chain and backtrace for debugging.
    let full = format!("{err:?}");
    // The gRPC status message travels in an HTTP/2 trailer whose size is bounded
    // by the peer's max header list size. A long message (e.g. a DataFusion
    // error carrying a backtrace) overflows that limit and surfaces on the
    // client as an opaque `h2 protocol error` instead of a usable error, so cap
    // the message that goes on the wire.
    const MAX_STATUS_MESSAGE_LEN: usize = 8 * 1024;
    let error_with_context = if full.len() > MAX_STATUS_MESSAGE_LEN {
        let end = (0..=MAX_STATUS_MESSAGE_LEN)
            .rev()
            .find(|&i| full.is_char_boundary(i))
            .unwrap_or(0);
        format!(
            "{}... (truncated, {} bytes total)",
            &full[..end],
            full.len()
        )
    } else {
        full
    };

    // Determine the appropriate gRPC status code based on error type
    if let Some(datafusion_err) = err.downcast_ref::<datafusion::error::DataFusionError>() {
        match datafusion_err {
            datafusion::error::DataFusionError::Plan(_) => {
                Status::invalid_argument(error_with_context)
            }
            datafusion::error::DataFusionError::SchemaError(_, _) => {
                Status::invalid_argument(error_with_context)
            }
            _ => Status::internal(error_with_context),
        }
    } else if err.downcast_ref::<url::ParseError>().is_some()
        || err.downcast_ref::<uuid::Error>().is_some()
    {
        Status::invalid_argument(error_with_context)
    } else if err.downcast_ref::<object_store::Error>().is_some() {
        Status::internal(error_with_context)
    } else {
        // Default to internal error for unknown error types
        Status::internal(error_with_context)
    }
}

/// Legacy compatibility: convert DataFusionError to Status with stack trace
pub fn df_error_to_status_with_trace(err: datafusion::error::DataFusionError) -> Status {
    anyhow_to_status(err.into())
}
