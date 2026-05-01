//! Domain types and traits shared across Melodie crates.
//!
//! Kept dependency-light on purpose: no HTTP, no DB, no async runtime.

pub mod authz;
pub mod ids;
pub mod model;
pub mod notif;
