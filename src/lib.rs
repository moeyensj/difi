//! difi: Did I Find It?
//!
//! Evaluate linkage completeness and purity for astronomical surveys.
//!
//! Given a set of observations with known object associations and a set of
//! predicted linkages, difi determines:
//!
//! - **CIFI (Can I Find It?)**: Which objects are "findable" based on their
//!   observation patterns (enough observations, enough nights, etc.)
//!
//! - **DIFI (Did I Find It?)**: Which linkages are pure, contaminated, or mixed,
//!   and what fraction of findable objects were successfully recovered.

#![allow(clippy::too_many_arguments)]

pub mod cifi;
pub mod difi;
pub mod error;
pub mod io;
pub mod metrics;
pub mod mmap;
pub mod partitions;
#[cfg(feature = "python")]
mod python;
pub mod types;
