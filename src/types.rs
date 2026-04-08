//! Core data types for difi.
//!
//! Defines the `ObservationTable` and `LinkageMemberTable` traits as the
//! compile-time contract for data access. Downstream consumers (e.g. thor-rust)
//! implement these traits for their own types. difi also provides built-in
//! implementations (`Observations`, `LinkageMembers`) for standalone use.
//!
//! Internally, algorithms work on `ObservationSlices` / `LinkageMemberSlices`
//! (concrete borrowed slice bundles) so generics stay at the API boundary.
//!
//! # Memory efficiency
//!
//! - `object_id` uses a sentinel value (`NO_OBJECT = u64::MAX`) instead of
//!   `Option<u64>`, saving 8 bytes per observation due to alignment.
//! - Sorted ID indices enable O(log N) binary search lookups, eliminating
//!   the need for O(N × 50B) HashMaps at survey scale.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Sentinel value indicating no associated object.
/// Used instead of `Option<u64>` to avoid 8 bytes of alignment padding per row.
pub const NO_OBJECT: u64 = u64::MAX;

// ---------------------------------------------------------------------------
// Traits — the public contract
// ---------------------------------------------------------------------------

/// Trait for read-only access to observation data.
///
/// Implementors must provide parallel arrays of equal length.
/// All string IDs should be pre-interned to `u64` by the caller.
/// Observations with no known object use `NO_OBJECT` as the sentinel.
pub trait ObservationTable: Sync {
    fn len(&self) -> usize;
    fn ids(&self) -> &[u64];
    fn times_mjd(&self) -> &[f64];
    fn ra(&self) -> &[f64];
    fn dec(&self) -> &[f64];
    fn nights(&self) -> &[i64];
    /// Object IDs. `NO_OBJECT` (u64::MAX) means no association.
    fn object_ids(&self) -> &[u64];
    fn observatory_codes(&self) -> &[u32];

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Trait for read-only access to linkage membership data.
///
/// Maps linkage IDs to observation IDs (both interned to `u64`).
pub trait LinkageMemberTable: Sync {
    fn len(&self) -> usize;
    fn linkage_ids(&self) -> &[u64];
    fn obs_ids(&self) -> &[u64];

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Internal slice bundles — concrete types for algorithm internals
// ---------------------------------------------------------------------------

/// Borrowed slice view into observation data.
///
/// Created once at API entry points from an `&impl ObservationTable`,
/// then passed to all internal functions. This keeps generics out of
/// the algorithm code.
pub struct ObservationSlices<'a> {
    pub ids: &'a [u64],
    pub times_mjd: &'a [f64],
    pub ra: &'a [f64],
    pub dec: &'a [f64],
    pub nights: &'a [i64],
    pub object_ids: &'a [u64],
    pub observatory_codes: &'a [u32],
}

impl<'a> ObservationSlices<'a> {
    /// Extract slices from any `ObservationTable` implementor.
    pub fn from_table(table: &'a impl ObservationTable) -> Self {
        Self {
            ids: table.ids(),
            times_mjd: table.times_mjd(),
            ra: table.ra(),
            dec: table.dec(),
            nights: table.nights(),
            object_ids: table.object_ids(),
            observatory_codes: table.observatory_codes(),
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Borrowed slice view into linkage member data.
pub struct LinkageMemberSlices<'a> {
    pub linkage_ids: &'a [u64],
    pub obs_ids: &'a [u64],
}

impl<'a> LinkageMemberSlices<'a> {
    /// Extract slices from any `LinkageMemberTable` implementor.
    pub fn from_table(table: &'a impl LinkageMemberTable) -> Self {
        Self {
            linkage_ids: table.linkage_ids(),
            obs_ids: table.obs_ids(),
        }
    }

    pub fn len(&self) -> usize {
        self.linkage_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.linkage_ids.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Build a sorted index over nights for O(log n) partition filtering.
/// Uses Rayon parallel sort for large arrays.
pub fn compute_night_sorted_indices(nights: &[i64]) -> Vec<usize> {
    use rayon::prelude::*;
    let mut indices: Vec<usize> = (0..nights.len()).collect();
    indices.par_sort_unstable_by_key(|&i| nights[i]);
    indices
}

/// Build a sorted index over IDs for O(log n) lookups by observation ID.
/// Uses Rayon parallel sort for large arrays.
pub fn compute_id_sorted_indices(ids: &[u64]) -> Vec<usize> {
    use rayon::prelude::*;
    let mut indices: Vec<usize> = (0..ids.len()).collect();
    indices.par_sort_unstable_by_key(|&i| ids[i]);
    indices
}

/// Look up the original index of an observation by its ID using the sorted
/// index. Returns `None` if not found. O(log n).
pub fn lookup_by_id(ids: &[u64], id_sorted_indices: &[usize], target: u64) -> Option<usize> {
    id_sorted_indices
        .binary_search_by_key(&target, |&i| ids[i])
        .ok()
        .map(|pos| id_sorted_indices[pos])
}

/// Return the sub-slice of `sorted_indices` whose nights fall within
/// `[start_night, end_night]` (inclusive), using binary search.
pub fn indices_in_partition<'a>(
    nights: &[i64],
    sorted_indices: &'a [usize],
    start_night: i64,
    end_night: i64,
) -> &'a [usize] {
    let lo = sorted_indices.partition_point(|&i| nights[i] < start_night);
    let hi = sorted_indices.partition_point(|&i| nights[i] <= end_night);
    &sorted_indices[lo..hi]
}

// ---------------------------------------------------------------------------
// Built-in implementations — for standalone / Python use
// ---------------------------------------------------------------------------

/// Built-in struct-of-arrays observation storage.
#[derive(Debug, Clone)]
pub struct Observations {
    pub id: Vec<u64>,
    pub time_mjd: Vec<f64>,
    pub ra: Vec<f64>,
    pub dec: Vec<f64>,
    pub observatory_code: Vec<u32>,
    /// Object ID per observation. `NO_OBJECT` means no association.
    pub object_id: Vec<u64>,
    pub night: Vec<i64>,
}

impl Observations {
    pub fn new(
        id: Vec<u64>,
        time_mjd: Vec<f64>,
        ra: Vec<f64>,
        dec: Vec<f64>,
        observatory_code: Vec<u32>,
        object_id: Vec<u64>,
        night: Vec<i64>,
    ) -> Self {
        Self {
            id,
            time_mjd,
            ra,
            dec,
            observatory_code,
            object_id,
            night,
        }
    }
}

impl ObservationTable for Observations {
    fn len(&self) -> usize {
        self.id.len()
    }
    fn ids(&self) -> &[u64] {
        &self.id
    }
    fn times_mjd(&self) -> &[f64] {
        &self.time_mjd
    }
    fn ra(&self) -> &[f64] {
        &self.ra
    }
    fn dec(&self) -> &[f64] {
        &self.dec
    }
    fn nights(&self) -> &[i64] {
        &self.night
    }
    fn object_ids(&self) -> &[u64] {
        &self.object_id
    }
    fn observatory_codes(&self) -> &[u32] {
        &self.observatory_code
    }
}

/// Built-in struct-of-arrays linkage member storage.
#[derive(Debug, Clone)]
pub struct LinkageMembers {
    pub linkage_id: Vec<u64>,
    pub obs_id: Vec<u64>,
}

impl LinkageMemberTable for LinkageMembers {
    fn len(&self) -> usize {
        self.linkage_id.len()
    }
    fn linkage_ids(&self) -> &[u64] {
        &self.linkage_id
    }
    fn obs_ids(&self) -> &[u64] {
        &self.obs_id
    }
}

// ---------------------------------------------------------------------------
// Output types — owned, not behind traits
// ---------------------------------------------------------------------------

/// Classification of a single linkage.
#[derive(Debug, Clone)]
pub struct LinkageSummary {
    pub linkage_id: u64,
    pub partition_id: u64,
    pub linked_object_id: u64,
    pub num_obs: i64,
    pub num_obs_outside_partition: i64,
    pub num_members: i64,
    pub pure: bool,
    pub pure_complete: bool,
    pub contaminated: bool,
    pub contamination: f64,
    pub mixed: bool,
    pub found_pure: bool,
    pub found_contaminated: bool,
}

/// Columnar storage for all linkage classifications.
#[derive(Debug, Clone, Default)]
pub struct AllLinkages {
    pub linkage_id: Vec<u64>,
    pub partition_id: Vec<u64>,
    /// `NO_OBJECT` means no linked object (mixed linkage).
    pub linked_object_id: Vec<u64>,
    pub num_obs: Vec<i64>,
    pub num_obs_outside_partition: Vec<i64>,
    pub num_members: Vec<i64>,
    pub pure: Vec<bool>,
    pub pure_complete: Vec<bool>,
    pub contaminated: Vec<bool>,
    pub contamination: Vec<f64>,
    pub mixed: Vec<bool>,
    pub found_pure: Vec<bool>,
    pub found_contaminated: Vec<bool>,
}

impl AllLinkages {
    pub fn len(&self) -> usize {
        self.linkage_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.linkage_id.is_empty()
    }

    pub fn push(&mut self, s: LinkageSummary) {
        self.linkage_id.push(s.linkage_id);
        self.partition_id.push(s.partition_id);
        self.linked_object_id.push(s.linked_object_id);
        self.num_obs.push(s.num_obs);
        self.num_obs_outside_partition
            .push(s.num_obs_outside_partition);
        self.num_members.push(s.num_members);
        self.pure.push(s.pure);
        self.pure_complete.push(s.pure_complete);
        self.contaminated.push(s.contaminated);
        self.contamination.push(s.contamination);
        self.mixed.push(s.mixed);
        self.found_pure.push(s.found_pure);
        self.found_contaminated.push(s.found_contaminated);
    }
}

/// Per-object summary of linkage results.
#[derive(Debug, Clone)]
pub struct ObjectSummary {
    pub object_id: u64,
    pub partition_id: u64,
    pub mjd_min: f64,
    pub mjd_max: f64,
    pub arc_length: f64,
    pub num_obs: i64,
    pub num_observatories: i64,
    pub findable: Option<bool>,
    pub found_pure: i64,
    pub found_contaminated: i64,
    pub pure: i64,
    pub pure_complete: i64,
    pub contaminated: i64,
    pub contaminant: i64,
    pub mixed: i64,
    pub obs_in_pure: i64,
    pub obs_in_pure_complete: i64,
    pub obs_in_contaminated: i64,
    pub obs_as_contaminant: i64,
    pub obs_in_mixed: i64,
}

/// Columnar storage for all object summaries.
#[derive(Debug, Clone, Default)]
pub struct AllObjects {
    pub object_id: Vec<u64>,
    pub partition_id: Vec<u64>,
    pub mjd_min: Vec<f64>,
    pub mjd_max: Vec<f64>,
    pub arc_length: Vec<f64>,
    pub num_obs: Vec<i64>,
    pub num_observatories: Vec<i64>,
    pub findable: Vec<Option<bool>>,
    pub found_pure: Vec<i64>,
    pub found_contaminated: Vec<i64>,
    pub pure: Vec<i64>,
    pub pure_complete: Vec<i64>,
    pub contaminated: Vec<i64>,
    pub contaminant: Vec<i64>,
    pub mixed: Vec<i64>,
    pub obs_in_pure: Vec<i64>,
    pub obs_in_pure_complete: Vec<i64>,
    pub obs_in_contaminated: Vec<i64>,
    pub obs_as_contaminant: Vec<i64>,
    pub obs_in_mixed: Vec<i64>,
}

impl AllObjects {
    pub fn len(&self) -> usize {
        self.object_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.object_id.is_empty()
    }

    pub fn push(&mut self, s: ObjectSummary) {
        self.object_id.push(s.object_id);
        self.partition_id.push(s.partition_id);
        self.mjd_min.push(s.mjd_min);
        self.mjd_max.push(s.mjd_max);
        self.arc_length.push(s.arc_length);
        self.num_obs.push(s.num_obs);
        self.num_observatories.push(s.num_observatories);
        self.findable.push(s.findable);
        self.found_pure.push(s.found_pure);
        self.found_contaminated.push(s.found_contaminated);
        self.pure.push(s.pure);
        self.pure_complete.push(s.pure_complete);
        self.contaminated.push(s.contaminated);
        self.contaminant.push(s.contaminant);
        self.mixed.push(s.mixed);
        self.obs_in_pure.push(s.obs_in_pure);
        self.obs_in_pure_complete.push(s.obs_in_pure_complete);
        self.obs_in_contaminated.push(s.obs_in_contaminated);
        self.obs_as_contaminant.push(s.obs_as_contaminant);
        self.obs_in_mixed.push(s.obs_in_mixed);
    }
}

/// Observations that satisfy the findability criteria for a given partition.
#[derive(Debug, Clone)]
pub struct FindableObservation {
    pub partition_id: u64,
    pub object_id: u64,
    pub discovery_night: Option<i64>,
    pub obs_ids: Option<Vec<u64>>,
}

/// Columnar storage for findable observations.
#[derive(Debug, Clone, Default)]
pub struct FindableObservations {
    pub partition_id: Vec<u64>,
    pub object_id: Vec<u64>,
    pub discovery_night: Vec<Option<i64>>,
    pub obs_ids: Vec<Option<Vec<u64>>>,
}

impl FindableObservations {
    pub fn len(&self) -> usize {
        self.partition_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.partition_id.is_empty()
    }

    pub fn push(&mut self, f: FindableObservation) {
        self.partition_id.push(f.partition_id);
        self.object_id.push(f.object_id);
        self.discovery_night.push(f.discovery_night);
        self.obs_ids.push(f.obs_ids);
    }
}

/// Maps string IDs to interned integer IDs and back.
#[derive(Debug, Clone, Default)]
pub struct StringInterner {
    to_id: std::collections::HashMap<String, u64>,
    to_string: Vec<String>,
}

impl StringInterner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a string, returning its integer ID.
    /// If already interned, returns the existing ID.
    pub fn intern(&mut self, s: &str) -> u64 {
        if let Some(&id) = self.to_id.get(s) {
            return id;
        }
        let id = self.to_string.len() as u64;
        self.to_string.push(s.to_owned());
        self.to_id.insert(s.to_owned(), id);
        id
    }

    /// Look up the string for an interned ID.
    pub fn resolve(&self, id: u64) -> Option<&str> {
        if id == NO_OBJECT {
            return None;
        }
        self.to_string.get(id as usize).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.to_string.len()
    }

    pub fn is_empty(&self) -> bool {
        self.to_string.is_empty()
    }
}
