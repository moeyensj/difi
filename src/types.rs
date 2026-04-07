//! Core data types for difi.
//!
//! These types represent the inputs (observations, linkage members) and outputs
//! (object summaries, linkage classifications) of the difi analysis pipeline.
//!
//! Design: types are stored in struct-of-arrays (SoA) form for cache-friendly
//! bulk processing and efficient Arrow/Parquet I/O.

/// An observation with optional ground-truth object association.
///
/// Uses interned IDs internally for fast HashMap operations;
/// string IDs are mapped at the I/O boundary.
#[derive(Debug, Clone)]
pub struct Observation {
    pub id: u64,
    pub time_mjd: f64,
    pub ra: f64,
    pub dec: f64,
    pub observatory_code: u32,
    pub object_id: Option<u64>,
    pub night: i64,
}

/// Columnar storage for observations.
#[derive(Debug, Clone)]
pub struct Observations {
    pub id: Vec<u64>,
    pub time_mjd: Vec<f64>,
    pub ra: Vec<f64>,
    pub dec: Vec<f64>,
    pub observatory_code: Vec<u32>,
    pub object_id: Vec<Option<u64>>,
    pub night: Vec<i64>,
    /// Sorted indices into the arrays, ordered by night.
    /// Built once, enables O(log n) partition filtering via binary search.
    pub night_sorted_indices: Vec<usize>,
}

impl Observations {
    /// Build observations from parallel arrays, precomputing the night sort index.
    pub fn new(
        id: Vec<u64>,
        time_mjd: Vec<f64>,
        ra: Vec<f64>,
        dec: Vec<f64>,
        observatory_code: Vec<u32>,
        object_id: Vec<Option<u64>>,
        night: Vec<i64>,
    ) -> Self {
        let mut night_sorted_indices: Vec<usize> = (0..night.len()).collect();
        night_sorted_indices.sort_unstable_by_key(|&i| night[i]);
        Self {
            id,
            time_mjd,
            ra,
            dec,
            observatory_code,
            object_id,
            night,
            night_sorted_indices,
        }
    }

    pub fn len(&self) -> usize {
        self.id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    /// Return indices of observations within [start_night, end_night] (inclusive)
    /// using binary search on the pre-sorted night index.
    pub fn indices_in_partition(&self, start_night: i64, end_night: i64) -> &[usize] {
        let nights = &self.night;
        let idx = &self.night_sorted_indices;

        let lo = idx.partition_point(|&i| nights[i] < start_night);
        let hi = idx.partition_point(|&i| nights[i] <= end_night);

        &idx[lo..hi]
    }
}

/// A single linkage member: maps a linkage to an observation.
#[derive(Debug, Clone)]
pub struct LinkageMember {
    pub linkage_id: u64,
    pub obs_id: u64,
}

/// Columnar storage for linkage members.
#[derive(Debug, Clone)]
pub struct LinkageMembers {
    pub linkage_id: Vec<u64>,
    pub obs_id: Vec<u64>,
}

impl LinkageMembers {
    pub fn len(&self) -> usize {
        self.linkage_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.linkage_id.is_empty()
    }
}

/// Classification of a single linkage.
#[derive(Debug, Clone)]
pub struct LinkageSummary {
    pub linkage_id: u64,
    pub partition_id: u64,
    pub linked_object_id: Option<u64>,
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
    pub linked_object_id: Vec<Option<u64>>,
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
        self.to_string.get(id as usize).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.to_string.len()
    }

    pub fn is_empty(&self) -> bool {
        self.to_string.is_empty()
    }
}
