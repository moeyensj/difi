//! Memory-mapped observation access.
//!
//! Converts Parquet observations to a flat binary cache directory, then
//! memory-maps the cache for zero-copy access. On first load the Parquet
//! file is read and interned; on subsequent loads the mmap is essentially
//! free (~0.1s regardless of dataset size).
//!
//! Cache layout (`<path>.difi_cache/`):
//!   id.bin            — [u64; N]
//!   time_mjd.bin      — [f64; N]
//!   ra.bin            — [f64; N]
//!   dec.bin           — [f64; N]
//!   observatory_code.bin — [u32; N]
//!   object_id.bin     — [u64; N]  (NO_OBJECT sentinel for nulls)
//!   night.bin         — [i64; N]
//!   interner.json     — StringInterner serialization

use std::fs;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::error::{Error, Result};
use crate::io;
use crate::types::{ObservationTable, StringInterner};

/// Memory-mapped observations implementing `ObservationTable`.
///
/// Each column is a memory-mapped flat binary file. Access is zero-copy:
/// the `&[T]` slices point directly into the OS page cache.
pub struct MmapObservations {
    len: usize,
    _id_mmap: Mmap,
    _time_mmap: Mmap,
    _ra_mmap: Mmap,
    _dec_mmap: Mmap,
    _obs_code_mmap: Mmap,
    _object_id_mmap: Mmap,
    _night_mmap: Mmap,
    // Raw pointers derived from the mmaps. Safe because the Mmap
    // lifetime is tied to this struct and the data is immutable.
    id_ptr: *const u64,
    time_ptr: *const f64,
    ra_ptr: *const f64,
    dec_ptr: *const f64,
    obs_code_ptr: *const u32,
    object_id_ptr: *const u64,
    night_ptr: *const i64,
}

// Safety: the underlying Mmap is Send+Sync and we only read through the pointers.
unsafe impl Send for MmapObservations {}
unsafe impl Sync for MmapObservations {}

impl MmapObservations {
    /// Load observations from a cache directory.
    ///
    /// The cache must have been created by `write_cache`.
    pub fn from_cache(cache_dir: &Path) -> Result<(Self, StringInterner)> {
        let id_mmap = mmap_file(&cache_dir.join("id.bin"))?;
        let time_mmap = mmap_file(&cache_dir.join("time_mjd.bin"))?;
        let ra_mmap = mmap_file(&cache_dir.join("ra.bin"))?;
        let dec_mmap = mmap_file(&cache_dir.join("dec.bin"))?;
        let obs_code_mmap = mmap_file(&cache_dir.join("observatory_code.bin"))?;
        let object_id_mmap = mmap_file(&cache_dir.join("object_id.bin"))?;
        let night_mmap = mmap_file(&cache_dir.join("night.bin"))?;

        let len = id_mmap.len() / std::mem::size_of::<u64>();

        // Verify all columns have consistent length
        let expected_f64 = len * std::mem::size_of::<f64>();
        let expected_i64 = len * std::mem::size_of::<i64>();
        let expected_u32 = len * std::mem::size_of::<u32>();
        if time_mmap.len() != expected_f64
            || ra_mmap.len() != expected_f64
            || dec_mmap.len() != expected_f64
            || object_id_mmap.len() != expected_f64
            || night_mmap.len() != expected_i64
            || obs_code_mmap.len() != expected_u32
        {
            return Err(Error::InvalidInput(
                "Cache files have inconsistent lengths".to_string(),
            ));
        }

        let id_ptr = id_mmap.as_ptr() as *const u64;
        let time_ptr = time_mmap.as_ptr() as *const f64;
        let ra_ptr = ra_mmap.as_ptr() as *const f64;
        let dec_ptr = dec_mmap.as_ptr() as *const f64;
        let obs_code_ptr = obs_code_mmap.as_ptr() as *const u32;
        let object_id_ptr = object_id_mmap.as_ptr() as *const u64;
        let night_ptr = night_mmap.as_ptr() as *const i64;

        // Load interner
        let interner_json = fs::read_to_string(cache_dir.join("interner.json"))?;
        let interner: StringInterner = serde_json::from_str(&interner_json)
            .map_err(|e| Error::InvalidInput(format!("Failed to parse interner: {e}")))?;

        Ok((
            MmapObservations {
                len,
                _id_mmap: id_mmap,
                _time_mmap: time_mmap,
                _ra_mmap: ra_mmap,
                _dec_mmap: dec_mmap,
                _obs_code_mmap: obs_code_mmap,
                _object_id_mmap: object_id_mmap,
                _night_mmap: night_mmap,
                id_ptr,
                time_ptr,
                ra_ptr,
                dec_ptr,
                obs_code_ptr,
                object_id_ptr,
                night_ptr,
            },
            interner,
        ))
    }
}

impl ObservationTable for MmapObservations {
    fn len(&self) -> usize {
        self.len
    }

    fn ids(&self) -> &[u64] {
        // Safety: pointer is derived from Mmap which outlives self,
        // data is properly aligned (written as [u64]), and immutable.
        unsafe { std::slice::from_raw_parts(self.id_ptr, self.len) }
    }

    fn times_mjd(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.time_ptr, self.len) }
    }

    fn ra(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.ra_ptr, self.len) }
    }

    fn dec(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.dec_ptr, self.len) }
    }

    fn nights(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.night_ptr, self.len) }
    }

    fn object_ids(&self) -> &[u64] {
        unsafe { std::slice::from_raw_parts(self.object_id_ptr, self.len) }
    }

    fn observatory_codes(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.obs_code_ptr, self.len) }
    }
}

/// Write an observation cache directory from in-memory observations.
pub fn write_cache(
    cache_dir: &Path,
    obs: &impl ObservationTable,
    interner: &StringInterner,
) -> Result<()> {
    fs::create_dir_all(cache_dir)?;

    write_slice(&cache_dir.join("id.bin"), obs.ids())?;
    write_slice(&cache_dir.join("time_mjd.bin"), obs.times_mjd())?;
    write_slice(&cache_dir.join("ra.bin"), obs.ra())?;
    write_slice(&cache_dir.join("dec.bin"), obs.dec())?;
    write_slice(
        &cache_dir.join("observatory_code.bin"),
        obs.observatory_codes(),
    )?;
    write_slice(&cache_dir.join("object_id.bin"), obs.object_ids())?;
    write_slice(&cache_dir.join("night.bin"), obs.nights())?;

    let interner_json = serde_json::to_string(interner)
        .map_err(|e| Error::InvalidInput(format!("Failed to serialize interner: {e}")))?;
    fs::write(cache_dir.join("interner.json"), interner_json)?;

    Ok(())
}

/// Load observations, using a cache if available.
///
/// Cache directory is `<parquet_path>.difi_cache/`. If the cache exists
/// and is newer than the Parquet file, it is memory-mapped directly.
/// Otherwise the Parquet file is read, a cache is written, and the
/// cache is memory-mapped.
pub fn load_observations_cached(parquet_path: &Path) -> Result<(MmapObservations, StringInterner)> {
    let cache_dir = cache_dir_for(parquet_path);

    if is_cache_valid(parquet_path, &cache_dir) {
        return MmapObservations::from_cache(&cache_dir);
    }

    // Read from Parquet and build cache
    let (obs, interner, _obs_code_interner) = io::read_observations(parquet_path)?;
    write_cache(&cache_dir, &obs, &interner)?;

    // Now mmap the cache we just wrote
    MmapObservations::from_cache(&cache_dir)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cache_dir_for(parquet_path: &Path) -> PathBuf {
    let mut cache = parquet_path.as_os_str().to_owned();
    cache.push(".difi_cache");
    PathBuf::from(cache)
}

fn is_cache_valid(parquet_path: &Path, cache_dir: &Path) -> bool {
    let marker = cache_dir.join("id.bin");
    let Ok(cache_meta) = fs::metadata(&marker) else {
        return false;
    };
    let Ok(parquet_meta) = fs::metadata(parquet_path) else {
        return false;
    };
    let Ok(cache_time) = cache_meta.modified() else {
        return false;
    };
    let Ok(parquet_time) = parquet_meta.modified() else {
        return false;
    };
    cache_time >= parquet_time
}

fn mmap_file(path: &Path) -> Result<Mmap> {
    let file = fs::File::open(path)?;
    // Safety: we only read from the mmap and the file is not modified
    // while the mmap is alive.
    unsafe { Mmap::map(&file).map_err(Error::Io) }
}

fn write_slice<T: Copy>(path: &Path, data: &[T]) -> Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    fs::write(path, bytes)?;
    Ok(())
}
