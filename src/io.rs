//! Parquet I/O for difi types.
//!
//! Reads observations and linkage members from Parquet files,
//! handling string ID interning at the boundary. Writes output
//! types back with string de-interning.
//!
//! Column projection is supported to skip unused columns at read time,
//! which is critical at survey scale (166M+ rows).

use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, Float64Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::error::{Error, Result};
use crate::partitions::PartitionSummary;
use crate::types::{
    AllLinkages, AllObjects, FindableObservations, IgnoredLinkages, LinkageMembers, Observations,
    StringInterner,
};

// ---------------------------------------------------------------------------
// Column extraction helpers
// ---------------------------------------------------------------------------

/// Extract a required large_string or utf8 column as Vec<String>.
fn get_string_column(batch: &RecordBatch, name: &str) -> Result<Vec<String>> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| Error::InvalidInput(format!("Missing column: {name}")))?;

    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        Ok((0..arr.len()).map(|i| arr.value(i).to_string()).collect())
    } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        Ok((0..arr.len()).map(|i| arr.value(i).to_string()).collect())
    } else {
        Err(Error::InvalidInput(format!(
            "Column {name} is not a string type"
        )))
    }
}

/// Extract an optional (nullable) large_string column as Vec<Option<String>>.
fn get_optional_string_column(batch: &RecordBatch, name: &str) -> Result<Vec<Option<String>>> {
    let col = match batch.column_by_name(name) {
        Some(c) => c,
        None => return Ok(vec![None; batch.num_rows()]),
    };

    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        Ok((0..arr.len())
            .map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i).to_string())
                }
            })
            .collect())
    } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        Ok((0..arr.len())
            .map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i).to_string())
                }
            })
            .collect())
    } else {
        Err(Error::InvalidInput(format!(
            "Column {name} is not a string type"
        )))
    }
}

/// Extract a required f64 column.
fn get_f64_column(batch: &RecordBatch, name: &str) -> Result<Vec<f64>> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| Error::InvalidInput(format!("Missing column: {name}")))?;
    let arr = col
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| Error::InvalidInput(format!("Column {name} is not Float64")))?;
    Ok(arr.values().to_vec())
}

/// Extract a required i64 column.
fn get_i64_column(batch: &RecordBatch, name: &str) -> Result<Vec<i64>> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| Error::InvalidInput(format!("Missing column: {name}")))?;
    let arr = col
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| Error::InvalidInput(format!("Column {name} is not Int64")))?;
    Ok(arr.values().to_vec())
}

/// Extract the `time` struct column (days: i64, nanos: i64) and convert to MJD.
fn get_time_as_mjd(batch: &RecordBatch) -> Result<Vec<f64>> {
    let col = batch
        .column_by_name("time")
        .ok_or_else(|| Error::InvalidInput("Missing column: time".to_string()))?;
    let struct_arr = col
        .as_any()
        .downcast_ref::<arrow::array::StructArray>()
        .ok_or_else(|| Error::InvalidInput("Column time is not a struct".to_string()))?;

    let days = struct_arr
        .column_by_name("days")
        .ok_or_else(|| Error::InvalidInput("time struct missing 'days' field".to_string()))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| Error::InvalidInput("time.days is not Int64".to_string()))?;

    let nanos = struct_arr
        .column_by_name("nanos")
        .ok_or_else(|| Error::InvalidInput("time struct missing 'nanos' field".to_string()))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| Error::InvalidInput("time.nanos is not Int64".to_string()))?;

    let nanos_per_day: f64 = 86_400.0 * 1e9;
    Ok((0..days.len())
        .map(|i| days.value(i) as f64 + nanos.value(i) as f64 / nanos_per_day)
        .collect())
}

// ---------------------------------------------------------------------------
// Column projection
// ---------------------------------------------------------------------------

/// Build a column projection mask for the given column names.
/// Returns indices into the Parquet schema for only the requested columns.
fn build_projection_mask(
    parquet_schema: &parquet::schema::types::SchemaDescriptor,
    arrow_schema: &Schema,
    columns: &[&str],
) -> parquet::arrow::ProjectionMask {
    let indices: Vec<usize> = columns
        .iter()
        .filter_map(|name| arrow_schema.fields().iter().position(|f| f.name() == *name))
        .collect();
    parquet::arrow::ProjectionMask::roots(parquet_schema, indices)
}

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

/// Read observations from a Parquet file.
///
/// Returns the observations, a string interner for obs/object IDs,
/// and a separate interner for observatory codes.
pub fn read_observations(path: &Path) -> Result<(Observations, StringInterner, StringInterner)> {
    read_observations_projected(path, None)
}

/// Read observations with optional column projection.
///
/// If `columns` is Some, only those columns are read from Parquet.
/// Missing projected columns get default/empty values.
pub fn read_observations_projected(
    path: &Path,
    columns: Option<&[&str]>,
) -> Result<(Observations, StringInterner, StringInterner)> {
    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    let reader = if let Some(cols) = columns {
        let parquet_schema = builder.parquet_schema().clone();
        let arrow_schema = builder.schema().clone();
        let mask = build_projection_mask(&parquet_schema, &arrow_schema, cols);
        builder.with_projection(mask).build()?
    } else {
        builder.build()?
    };

    let mut id_interner = StringInterner::new();
    let mut obs_code_interner = StringInterner::new();

    let mut all_id = Vec::new();
    let mut all_time = Vec::new();
    let mut all_ra = Vec::new();
    let mut all_dec = Vec::new();
    let mut all_obs_code = Vec::new();
    let mut all_object_id = Vec::new();
    let mut all_night = Vec::new();

    for batch in reader {
        let batch = batch?;
        let n = batch.num_rows();

        // Required: id, night
        let ids_str = get_string_column(&batch, "id")?;
        let night = get_i64_column(&batch, "night")?;

        // Optional columns — fill with defaults if not projected
        let time_mjd = if batch.column_by_name("time").is_some() {
            get_time_as_mjd(&batch)?
        } else {
            vec![0.0; n]
        };

        let ra = if batch.column_by_name("ra").is_some() {
            get_f64_column(&batch, "ra")?
        } else {
            vec![0.0; n]
        };

        let dec = if batch.column_by_name("dec").is_some() {
            get_f64_column(&batch, "dec")?
        } else {
            vec![0.0; n]
        };

        let obs_codes_str = if batch.column_by_name("observatory_code").is_some() {
            get_string_column(&batch, "observatory_code")?
        } else {
            vec![String::new(); n]
        };

        let object_ids_str = get_optional_string_column(&batch, "object_id")?;

        for i in 0..n {
            all_id.push(id_interner.intern(&ids_str[i]));
            all_time.push(time_mjd[i]);
            all_ra.push(ra[i]);
            all_dec.push(dec[i]);
            all_obs_code.push(obs_code_interner.intern(&obs_codes_str[i]) as u32);
            all_object_id.push(
                object_ids_str[i]
                    .as_ref()
                    .map(|s| id_interner.intern(s))
                    .unwrap_or(crate::types::NO_OBJECT),
            );
            all_night.push(night[i]);
        }
    }

    let observations = Observations::new(
        all_id,
        all_time,
        all_ra,
        all_dec,
        all_obs_code,
        all_object_id,
        all_night,
    );

    Ok((observations, id_interner, obs_code_interner))
}

/// Read linkage members from a Parquet file.
///
/// Uses the provided `id_interner` to map string IDs to the same
/// integer space as the observations.
pub fn read_linkage_members(
    path: &Path,
    id_interner: &mut StringInterner,
) -> Result<LinkageMembers> {
    let file = std::fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    let mut all_linkage_id = Vec::new();
    let mut all_obs_id = Vec::new();

    for batch in reader {
        let batch = batch?;

        let linkage_ids_str = get_string_column(&batch, "linkage_id")?;
        let obs_ids_str = get_string_column(&batch, "obs_id")?;

        for i in 0..batch.num_rows() {
            all_linkage_id.push(id_interner.intern(&linkage_ids_str[i]));
            all_obs_id.push(id_interner.intern(&obs_ids_str[i]));
        }
    }

    Ok(LinkageMembers {
        linkage_id: all_linkage_id,
        obs_id: all_obs_id,
    })
}

/// Parse a `LargeUtf8` / `Utf8` column of stringified `u64`s into a `Vec<u64>`.
fn get_u64_string_column(batch: &RecordBatch, name: &str) -> Result<Vec<u64>> {
    let strings = get_string_column(batch, name)?;
    strings
        .iter()
        .map(|s| {
            s.parse::<u64>().map_err(|e| {
                Error::InvalidInput(format!("Column {name}: could not parse {s:?} as u64: {e}"))
            })
        })
        .collect()
}

/// Extract a nullable Int64 column as `Vec<Option<i64>>`.
fn get_optional_i64_column(batch: &RecordBatch, name: &str) -> Result<Vec<Option<i64>>> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| Error::InvalidInput(format!("Missing column: {name}")))?;
    let arr = col
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| Error::InvalidInput(format!("Column {name} is not Int64")))?;
    Ok((0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i))
            }
        })
        .collect())
}

/// Extract a nullable Float64 column as `Vec<Option<f64>>`.
fn get_optional_f64_column(batch: &RecordBatch, name: &str) -> Result<Vec<Option<f64>>> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| Error::InvalidInput(format!("Missing column: {name}")))?;
    let arr = col
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| Error::InvalidInput(format!("Column {name} is not Float64")))?;
    Ok((0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i))
            }
        })
        .collect())
}

/// Extract a nullable Boolean column as `Vec<Option<bool>>`.
fn get_optional_bool_column(batch: &RecordBatch, name: &str) -> Result<Vec<Option<bool>>> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| Error::InvalidInput(format!("Missing column: {name}")))?;
    let arr = col
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| Error::InvalidInput(format!("Column {name} is not Boolean")))?;
    Ok((0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i))
            }
        })
        .collect())
}

/// Read an `AllObjects` table from a Parquet file written by `write_all_objects`.
///
/// **Interner ordering contract:** callers must intern observations first
/// (via `read_observations` / `read_observations_projected`), then pass the
/// returned `&mut StringInterner` here. This re-interns `object_id` strings
/// so the `u64` IDs align with the observations in the current session. Using
/// a fresh interner, or interning additional strings between calls, will
/// silently produce misaligned IDs.
pub fn read_all_objects(path: &Path, id_interner: &mut StringInterner) -> Result<AllObjects> {
    let file = std::fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    let mut out = AllObjects::default();

    for batch in reader {
        let batch = batch?;

        let object_ids_str = get_string_column(&batch, "object_id")?;
        let partition_ids = get_u64_string_column(&batch, "partition_id")?;
        let mjd_min = get_f64_column(&batch, "mjd_min")?;
        let mjd_max = get_f64_column(&batch, "mjd_max")?;
        let arc_length = get_f64_column(&batch, "arc_length")?;
        let num_obs = get_i64_column(&batch, "num_obs")?;
        let num_observatories = get_i64_column(&batch, "num_observatories")?;
        let findable = get_optional_bool_column(&batch, "findable")?;
        let found_pure = get_i64_column(&batch, "found_pure")?;
        let found_contaminated = get_i64_column(&batch, "found_contaminated")?;
        let pure = get_i64_column(&batch, "pure")?;
        let pure_complete = get_i64_column(&batch, "pure_complete")?;
        let contaminated = get_i64_column(&batch, "contaminated")?;
        let contaminant = get_i64_column(&batch, "contaminant")?;
        let mixed = get_i64_column(&batch, "mixed")?;
        let obs_in_pure = get_i64_column(&batch, "obs_in_pure")?;
        let obs_in_pure_complete = get_i64_column(&batch, "obs_in_pure_complete")?;
        let obs_in_contaminated = get_i64_column(&batch, "obs_in_contaminated")?;
        let obs_as_contaminant = get_i64_column(&batch, "obs_as_contaminant")?;
        let obs_in_mixed = get_i64_column(&batch, "obs_in_mixed")?;

        for s in &object_ids_str {
            out.object_id.push(id_interner.intern(s));
        }
        out.partition_id.extend(partition_ids);
        out.mjd_min.extend(mjd_min);
        out.mjd_max.extend(mjd_max);
        out.arc_length.extend(arc_length);
        out.num_obs.extend(num_obs);
        out.num_observatories.extend(num_observatories);
        out.findable.extend(findable);
        out.found_pure.extend(found_pure);
        out.found_contaminated.extend(found_contaminated);
        out.pure.extend(pure);
        out.pure_complete.extend(pure_complete);
        out.contaminated.extend(contaminated);
        out.contaminant.extend(contaminant);
        out.mixed.extend(mixed);
        out.obs_in_pure.extend(obs_in_pure);
        out.obs_in_pure_complete.extend(obs_in_pure_complete);
        out.obs_in_contaminated.extend(obs_in_contaminated);
        out.obs_as_contaminant.extend(obs_as_contaminant);
        out.obs_in_mixed.extend(obs_in_mixed);
    }

    Ok(out)
}

/// Read a `Vec<PartitionSummary>` from a Parquet file written by
/// `write_partition_summaries`. `id` is parsed from its `LargeUtf8`
/// representation back to `u64`.
pub fn read_partition_summaries(path: &Path) -> Result<Vec<PartitionSummary>> {
    let file = std::fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    let mut out = Vec::new();

    for batch in reader {
        let batch = batch?;

        let ids = get_u64_string_column(&batch, "id")?;
        let start_night = get_i64_column(&batch, "start_night")?;
        let end_night = get_i64_column(&batch, "end_night")?;
        let observations = get_i64_column(&batch, "observations")?;
        let findable = get_optional_i64_column(&batch, "findable")?;
        let found = get_optional_i64_column(&batch, "found")?;
        let completeness = get_optional_f64_column(&batch, "completeness")?;
        let pure_known = get_optional_i64_column(&batch, "pure_known")?;
        let pure_unknown = get_optional_i64_column(&batch, "pure_unknown")?;
        let contaminated = get_optional_i64_column(&batch, "contaminated")?;
        let mixed = get_optional_i64_column(&batch, "mixed")?;

        for i in 0..batch.num_rows() {
            out.push(PartitionSummary {
                id: ids[i],
                start_night: start_night[i],
                end_night: end_night[i],
                observations: observations[i],
                findable: findable[i],
                found: found[i],
                completeness: completeness[i],
                pure_known: pure_known[i],
                pure_unknown: pure_unknown[i],
                contaminated: contaminated[i],
                mixed: mixed[i],
            });
        }
    }

    Ok(out)
}

/// Read `FindableObservations` from a Parquet file written by
/// `write_findable_observations`.
///
/// Note: the writer does not persist the `obs_ids` field, so the returned
/// `FindableObservations.obs_ids` is filled with `None` for every row. The
/// DIFI phase does not consume `obs_ids`, so this is sufficient for CIFI-output
/// reuse today.
///
/// Same interner ordering contract as `read_all_objects`.
pub fn read_findable_observations(
    path: &Path,
    id_interner: &mut StringInterner,
) -> Result<FindableObservations> {
    let file = std::fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    let mut out = FindableObservations::default();

    for batch in reader {
        let batch = batch?;

        let partition_ids = get_u64_string_column(&batch, "partition_id")?;
        let object_ids_str = get_string_column(&batch, "object_id")?;
        let discovery_night = get_optional_i64_column(&batch, "discovery_night")?;

        for i in 0..batch.num_rows() {
            out.partition_id.push(partition_ids[i]);
            out.object_id.push(id_interner.intern(&object_ids_str[i]));
            out.discovery_night.push(discovery_night[i]);
            out.obs_ids.push(None);
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------

fn write_props() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build()
}

/// Write AllObjects to a Parquet file, de-interning IDs back to strings.
pub fn write_all_objects(
    path: &Path,
    all_objects: &AllObjects,
    id_interner: &StringInterner,
) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("object_id", DataType::LargeUtf8, false),
        Field::new("partition_id", DataType::LargeUtf8, false),
        Field::new("mjd_min", DataType::Float64, false),
        Field::new("mjd_max", DataType::Float64, false),
        Field::new("arc_length", DataType::Float64, false),
        Field::new("num_obs", DataType::Int64, false),
        Field::new("num_observatories", DataType::Int64, false),
        Field::new("findable", DataType::Boolean, true),
        Field::new("found_pure", DataType::Int64, false),
        Field::new("found_contaminated", DataType::Int64, false),
        Field::new("pure", DataType::Int64, false),
        Field::new("pure_complete", DataType::Int64, false),
        Field::new("contaminated", DataType::Int64, false),
        Field::new("contaminant", DataType::Int64, false),
        Field::new("mixed", DataType::Int64, false),
        Field::new("obs_in_pure", DataType::Int64, false),
        Field::new("obs_in_pure_complete", DataType::Int64, false),
        Field::new("obs_in_contaminated", DataType::Int64, false),
        Field::new("obs_as_contaminant", DataType::Int64, false),
        Field::new("obs_in_mixed", DataType::Int64, false),
    ]));

    let object_ids: Vec<&str> = all_objects
        .object_id
        .iter()
        .map(|&id| id_interner.resolve(id).unwrap_or(""))
        .collect();
    let partition_ids: Vec<String> = all_objects
        .partition_id
        .iter()
        .map(|id| id.to_string())
        .collect();
    let partition_id_refs: Vec<&str> = partition_ids.iter().map(|s| s.as_str()).collect();

    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(LargeStringArray::from(object_ids)),
        Arc::new(LargeStringArray::from(partition_id_refs)),
        Arc::new(Float64Array::from(all_objects.mjd_min.clone())),
        Arc::new(Float64Array::from(all_objects.mjd_max.clone())),
        Arc::new(Float64Array::from(all_objects.arc_length.clone())),
        Arc::new(Int64Array::from(all_objects.num_obs.clone())),
        Arc::new(Int64Array::from(all_objects.num_observatories.clone())),
        Arc::new(BooleanArray::from(all_objects.findable.clone())),
        Arc::new(Int64Array::from(all_objects.found_pure.clone())),
        Arc::new(Int64Array::from(all_objects.found_contaminated.clone())),
        Arc::new(Int64Array::from(all_objects.pure.clone())),
        Arc::new(Int64Array::from(all_objects.pure_complete.clone())),
        Arc::new(Int64Array::from(all_objects.contaminated.clone())),
        Arc::new(Int64Array::from(all_objects.contaminant.clone())),
        Arc::new(Int64Array::from(all_objects.mixed.clone())),
        Arc::new(Int64Array::from(all_objects.obs_in_pure.clone())),
        Arc::new(Int64Array::from(all_objects.obs_in_pure_complete.clone())),
        Arc::new(Int64Array::from(all_objects.obs_in_contaminated.clone())),
        Arc::new(Int64Array::from(all_objects.obs_as_contaminant.clone())),
        Arc::new(Int64Array::from(all_objects.obs_in_mixed.clone())),
    ];

    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(write_props()))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Write AllLinkages to a Parquet file, de-interning IDs back to strings.
pub fn write_all_linkages(
    path: &Path,
    all_linkages: &AllLinkages,
    id_interner: &StringInterner,
) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("linkage_id", DataType::LargeUtf8, false),
        Field::new("partition_id", DataType::LargeUtf8, false),
        Field::new("linked_object_id", DataType::LargeUtf8, true),
        Field::new("num_obs", DataType::Int64, false),
        Field::new("num_obs_outside_partition", DataType::Int64, false),
        Field::new("num_members", DataType::Int64, false),
        Field::new("pure", DataType::Boolean, false),
        Field::new("pure_complete", DataType::Boolean, false),
        Field::new("contaminated", DataType::Boolean, false),
        Field::new("contamination", DataType::Float64, false),
        Field::new("mixed", DataType::Boolean, false),
        Field::new("found_pure", DataType::Boolean, false),
        Field::new("found_contaminated", DataType::Boolean, false),
    ]));

    let linkage_ids: Vec<&str> = all_linkages
        .linkage_id
        .iter()
        .map(|&id| id_interner.resolve(id).unwrap_or(""))
        .collect();
    let partition_ids: Vec<String> = all_linkages
        .partition_id
        .iter()
        .map(|id| id.to_string())
        .collect();
    let partition_id_refs: Vec<&str> = partition_ids.iter().map(|s| s.as_str()).collect();
    let linked_obj_ids: Vec<Option<&str>> = all_linkages
        .linked_object_id
        .iter()
        .map(|&id| id_interner.resolve(id))
        .collect();

    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(LargeStringArray::from(linkage_ids)),
        Arc::new(LargeStringArray::from(partition_id_refs)),
        Arc::new(LargeStringArray::from(linked_obj_ids)),
        Arc::new(Int64Array::from(all_linkages.num_obs.clone())),
        Arc::new(Int64Array::from(
            all_linkages.num_obs_outside_partition.clone(),
        )),
        Arc::new(Int64Array::from(all_linkages.num_members.clone())),
        Arc::new(BooleanArray::from(all_linkages.pure.clone())),
        Arc::new(BooleanArray::from(all_linkages.pure_complete.clone())),
        Arc::new(BooleanArray::from(all_linkages.contaminated.clone())),
        Arc::new(Float64Array::from(all_linkages.contamination.clone())),
        Arc::new(BooleanArray::from(all_linkages.mixed.clone())),
        Arc::new(BooleanArray::from(all_linkages.found_pure.clone())),
        Arc::new(BooleanArray::from(all_linkages.found_contaminated.clone())),
    ];

    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(write_props()))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Write partition summaries to a Parquet file.
pub fn write_partition_summaries(path: &Path, summaries: &[PartitionSummary]) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::LargeUtf8, false),
        Field::new("start_night", DataType::Int64, false),
        Field::new("end_night", DataType::Int64, false),
        Field::new("observations", DataType::Int64, false),
        Field::new("findable", DataType::Int64, true),
        Field::new("found", DataType::Int64, true),
        Field::new("completeness", DataType::Float64, true),
        Field::new("pure_known", DataType::Int64, true),
        Field::new("pure_unknown", DataType::Int64, true),
        Field::new("contaminated", DataType::Int64, true),
        Field::new("mixed", DataType::Int64, true),
    ]));

    let ids: Vec<String> = summaries.iter().map(|s| s.id.to_string()).collect();
    let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();

    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(LargeStringArray::from(id_refs)),
        Arc::new(Int64Array::from(
            summaries.iter().map(|s| s.start_night).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            summaries.iter().map(|s| s.end_night).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            summaries.iter().map(|s| s.observations).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            summaries.iter().map(|s| s.findable).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            summaries.iter().map(|s| s.found).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            summaries.iter().map(|s| s.completeness).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            summaries.iter().map(|s| s.pure_known).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            summaries.iter().map(|s| s.pure_unknown).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            summaries.iter().map(|s| s.contaminated).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            summaries.iter().map(|s| s.mixed).collect::<Vec<_>>(),
        )),
    ];

    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(write_props()))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Write findable observations to a Parquet file.
pub fn write_findable_observations(
    path: &Path,
    findable: &FindableObservations,
    id_interner: &StringInterner,
) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("partition_id", DataType::LargeUtf8, false),
        Field::new("object_id", DataType::LargeUtf8, false),
        Field::new("discovery_night", DataType::Int64, true),
    ]));

    let partition_ids: Vec<String> = findable
        .partition_id
        .iter()
        .map(|id| id.to_string())
        .collect();
    let partition_id_refs: Vec<&str> = partition_ids.iter().map(|s| s.as_str()).collect();
    let object_ids: Vec<&str> = findable
        .object_id
        .iter()
        .map(|&id| id_interner.resolve(id).unwrap_or(""))
        .collect();

    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(LargeStringArray::from(partition_id_refs)),
        Arc::new(LargeStringArray::from(object_ids)),
        Arc::new(Int64Array::from(findable.discovery_night.clone())),
    ];

    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(write_props()))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Write ignored-linkage records to a Parquet file, de-interning IDs back to
/// strings. Consumers can union with `all_linkages.parquet` by
/// `(linkage_id, partition_id)`.
pub fn write_ignored_linkages(
    path: &Path,
    ignored: &IgnoredLinkages,
    id_interner: &StringInterner,
) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("linkage_id", DataType::LargeUtf8, false),
        Field::new("partition_id", DataType::LargeUtf8, false),
        Field::new("reason", DataType::LargeUtf8, false),
        Field::new("num_obs", DataType::Int64, false),
        Field::new("num_members", DataType::Int64, false),
    ]));

    let linkage_ids: Vec<&str> = ignored
        .linkage_id
        .iter()
        .map(|&id| id_interner.resolve(id).unwrap_or(""))
        .collect();
    let partition_ids: Vec<String> = ignored
        .partition_id
        .iter()
        .map(|id| id.to_string())
        .collect();
    let partition_id_refs: Vec<&str> = partition_ids.iter().map(|s| s.as_str()).collect();
    let reasons: Vec<&str> = ignored.reason.iter().map(|r| r.as_str()).collect();

    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(LargeStringArray::from(linkage_ids)),
        Arc::new(LargeStringArray::from(partition_id_refs)),
        Arc::new(LargeStringArray::from(reasons)),
        Arc::new(Int64Array::from(ignored.num_obs.clone())),
        Arc::new(Int64Array::from(ignored.num_members.clone())),
    ];

    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(write_props()))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Predefined column sets for common use cases.
pub mod columns {
    /// Columns needed for CIFI analysis (both singleton and tracklet metrics).
    pub const CIFI: &[&str] = &[
        "id",
        "night",
        "object_id",
        "time",
        "ra",
        "dec",
        "observatory_code",
    ];

    /// Minimal columns for DIFI linkage classification.
    pub const DIFI: &[&str] = &["id", "night", "object_id"];
}
