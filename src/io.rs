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
    AllLinkages, AllObjects, FindableObservations, LinkageMembers, Observations, StringInterner,
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
