use clap::{Command, arg, value_parser};
use datafusion::{
    arrow::{
        array::{
            Array, ArrayRef, AsArray, DictionaryArray, GenericByteArray,
            GenericByteDictionaryBuilder, PrimitiveArray, RecordBatch, StringArray,
        },
        datatypes::{DataType, Field, Schema, UInt32Type, Utf8Type},
    },
    parquet::{
        arrow::{
            ArrowWriter, ProjectionMask,
            arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder},
        },
        basic::Compression,
        file::{metadata::ParquetMetaDataReader, properties::WriterProperties},
    },
};
use fsst::Compressor;
use liquid_cache_parquet::liquid_array::{LiquidByteArray, raw::BitPackedArray, raw::FsstArray};
use std::{
    fs::File,
    num::NonZero,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Debug, serde::Serialize)]
struct Bencher {
    column_id: usize,
    parquet_to_arrow_time: Duration,
    arrow_to_dict_time: Duration,
    dict_to_bit_packed_time: Duration,
    bit_packed_to_liquid_time: Duration,
    parquet_column_size: usize,
    arrow_column_size: usize,
    dict_column_size: usize,
    bit_packed_column_size: usize,
    liquid_column_size: usize,
    liquid_to_bit_packed_time: Duration,
    bit_packed_to_dict_time: Duration,
    dict_to_arrow_time: Duration,
    arrow_to_parquet_time: Duration,
}

impl Bencher {
    fn new(path: &Path, column_id: usize) -> Self {
        let mut parquet_column_size = 0;
        let file = std::fs::File::open(path).unwrap();
        let metadata = ParquetMetaDataReader::new()
            .parse_and_finish(&file)
            .unwrap();
        for rg in metadata.row_groups() {
            for (i, col) in rg.columns().iter().enumerate() {
                if i == column_id {
                    parquet_column_size += col.compressed_size() as usize;
                }
            }
        }
        log::info!("Parquet column size: {}", parquet_column_size);

        Self {
            column_id,
            parquet_column_size,
            arrow_column_size: 0,
            dict_column_size: 0,
            bit_packed_column_size: 0,
            liquid_column_size: 0,
            parquet_to_arrow_time: Duration::from_secs(0),
            arrow_to_dict_time: Duration::from_secs(0),
            dict_to_bit_packed_time: Duration::from_secs(0),
            bit_packed_to_liquid_time: Duration::from_secs(0),
            liquid_to_bit_packed_time: Duration::from_secs(0),
            bit_packed_to_dict_time: Duration::from_secs(0),
            dict_to_arrow_time: Duration::from_secs(0),
            arrow_to_parquet_time: Duration::from_secs(0),
        }
    }

    fn parquet_to_arrow_one(
        &mut self,
        reader: &mut ParquetRecordBatchReader,
    ) -> Option<RecordBatch> {
        let start = Instant::now();
        let batch = reader.next();
        let duration = start.elapsed();
        if let Some(batch) = batch {
            let batch = batch.unwrap();
            self.parquet_to_arrow_time += duration;
            self.arrow_column_size += batch.get_array_memory_size();
            Some(batch)
        } else {
            self.parquet_to_arrow_time += duration;
            None
        }
    }

    fn arrow_to_dict_one(&mut self, array: &ArrayRef) -> DictionaryArray<UInt32Type> {
        let array = array.as_string::<i32>();
        let start = Instant::now();
        let mut builder = GenericByteDictionaryBuilder::<UInt32Type, Utf8Type>::new();
        for s in array.iter() {
            builder.append_option(s);
        }
        let dict = builder.finish();
        let duration = start.elapsed();
        self.dict_column_size += dict.get_array_memory_size();
        self.arrow_to_dict_time += duration;
        dict
    }

    fn dict_to_bit_pack_one(
        &mut self,
        dict: DictionaryArray<UInt32Type>,
    ) -> (BitPackedArray<UInt32Type>, ArrayRef) {
        let start = Instant::now();
        let (keys, values) = dict.into_parts();
        let distinct_count = values.len();
        let bit_width = get_bit_width(distinct_count as u64);
        let bit_packed_array =
            BitPackedArray::from_primitive(keys, NonZero::new(bit_width).unwrap());
        let duration = start.elapsed();
        self.bit_packed_column_size += bit_packed_array.get_array_memory_size();
        self.bit_packed_column_size += values.get_array_memory_size();
        self.dict_to_bit_packed_time += duration;
        (bit_packed_array, values)
    }

    fn bit_packed_to_liquid_one(
        &mut self,
        keys: BitPackedArray<UInt32Type>,
        values: ArrayRef,
        compressor: Arc<Compressor>,
    ) -> (BitPackedArray<UInt32Type>, FsstArray) {
        let start = Instant::now();
        let fsst_values = if let Some(values) = values.as_string_opt::<i32>() {
            FsstArray::from_byte_array_with_compressor(values, compressor)
        } else if let Some(values) = values.as_binary_opt::<i32>() {
            FsstArray::from_byte_array_with_compressor(values, compressor)
        } else {
            panic!("Unsupported dictionary type")
        };
        let duration = start.elapsed();
        self.bit_packed_to_liquid_time += duration;
        self.liquid_column_size += fsst_values.get_array_memory_size();
        self.liquid_column_size += keys.get_array_memory_size();
        (keys, fsst_values)
    }

    fn liquid_to_bit_packed_one(&mut self, fsst_values: FsstArray) -> StringArray {
        let start = Instant::now();
        let array = fsst_values.to_arrow_byte_array::<Utf8Type>();
        let duration = start.elapsed();
        self.liquid_to_bit_packed_time += duration;
        array
    }

    fn bit_packed_to_dict_one(
        &mut self,
        bit_packed_array: BitPackedArray<UInt32Type>,
    ) -> PrimitiveArray<UInt32Type> {
        let start = Instant::now();
        let array = bit_packed_array.to_primitive();
        let duration = start.elapsed();
        self.bit_packed_to_dict_time += duration;
        array
    }

    fn dict_to_arrow_one(
        &mut self,
        key: PrimitiveArray<UInt32Type>,
        value: GenericByteArray<Utf8Type>,
    ) -> ArrayRef {
        let start = Instant::now();
        let dict = unsafe { DictionaryArray::new_unchecked(key, Arc::new(value)) };
        let array =
            datafusion::arrow::compute::cast(&dict, &datafusion::arrow::datatypes::DataType::Utf8)
                .unwrap();
        let duration = start.elapsed();
        self.dict_to_arrow_time += duration;
        array
    }

    fn arrow_to_parquet_one(&mut self, array: ArrayRef, writer: &mut ArrowWriter<&mut Vec<u8>>) {
        let field = Field::new("test", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();

        let start = Instant::now();
        writer.write(&batch).unwrap();
        let duration = start.elapsed();
        self.arrow_to_parquet_time += duration;
    }
}

pub(crate) fn get_bit_width(max_value: u64) -> u8 {
    64 - max_value.leading_zeros() as u8
}

fn train_compressor(values: &StringArray) -> Arc<Compressor> {
    LiquidByteArray::train_compressor(values.iter())
}

fn build_reader(path: &Path, column_id: usize) -> ParquetRecordBatchReader {
    let file = File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let mask = ProjectionMask::roots(builder.parquet_schema(), [column_id]);
    builder
        .with_projection(mask)
        .with_batch_size(8192)
        .build()
        .unwrap()
}

fn bench(path: &Path, column_id: usize) -> Bencher {
    let mut bencher = Bencher::new(path, column_id);

    let compressor = {
        let mut reader = build_reader(path, column_id);
        let batch = reader.next().unwrap().unwrap();
        let array = batch.column(0).as_string::<i32>();
        train_compressor(array)
    };
    let mut reader = build_reader(path, column_id);

    let mut buffer = Vec::new();
    let props = WriterProperties::builder()
        .set_compression(Compression::LZ4)
        .build();
    let schema = Arc::new(Schema::new(vec![Field::new("test", DataType::Utf8, false)]));
    let mut writer = ArrowWriter::try_new(&mut buffer, schema.clone(), Some(props)).unwrap();

    while let Some(batch) = bencher.parquet_to_arrow_one(&mut reader) {
        let dict = bencher.arrow_to_dict_one(batch.column(0));
        let (bit_packed_array, values) = bencher.dict_to_bit_pack_one(dict);
        let (keys, fsst_values) =
            bencher.bit_packed_to_liquid_one(bit_packed_array, values, compressor.clone());
        let values = bencher.liquid_to_bit_packed_one(fsst_values);
        let key = bencher.bit_packed_to_dict_one(keys);
        let array = bencher.dict_to_arrow_one(key, values);
        bencher.arrow_to_parquet_one(array, &mut writer);
    }
    bencher
}

fn main() {
    env_logger::builder().format_timestamp(None).init();

    let matches = Command::new("LiquidParquet Encoding Bench")
        .arg(
            arg!(--file <PATH>)
                .required(true)
                .help("Path to the parquet file")
                .value_parser(value_parser!(std::path::PathBuf)),
        )
        .arg(
            arg!(--column <COLUMN>)
                .required(false)
                .default_value("13")
                .help("Column index to benchmark")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            arg!(--iterations <ITERATIONS>)
                .required(false)
                .default_value("1")
                .help("Number of iterations to run")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            arg!(--output <OUTPUT>)
                .required(false)
                .default_value("output.json")
                .help("Output file")
                .value_parser(value_parser!(std::path::PathBuf)),
        )
        .get_matches();

    let path = matches.get_one::<std::path::PathBuf>("file").unwrap();
    let column = matches.get_one::<usize>("column").unwrap_or(&13);
    let output = matches.get_one::<std::path::PathBuf>("output").unwrap();
    let iterations = matches.get_one::<usize>("iterations").unwrap_or(&1);

    let mut results = Vec::new();
    for _ in 0..*iterations {
        let bencher = bench(path, *column);
        log::info!("Bencher: {:?}", bencher);
        results.push(bencher);
    }
    let file = File::create(output).unwrap();
    serde_json::to_writer(file, &results).unwrap();
}
