// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Contains implementations of the reader traits FileReader, RowGroupReader and PageReader
//! Also contains implementations of the ChunkReader for files (with buffering) and byte arrays (RAM)

use bytes::Bytes;
use parquet::basic::{Encoding, Type};
use parquet::column::page::{Page, PageMetadata, PageReader};
use parquet::compression::{Codec, CodecOptions, create_codec};
use parquet::errors::{ParquetError, Result};
use parquet::file::metadata::ColumnChunkMetaData;
use parquet::file::reader::ChunkReader;
use parquet::file::statistics;
use parquet::format::{PageHeader, PageLocation, PageType};
use parquet::thrift::TSerializable;
use std::collections::VecDeque;
use std::{io::Read, sync::Arc};

/// Reads a [`PageHeader`] from the provided [`Read`]
pub(crate) fn read_page_header<T: Read>(input: &mut T) -> Result<PageHeader> {
    let mut prot = thrift::protocol::TCompactInputProtocol::new(input);
    Ok(PageHeader::read_from_in_protocol(&mut prot)?)
}

/// Reads a [`PageHeader`] from the provided [`Read`] returning the number of bytes read.
fn read_page_header_len<T: Read>(input: &mut T) -> Result<(usize, PageHeader)> {
    /// A wrapper around a [`std::io::Read`] that keeps track of the bytes read
    struct TrackedRead<R> {
        inner: R,
        bytes_read: usize,
    }

    impl<R: Read> Read for TrackedRead<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let v = self.inner.read(buf)?;
            self.bytes_read += v;
            Ok(v)
        }
    }

    let mut tracked = TrackedRead {
        inner: input,
        bytes_read: 0,
    };
    let header = read_page_header(&mut tracked)?;
    Ok((tracked.bytes_read, header))
}

/// Decodes a [`Page`] from the provided `buffer`
pub(crate) fn decode_page(
    page_header: PageHeader,
    buffer: Bytes,
    physical_type: Type,
    decompressor: Option<&mut Box<dyn Codec>>,
) -> Result<Page> {
    // When processing data page v2, depending on enabled compression for the
    // page, we should account for uncompressed data ('offset') of
    // repetition and definition levels.
    //
    // We always use 0 offset for other pages other than v2, `true` flag means
    // that compression will be applied if decompressor is defined
    let mut offset: usize = 0;
    let mut can_decompress = true;

    if let Some(ref header_v2) = page_header.data_page_header_v2 {
        if header_v2.definition_levels_byte_length < 0
            || header_v2.repetition_levels_byte_length < 0
            || header_v2.definition_levels_byte_length + header_v2.repetition_levels_byte_length
                > page_header.uncompressed_page_size
        {
            return Err(ParquetError::General(format!(
                "DataPage v2 header contains implausible values \
                    for definition_levels_byte_length ({}) \
                    and repetition_levels_byte_length ({}) \
                    given DataPage header provides uncompressed_page_size ({})",
                header_v2.definition_levels_byte_length,
                header_v2.repetition_levels_byte_length,
                page_header.uncompressed_page_size
            )));
        }
        offset = usize::try_from(
            header_v2.definition_levels_byte_length + header_v2.repetition_levels_byte_length,
        )?;
        // When is_compressed flag is missing the page is considered compressed
        can_decompress = header_v2.is_compressed.unwrap_or(true);
    }

    // TODO: page header could be huge because of statistics. We should set a
    // maximum page header size and abort if that is exceeded.
    let buffer = match decompressor {
        Some(decompressor) if can_decompress => {
            let uncompressed_page_size = usize::try_from(page_header.uncompressed_page_size)?;
            let decompressed_size = uncompressed_page_size - offset;
            let mut decompressed = Vec::with_capacity(uncompressed_page_size);
            decompressed.extend_from_slice(&buffer.as_ref()[..offset]);
            if decompressed_size > 0 {
                let compressed = &buffer.as_ref()[offset..];
                decompressor.decompress(compressed, &mut decompressed, Some(decompressed_size))?;
            }

            if decompressed.len() != uncompressed_page_size {
                return Err(ParquetError::General(format!(
                    "Actual decompressed size doesn't match the expected one ({} vs {})",
                    decompressed.len(),
                    uncompressed_page_size
                )));
            }

            Bytes::from(decompressed)
        }
        _ => buffer,
    };

    let result = match page_header.type_ {
        PageType::DICTIONARY_PAGE => {
            let dict_header = page_header.dictionary_page_header.as_ref().ok_or_else(|| {
                ParquetError::General("Missing dictionary page header".to_string())
            })?;
            let is_sorted = dict_header.is_sorted.unwrap_or(false);
            Page::DictionaryPage {
                buf: buffer,
                num_values: dict_header.num_values.try_into()?,
                encoding: Encoding::try_from(dict_header.encoding)?,
                is_sorted,
            }
        }
        PageType::DATA_PAGE => {
            let header = page_header
                .data_page_header
                .ok_or_else(|| ParquetError::General("Missing V1 data page header".to_string()))?;
            Page::DataPage {
                buf: buffer,
                num_values: header.num_values.try_into()?,
                encoding: Encoding::try_from(header.encoding)?,
                def_level_encoding: Encoding::try_from(header.definition_level_encoding)?,
                rep_level_encoding: Encoding::try_from(header.repetition_level_encoding)?,
                statistics: statistics::from_thrift(physical_type, header.statistics)?,
            }
        }
        PageType::DATA_PAGE_V2 => {
            let header = page_header
                .data_page_header_v2
                .ok_or_else(|| ParquetError::General("Missing V2 data page header".to_string()))?;
            let is_compressed = header.is_compressed.unwrap_or(true);
            Page::DataPageV2 {
                buf: buffer,
                num_values: header.num_values.try_into()?,
                encoding: Encoding::try_from(header.encoding)?,
                num_nulls: header.num_nulls.try_into()?,
                num_rows: header.num_rows.try_into()?,
                def_levels_byte_len: header.definition_levels_byte_length.try_into()?,
                rep_levels_byte_len: header.repetition_levels_byte_length.try_into()?,
                is_compressed,
                statistics: statistics::from_thrift(physical_type, header.statistics)?,
            }
        }
        _ => {
            // For unknown page type (e.g., INDEX_PAGE), skip and read next.
            unimplemented!("Page type {:?} is not supported", page_header.type_)
        }
    };

    Ok(result)
}

enum SerializedPageReaderState {
    Values {
        /// The current byte offset in the reader
        offset: usize,

        /// The length of the chunk in bytes
        remaining_bytes: usize,

        // If the next page header has already been "peeked", we will cache it and it`s length here
        next_page_header: Option<Box<PageHeader>>,

        /// The index of the data page within this column chunk
        page_ordinal: usize,

        /// Whether the next page is expected to be a dictionary page
        require_dictionary: bool,
    },
    Pages {
        /// Remaining page locations
        page_locations: VecDeque<PageLocation>,
        /// Remaining dictionary location if any
        dictionary_page: Option<PageLocation>,
        /// The total number of rows in this column chunk
        total_rows: usize,
    },
}

/// A serialized implementation for Parquet [`PageReader`].
pub struct SerializedPageReader<R: ChunkReader> {
    /// The chunk reader
    reader: Arc<R>,

    /// The compression codec for this column chunk. Only set for non-PLAIN codec.
    decompressor: Option<Box<dyn Codec>>,

    /// Column chunk type.
    physical_type: Type,

    state: SerializedPageReaderState,
}

impl<R: ChunkReader> SerializedPageReader<R> {
    /// Creates a new serialized page reader from a chunk reader and metadata
    pub fn new(
        reader: Arc<R>,
        column_chunk_metadata: &ColumnChunkMetaData,
        total_rows: usize,
        page_locations: Option<Vec<PageLocation>>,
    ) -> Result<Self> {
        SerializedPageReader::new_with_properties(
            reader,
            column_chunk_metadata,
            total_rows,
            page_locations,
        )
    }

    /// Creates a new serialized page with custom options.
    pub fn new_with_properties(
        reader: Arc<R>,
        meta: &ColumnChunkMetaData,
        total_rows: usize,
        page_locations: Option<Vec<PageLocation>>,
    ) -> Result<Self> {
        let codec_options = CodecOptions::default();
        let decompressor = create_codec(meta.compression(), &codec_options)?;
        let (start, len) = meta.byte_range();

        let state = match page_locations {
            Some(locations) => {
                let dictionary_page = match locations.first() {
                    Some(dict_offset) if dict_offset.offset as u64 != start => Some(PageLocation {
                        offset: start as i64,
                        compressed_page_size: (dict_offset.offset as u64 - start) as i32,
                        first_row_index: 0,
                    }),
                    _ => None,
                };

                SerializedPageReaderState::Pages {
                    page_locations: locations.into(),
                    dictionary_page,
                    total_rows,
                }
            }
            None => SerializedPageReaderState::Values {
                offset: usize::try_from(start)?,
                remaining_bytes: usize::try_from(len)?,
                next_page_header: None,
                page_ordinal: 0,
                require_dictionary: meta.dictionary_page_offset().is_some(),
            },
        };
        Ok(Self {
            reader,
            decompressor,
            state,
            physical_type: meta.column_type(),
        })
    }

    /// Similar to `peek_next_page`, but returns the offset of the next page instead of the page metadata.
    /// Unlike page metadata, an offset can uniquely identify a page.
    ///
    /// This is used when we need to read parquet with row-filter, and we don't want to decompress the page twice.
    /// This function allows us to check if the next page is being cached or read previously.
    pub(crate) fn peek_next_page_offset(&mut self) -> Result<Option<usize>> {
        match &mut self.state {
            SerializedPageReaderState::Values {
                offset,
                remaining_bytes,
                next_page_header,
                ..
            } => {
                loop {
                    if *remaining_bytes == 0 {
                        return Ok(None);
                    }
                    return if let Some(header) = next_page_header.as_ref() {
                        use parquet::column::page::PageMetadata;

                        if let Ok(_page_meta) = PageMetadata::try_from(&**header) {
                            Ok(Some(*offset))
                        } else {
                            // For unknown page type (e.g., INDEX_PAGE), skip and read next.
                            *next_page_header = None;
                            continue;
                        }
                    } else {
                        let mut read = self.reader.get_read(*offset as u64)?;
                        let (header_len, header) = read_page_header_len(&mut read)?;
                        *offset += header_len;
                        *remaining_bytes -= header_len;
                        let page_meta = if let Ok(_page_meta) = PageMetadata::try_from(&header) {
                            Ok(Some(*offset))
                        } else {
                            // For unknown page type (e.g., INDEX_PAGE), skip and read next.
                            continue;
                        };
                        *next_page_header = Some(Box::new(header));
                        page_meta
                    };
                }
            }
            SerializedPageReaderState::Pages {
                page_locations,
                dictionary_page,
                ..
            } => {
                if let Some(page) = dictionary_page {
                    Ok(Some(usize::try_from(page.offset)?))
                } else if let Some(page) = page_locations.front() {
                    Ok(Some(usize::try_from(page.offset)?))
                } else {
                    Ok(None)
                }
            }
        }
    }
}

impl<R: ChunkReader> Iterator for SerializedPageReader<R> {
    type Item = Result<Page>;

    fn next(&mut self) -> Option<Self::Item> {
        self.get_next_page().transpose()
    }
}

fn verify_page_header_len(header_len: usize, remaining_bytes: usize) -> Result<()> {
    if header_len > remaining_bytes {
        return Err(ParquetError::General("Invalid page header".to_string()));
    }
    Ok(())
}

fn verify_page_size(
    compressed_size: i32,
    uncompressed_size: i32,
    remaining_bytes: usize,
) -> Result<()> {
    // The page's compressed size should not exceed the remaining bytes that are
    // available to read. The page's uncompressed size is the expected size
    // after decompression, which can never be negative.
    if compressed_size < 0 || compressed_size as usize > remaining_bytes || uncompressed_size < 0 {
        return Err(ParquetError::General("Invalid page header".to_string()));
    }
    Ok(())
}

impl<R: ChunkReader> PageReader for SerializedPageReader<R> {
    fn get_next_page(&mut self) -> Result<Option<Page>> {
        loop {
            let page = match &mut self.state {
                SerializedPageReaderState::Values {
                    offset,
                    remaining_bytes: remaining,
                    next_page_header,
                    page_ordinal,
                    require_dictionary,
                } => {
                    if *remaining == 0 {
                        return Ok(None);
                    }

                    let mut read = self.reader.get_read(*offset as u64)?;
                    let header = if let Some(header) = next_page_header.take() {
                        *header
                    } else {
                        let (header_len, header) = read_page_header_len(&mut read)?;

                        verify_page_header_len(header_len, *remaining)?;
                        *offset += header_len;
                        *remaining -= header_len;
                        header
                    };
                    verify_page_size(
                        header.compressed_page_size,
                        header.uncompressed_page_size,
                        *remaining,
                    )?;
                    let data_len = header.compressed_page_size as usize;
                    *offset += data_len;
                    *remaining -= data_len;

                    if header.type_ == PageType::INDEX_PAGE {
                        continue;
                    }

                    let mut buffer = Vec::with_capacity(data_len);
                    let read = read.take(data_len as u64).read_to_end(&mut buffer)?;

                    if read != data_len {
                        return Err(ParquetError::General(format!(
                            "Expected to read {data_len} bytes of page, read only {read}"
                        )));
                    }

                    let page = decode_page(
                        header,
                        Bytes::from(buffer),
                        self.physical_type,
                        self.decompressor.as_mut(),
                    )?;
                    if page.is_data_page() {
                        *page_ordinal += 1;
                    } else if page.is_dictionary_page() {
                        *require_dictionary = false;
                    }
                    page
                }
                SerializedPageReaderState::Pages {
                    page_locations,
                    dictionary_page,
                    ..
                } => {
                    let front = match dictionary_page
                        .take()
                        .or_else(|| page_locations.pop_front())
                    {
                        Some(front) => front,
                        None => return Ok(None),
                    };

                    let page_len = usize::try_from(front.compressed_page_size)?;

                    let buffer = self.reader.get_bytes(front.offset as u64, page_len)?;

                    let mut prot = super::thrift::TCompactSliceInputProtocol::new(buffer.as_ref());
                    let header = PageHeader::read_from_in_protocol(&mut prot)?;
                    let offset = buffer.len() - prot.as_slice().len();

                    let bytes = buffer.slice(offset..);
                    decode_page(
                        header,
                        bytes,
                        self.physical_type,
                        self.decompressor.as_mut(),
                    )?
                }
            };

            return Ok(Some(page));
        }
    }

    fn peek_next_page(&mut self) -> Result<Option<PageMetadata>> {
        match &mut self.state {
            SerializedPageReaderState::Values {
                offset,
                remaining_bytes,
                next_page_header,
                ..
            } => {
                loop {
                    if *remaining_bytes == 0 {
                        return Ok(None);
                    }
                    return if let Some(header) = next_page_header.as_ref() {
                        if let Ok(page_meta) = (&**header).try_into() {
                            Ok(Some(page_meta))
                        } else {
                            // For unknown page type (e.g., INDEX_PAGE), skip and read next.
                            *next_page_header = None;
                            continue;
                        }
                    } else {
                        let mut read = self.reader.get_read(*offset as u64)?;
                        let (header_len, header) = read_page_header_len(&mut read)?;
                        verify_page_header_len(header_len, *remaining_bytes)?;
                        *offset += header_len;
                        *remaining_bytes -= header_len;
                        let page_meta = if let Ok(page_meta) = (&header).try_into() {
                            Ok(Some(page_meta))
                        } else {
                            // For unknown page type (e.g., INDEX_PAGE), skip and read next.
                            continue;
                        };
                        *next_page_header = Some(Box::new(header));
                        page_meta
                    };
                }
            }
            SerializedPageReaderState::Pages {
                page_locations,
                dictionary_page,
                total_rows,
            } => {
                if dictionary_page.is_some() {
                    Ok(Some(PageMetadata {
                        num_rows: None,
                        num_levels: None,
                        is_dict: true,
                    }))
                } else if let Some(page) = page_locations.front() {
                    let next_rows = page_locations
                        .get(1)
                        .map(|x| x.first_row_index as usize)
                        .unwrap_or(*total_rows);

                    Ok(Some(PageMetadata {
                        num_rows: Some(next_rows - page.first_row_index as usize),
                        num_levels: None,
                        is_dict: false,
                    }))
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn skip_next_page(&mut self) -> Result<()> {
        match &mut self.state {
            SerializedPageReaderState::Values {
                offset,
                remaining_bytes,
                next_page_header,
                ..
            } => {
                if let Some(buffered_header) = next_page_header.take() {
                    verify_page_size(
                        buffered_header.compressed_page_size,
                        buffered_header.uncompressed_page_size,
                        *remaining_bytes,
                    )?;
                    // The next page header has already been peeked, so just advance the offset
                    *offset += buffered_header.compressed_page_size as usize;
                    *remaining_bytes -= buffered_header.compressed_page_size as usize;
                } else {
                    let mut read = self.reader.get_read(*offset as u64)?;
                    let (header_len, header) = read_page_header_len(&mut read)?;
                    verify_page_header_len(header_len, *remaining_bytes)?;
                    verify_page_size(
                        header.compressed_page_size,
                        header.uncompressed_page_size,
                        *remaining_bytes,
                    )?;
                    let data_page_size = header.compressed_page_size as usize;
                    *offset += header_len + data_page_size;
                    *remaining_bytes -= header_len + data_page_size;
                }
                Ok(())
            }
            SerializedPageReaderState::Pages {
                page_locations,
                dictionary_page,
                ..
            } => {
                if dictionary_page.is_some() {
                    // If a dictionary page exists, consume it by taking it (sets to None)
                    dictionary_page.take();
                } else {
                    // If no dictionary page exists, simply pop the data page from page_locations
                    page_locations.pop_front();
                }

                Ok(())
            }
        }
    }

    fn at_record_boundary(&mut self) -> Result<bool> {
        match &mut self.state {
            SerializedPageReaderState::Values { .. } => Ok(self.peek_next_page()?.is_none()),
            SerializedPageReaderState::Pages { .. } => Ok(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::column::page::PageReader;
    use parquet::file::properties::WriterProperties;
    use parquet::file::reader::FileReader;
    use parquet::file::serialized_reader::SerializedFileReader as ParquetSerializedFileReader;
    use std::collections::HashSet;
    use std::fs::File;
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    // Helper to create test parquet files
    fn create_test_parquet_data(
        compression: Compression,
        num_row_groups: usize,
        rows_per_group: usize,
    ) -> NamedTempFile {
        use arrow::array::{ArrayRef, Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));

        let mut temp_file = NamedTempFile::new().unwrap();
        let props = WriterProperties::builder()
            .set_compression(compression)
            .set_dictionary_enabled(true)
            .build();

        let mut writer = ArrowWriter::try_new(&mut temp_file, schema.clone(), Some(props)).unwrap();

        for rg in 0..num_row_groups {
            let id_data: Vec<i32> = ((rg * rows_per_group)..((rg + 1) * rows_per_group))
                .map(|i| i as i32)
                .collect();
            let name_data: Vec<Option<String>> = id_data
                .iter()
                .map(|&i| {
                    if i % 3 == 0 {
                        None
                    } else {
                        Some(format!("name_{i}"))
                    }
                })
                .collect();

            let id_array: ArrayRef = Arc::new(Int32Array::from(id_data));
            let name_array: ArrayRef = Arc::new(StringArray::from(name_data));

            let batch = RecordBatch::try_new(schema.clone(), vec![id_array, name_array]).unwrap();
            writer.write(&batch).unwrap();
        }

        writer.close().unwrap();
        temp_file
    }

    fn create_test_page_reader_from_file(
        file: &File,
        row_group: usize,
        column: usize,
    ) -> Result<SerializedPageReader<File>> {
        let reader = ParquetSerializedFileReader::new(file.try_clone().unwrap())?;
        let metadata = reader.metadata();
        let row_group_metadata = metadata.row_group(row_group);
        let column_chunk_metadata = row_group_metadata.column(column);

        SerializedPageReader::new(
            Arc::new(file.try_clone().unwrap()),
            column_chunk_metadata,
            row_group_metadata.num_rows().try_into().unwrap(),
            None, // No page locations for this test
        )
    }

    fn create_test_page_reader_with_page_locations(
        file: &File,
        row_group: usize,
        column: usize,
    ) -> Result<SerializedPageReader<File>> {
        let reader = ParquetSerializedFileReader::new(file.try_clone().unwrap())?;
        let metadata = reader.metadata();
        let row_group_metadata = metadata.row_group(row_group);
        let column_chunk_metadata = row_group_metadata.column(column);

        // Create some dummy page locations for testing Pages state
        let page_locations = vec![parquet::format::PageLocation {
            offset: column_chunk_metadata.data_page_offset(),
            compressed_page_size: 100,
            first_row_index: 0,
        }];

        SerializedPageReader::new(
            Arc::new(file.try_clone().unwrap()),
            column_chunk_metadata,
            row_group_metadata.num_rows().try_into().unwrap(),
            Some(page_locations),
        )
    }

    #[test]
    fn test_basic_page_reading() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        let mut page_count = 0;
        while let Ok(Some(_page)) = page_reader.get_next_page() {
            page_count += 1;
        }

        assert!(page_count > 0, "Should read at least one page");
    }

    #[test]
    fn test_page_iterator() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();
        let page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        let pages: Vec<_> = page_reader.collect();
        assert!(!pages.is_empty(), "Iterator should yield pages");

        // All results should be Ok
        for page_result in &pages {
            assert!(page_result.is_ok(), "All pages should be readable");
        }
    }

    #[test]
    fn test_peek_next_page() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 50);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        // Test peeking before reading
        let peeked_metadata = page_reader.peek_next_page().unwrap();
        if let Some(_metadata) = peeked_metadata {
            // Should be able to peek multiple times
            let peeked_again = page_reader.peek_next_page().unwrap();
            assert!(peeked_again.is_some());

            // Now read the actual page
            let page = page_reader.get_next_page().unwrap();
            assert!(page.is_some(), "Should get a page after peeking");
        }
    }

    #[test]
    fn test_skip_next_page() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 200);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        let mut read_pages = 0;
        let mut skipped_pages = 0;
        let mut total_operations = 0;

        while page_reader.peek_next_page().unwrap().is_some() {
            if total_operations % 2 == 0 {
                // Skip every other page
                page_reader.skip_next_page().unwrap();
                skipped_pages += 1;
            } else {
                // Read the page
                let page = page_reader.get_next_page().unwrap();
                assert!(page.is_some());
                read_pages += 1;
            }
            total_operations += 1;
        }

        assert!(skipped_pages > 0, "Should have skipped some pages");
        assert!(read_pages > 0, "Should have read some pages");
    }

    #[test]
    fn test_peek_next_page_offset() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        let mut seen_offsets = HashSet::new();

        while let Ok(Some(offset)) = page_reader.peek_next_page_offset() {
            // Each offset should be unique
            assert!(
                seen_offsets.insert(offset),
                "Page offset {offset} should be unique"
            );

            // Read the page to advance
            let page = page_reader.get_next_page().unwrap();
            assert!(page.is_some());
        }

        assert!(
            !seen_offsets.is_empty(),
            "Should have seen some page offsets"
        );
    }

    #[test]
    fn test_peek_offset_matches_actual_reading() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        while let Ok(Some(peeked_offset)) = page_reader.peek_next_page_offset() {
            // Verify that the offset from peeking matches the state
            match &page_reader.state {
                SerializedPageReaderState::Values {
                    offset,
                    next_page_header,
                    ..
                } => {
                    if next_page_header.is_some() {
                        assert_eq!(*offset, peeked_offset);
                    }
                }
                SerializedPageReaderState::Pages {
                    page_locations,
                    dictionary_page,
                    ..
                } => {
                    if let Some(dict_page) = dictionary_page {
                        assert_eq!(dict_page.offset as usize, peeked_offset);
                    } else if let Some(page) = page_locations.front() {
                        assert_eq!(page.offset as usize, peeked_offset);
                    }
                }
            }

            // Read the page and continue
            let page = page_reader.get_next_page().unwrap();
            if page.is_none() {
                break;
            }
        }
    }

    #[test]
    fn test_at_record_boundary() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 50);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        // Should not be at boundary when there are pages to read
        assert!(!page_reader.at_record_boundary().unwrap());

        // Read all pages
        while page_reader.get_next_page().unwrap().is_some() {}

        // Should be at boundary when no more pages
        assert!(page_reader.at_record_boundary().unwrap());
    }

    #[test]
    fn test_different_compression_codecs() {
        let compressions = vec![
            Compression::UNCOMPRESSED,
            Compression::SNAPPY,
            Compression::GZIP(Default::default()),
        ];

        for compression in compressions {
            let temp_file = create_test_parquet_data(compression, 1, 100);
            let file = temp_file.reopen().unwrap();
            let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

            let mut page_count = 0;
            while let Ok(Some(_page)) = page_reader.get_next_page() {
                page_count += 1;
            }

            assert!(
                page_count > 0,
                "Should read pages with compression {compression:?}"
            );
        }
    }

    #[test]
    fn test_multiple_row_groups() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 3, 50);
        let file = temp_file.reopen().unwrap();

        // First check how many row groups actually exist
        let reader = ParquetSerializedFileReader::new(file.try_clone().unwrap()).unwrap();
        let metadata = reader.metadata();
        let actual_row_groups = metadata.num_row_groups();

        assert!(actual_row_groups > 0, "Should have at least one row group");

        for row_group in 0..actual_row_groups {
            let mut page_reader = create_test_page_reader_from_file(&file, row_group, 0).unwrap();

            let mut page_count = 0;
            while let Ok(Some(_page)) = page_reader.get_next_page() {
                page_count += 1;
            }

            assert!(
                page_count > 0,
                "Should read pages from row group {row_group}"
            );
        }
    }

    #[test]
    fn test_string_column_with_nulls() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();
        // Test column 1 (string column with nulls)
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 1).unwrap();

        let mut found_data = false;

        while let Ok(Some(page)) = page_reader.get_next_page() {
            match page {
                Page::DictionaryPage { .. } => {}
                Page::DataPage { .. } | Page::DataPageV2 { .. } => {
                    found_data = true;
                }
            }
        }

        assert!(found_data, "Should find data pages");
    }

    #[test]
    fn test_page_reader_with_page_locations() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_with_page_locations(&file, 0, 0).unwrap();

        // Should be in Pages state
        match page_reader.state {
            SerializedPageReaderState::Pages { .. } => {
                // Expected
            }
            SerializedPageReaderState::Values { .. } => {
                panic!("Expected Pages state when page locations are provided");
            }
        }

        let mut page_count = 0;
        while let Ok(Some(_page)) = page_reader.get_next_page() {
            page_count += 1;
        }

        assert!(page_count > 0, "Should read pages in Pages state");
    }

    #[test]
    fn test_peek_and_skip_consistency() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 200);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        let mut peek_count = 0;
        let mut skip_count = 0;

        while let Ok(Some(_metadata)) = page_reader.peek_next_page() {
            peek_count += 1;
            page_reader.skip_next_page().unwrap();
            skip_count += 1;
        }

        assert_eq!(peek_count, skip_count);
        assert!(peek_count > 0, "Should have peeked and skipped some pages");

        // Should be at end now
        assert!(page_reader.peek_next_page().unwrap().is_none());
        assert!(page_reader.get_next_page().unwrap().is_none());
    }

    #[test]
    fn test_error_handling_invalid_file() {
        // Create an invalid "parquet" file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"not a parquet file").unwrap();
        temp_file.flush().unwrap();

        let file = temp_file.reopen().unwrap();
        let result = ParquetSerializedFileReader::new(file);
        assert!(result.is_err(), "Should fail to read invalid parquet file");
    }

    #[test]
    fn test_page_metadata_properties() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        while let Ok(Some(metadata)) = page_reader.peek_next_page() {
            // Check basic properties of page metadata
            if metadata.is_dict {
                // Dictionary page properties
                assert!(metadata.num_rows.is_none() || metadata.num_levels.is_none());
            } else {
                // Data page properties
                assert!(metadata.num_rows.is_some() || metadata.num_levels.is_some());
            }

            // Read the actual page
            let page = page_reader.get_next_page().unwrap();
            if page.is_none() {
                break;
            }
        }
    }

    #[test]
    fn test_large_file_handling() {
        // Create a larger file to test edge cases
        let temp_file = create_test_parquet_data(Compression::SNAPPY, 5, 1000);
        let file = temp_file.reopen().unwrap();

        // First check how many row groups actually exist
        let reader = ParquetSerializedFileReader::new(file.try_clone().unwrap()).unwrap();
        let metadata = reader.metadata();
        let actual_row_groups = metadata.num_row_groups();

        assert!(actual_row_groups > 0, "Should have at least one row group");

        for row_group in 0..actual_row_groups {
            let mut page_reader = create_test_page_reader_from_file(&file, row_group, 0).unwrap();

            let mut page_count = 0;
            let mut total_bytes = 0usize;

            while let Ok(Some(page)) = page_reader.get_next_page() {
                page_count += 1;
                match page {
                    Page::DataPage { buf, .. } | Page::DictionaryPage { buf, .. } => {
                        total_bytes += buf.len();
                    }
                    Page::DataPageV2 { buf, .. } => {
                        total_bytes += buf.len();
                    }
                }
            }

            assert!(
                page_count > 0,
                "Should read pages from large file row group {row_group}"
            );
            assert!(
                total_bytes > 0,
                "Should read some data from row group {row_group}"
            );
        }
    }

    #[test]
    fn test_state_transitions() {
        // Test with Values state (no page locations)
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();
        let mut values_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        match values_reader.state {
            SerializedPageReaderState::Values { .. } => {} // Expected
            _ => panic!("Expected Values state"),
        }

        // Test with Pages state (with page locations)
        let mut pages_reader = create_test_page_reader_with_page_locations(&file, 0, 0).unwrap();

        match pages_reader.state {
            SerializedPageReaderState::Pages { .. } => {} // Expected
            _ => panic!("Expected Pages state"),
        }

        // Both should be able to read pages
        assert!(values_reader.get_next_page().unwrap().is_some());
        assert!(pages_reader.get_next_page().unwrap().is_some());
    }

    // Test using the example parquet file if it exists
    #[test]
    fn test_with_real_parquet_file() {
        let test_file_path = "../../examples/nano_hits.parquet";
        if let Ok(file) = File::open(test_file_path) {
            let reader = ParquetSerializedFileReader::new(file.try_clone().unwrap()).unwrap();
            let metadata = reader.metadata();

            // Test with the first row group and first column
            if metadata.num_row_groups() > 0 {
                let row_group_metadata = metadata.row_group(0);
                if row_group_metadata.num_columns() > 0 {
                    let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

                    let mut page_count = 0;
                    while let Ok(Some(_page)) = page_reader.get_next_page() {
                        page_count += 1;
                        if page_count > 10 {
                            break; // Don't read too many pages in tests
                        }
                    }

                    assert!(page_count > 0, "Should read pages from real parquet file");
                }
            }
        }
        // If file doesn't exist, test passes (optional test)
    }

    #[test]
    fn test_concurrent_peek_operations() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        // Test multiple consecutive peeks
        let peek1 = page_reader.peek_next_page().unwrap();
        let peek2 = page_reader.peek_next_page().unwrap();
        let peek3 = page_reader.peek_next_page().unwrap();

        // All peeks should return the same result
        assert_eq!(peek1.is_some(), peek2.is_some());
        assert_eq!(peek2.is_some(), peek3.is_some());

        if let (Some(meta1), Some(meta2), Some(meta3)) = (peek1, peek2, peek3) {
            assert_eq!(meta1.is_dict, meta2.is_dict);
            assert_eq!(meta2.is_dict, meta3.is_dict);
        }
    }

    #[test]
    fn test_boundary_detection_accuracy() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 50);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        // Not at boundary initially
        assert!(!page_reader.at_record_boundary().unwrap());

        let mut pages_read = 0;
        while let Ok(Some(_page)) = page_reader.get_next_page() {
            pages_read += 1;
            // After reading some pages, boundary detection should still work
            if page_reader.peek_next_page().unwrap().is_some() {
                assert!(!page_reader.at_record_boundary().unwrap());
            }
        }

        assert!(pages_read > 0);
        // At boundary after reading all pages
        assert!(page_reader.at_record_boundary().unwrap());
    }

    // Additional comprehensive tests for liquid cache specific scenarios

    #[test]
    fn test_page_type_identification() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 1).unwrap(); // String column

        let mut dictionary_pages = 0;
        let mut data_pages_v1 = 0;
        let mut data_pages_v2 = 0;

        while let Ok(Some(page)) = page_reader.get_next_page() {
            match page {
                Page::DictionaryPage { .. } => {
                    dictionary_pages += 1;
                }
                Page::DataPage { .. } => {
                    data_pages_v1 += 1;
                }
                Page::DataPageV2 { .. } => {
                    data_pages_v2 += 1;
                }
            }
        }

        // Should have at least some data pages
        assert!(data_pages_v1 + data_pages_v2 > 0, "Should have data pages");

        // Log the page type distribution for debugging
        println!(
            "Page distribution - Dict: {dictionary_pages}, V1: {data_pages_v1}, V2: {data_pages_v2}"
        );
    }

    #[test]
    fn test_page_buffer_content_validation() {
        let temp_file = create_test_parquet_data(Compression::SNAPPY, 1, 100);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        while let Ok(Some(page)) = page_reader.get_next_page() {
            match page {
                Page::DataPage {
                    buf, num_values, ..
                } => {
                    assert!(!buf.is_empty(), "Data page buffer should not be empty");
                    assert!(num_values > 0, "Data page should have values");
                }
                Page::DataPageV2 {
                    buf,
                    num_values,
                    num_rows,
                    ..
                } => {
                    assert!(!buf.is_empty(), "Data page v2 buffer should not be empty");
                    assert!(num_values > 0, "Data page v2 should have values");
                    assert!(num_rows > 0, "Data page v2 should have rows");
                }
                Page::DictionaryPage { .. } => {}
            }
        }
    }

    #[test]
    fn test_interleaved_peek_and_read_operations() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 150);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        let mut operation_count = 0;

        while page_reader.peek_next_page().unwrap().is_some() {
            operation_count += 1;

            match operation_count % 3 {
                0 => {
                    // Just peek, don't read
                    let _metadata = page_reader.peek_next_page().unwrap();
                }
                1 => {
                    // Peek then read
                    let _metadata = page_reader.peek_next_page().unwrap();
                    let page = page_reader.get_next_page().unwrap();
                    assert!(page.is_some());
                }
                2 => {
                    // Skip without peeking
                    page_reader.skip_next_page().unwrap();
                }
                _ => unreachable!(),
            }

            // Safety check to avoid infinite loops
            if operation_count > 1000 {
                break;
            }
        }

        assert!(operation_count > 0, "Should have performed operations");
    }

    #[test]
    fn test_page_offset_consistency_across_calls() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        while let Ok(Some(offset1)) = page_reader.peek_next_page_offset() {
            // Multiple calls should return the same offset
            let offset2 = page_reader.peek_next_page_offset().unwrap();
            let offset3 = page_reader.peek_next_page_offset().unwrap();

            assert_eq!(offset1, offset2.unwrap());
            assert_eq!(offset2, offset3);

            // Advance to next page
            let page = page_reader.get_next_page().unwrap();
            if page.is_none() {
                break;
            }
        }
    }

    #[test]
    fn test_large_page_handling() {
        // Test with larger row groups to potentially get larger pages
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 10000);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        let mut max_page_size = 0;
        let mut total_pages = 0;

        while let Ok(Some(page)) = page_reader.get_next_page() {
            total_pages += 1;
            let page_size = match page {
                Page::DataPage { buf, .. } | Page::DictionaryPage { buf, .. } => buf.len(),
                Page::DataPageV2 { buf, .. } => buf.len(),
            };
            max_page_size = max_page_size.max(page_size);
        }

        assert!(total_pages > 0, "Should have read pages");
        assert!(max_page_size > 0, "Should have non-zero page sizes");
        println!("Max page size: {max_page_size} bytes, Total pages: {total_pages}");
    }

    #[test]
    fn test_empty_page_handling() {
        // Create a minimal parquet file
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 1);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        // Even minimal files should have at least one page
        let first_page = page_reader.get_next_page().unwrap();
        assert!(first_page.is_some(), "Even minimal files should have pages");
    }

    #[test]
    fn test_repeated_peek_offset_calls() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        if let Ok(Some(first_offset)) = page_reader.peek_next_page_offset() {
            // Call peek_next_page_offset multiple times
            for _ in 0..10 {
                let offset = page_reader.peek_next_page_offset().unwrap();
                assert_eq!(
                    Some(first_offset),
                    offset,
                    "Repeated calls should return same offset"
                );
            }
        }
    }

    #[test]
    fn test_state_specific_behavior() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();

        // Test Values state behavior
        let values_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();
        if let SerializedPageReaderState::Values {
            offset,
            remaining_bytes,
            ..
        } = &values_reader.state
        {
            assert!(*offset > 0, "Values state should have non-zero offset");
            assert!(
                *remaining_bytes > 0,
                "Values state should have remaining bytes"
            );
        }

        // Test Pages state behavior
        let pages_reader = create_test_page_reader_with_page_locations(&file, 0, 0).unwrap();
        if let SerializedPageReaderState::Pages {
            page_locations,
            total_rows,
            ..
        } = &pages_reader.state
        {
            assert!(
                !page_locations.is_empty(),
                "Pages state should have page locations"
            );
            assert!(*total_rows > 0, "Pages state should have total rows");
        }
    }

    #[test]
    fn test_metadata_consistency() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();
        let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

        while let Ok(Some(metadata)) = page_reader.peek_next_page() {
            let page = page_reader.get_next_page().unwrap();

            if let Some(page) = page {
                // Verify metadata consistency with actual page
                match (metadata.is_dict, &page) {
                    (true, Page::DictionaryPage { .. }) => {
                        // Consistent
                    }
                    (false, Page::DataPage { .. }) | (false, Page::DataPageV2 { .. }) => {
                        // Consistent
                    }
                    _ => {
                        panic!("Metadata dictionary flag doesn't match page type");
                    }
                }
            } else {
                break;
            }
        }
    }

    #[test]
    fn test_compression_specific_edge_cases() {
        let compressions = vec![Compression::UNCOMPRESSED, Compression::SNAPPY];

        for compression in compressions {
            // Test with very small data to check compression edge cases
            let temp_file = create_test_parquet_data(compression, 1, 5);
            let file = temp_file.reopen().unwrap();
            let mut page_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();

            let mut pages_found = false;
            while let Ok(Some(_page)) = page_reader.get_next_page() {
                pages_found = true;
            }

            assert!(
                pages_found,
                "Should find pages even with small compressed data"
            );
        }
    }

    #[test]
    fn test_skip_behavior_with_different_states() {
        let temp_file = create_test_parquet_data(Compression::UNCOMPRESSED, 1, 100);
        let file = temp_file.reopen().unwrap();

        // Test skip in Values state
        let mut values_reader = create_test_page_reader_from_file(&file, 0, 0).unwrap();
        if values_reader.peek_next_page().unwrap().is_some() {
            values_reader.skip_next_page().unwrap();
            // Should still be able to continue
            let _ = values_reader.peek_next_page();
        }

        // Test skip in Pages state
        let mut pages_reader = create_test_page_reader_with_page_locations(&file, 0, 0).unwrap();
        if pages_reader.peek_next_page().unwrap().is_some() {
            pages_reader.skip_next_page().unwrap();
            // Should still be able to continue
            let _ = pages_reader.peek_next_page();
        }
    }
}
