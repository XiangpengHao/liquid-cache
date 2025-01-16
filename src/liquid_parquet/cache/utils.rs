use std::{
    io::{Read, Seek},
    ops::Range,
};

/// A reader that limits reading to a specific range while maintaining seek capability
pub(crate) struct RangedFile<R: Read + Seek> {
    inner: R,
    range: Range<u64>,
    pos: u64,
}

impl<R: Read + Seek> RangedFile<R> {
    pub(crate) fn new(mut inner: R, range: Range<u64>) -> std::io::Result<Self> {
        inner.seek(std::io::SeekFrom::Start(range.start))?;
        Ok(Self {
            inner,
            pos: range.start,
            range,
        })
    }
}

impl<R: Read + Seek> Read for RangedFile<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.range.end {
            return Ok(0);
        }
        let to_read = std::cmp::min(buf.len() as u64, self.range.end - self.pos) as usize;
        let bytes_read = self.inner.read(&mut buf[..to_read])?;
        self.pos += bytes_read as u64;
        Ok(bytes_read)
    }
}

impl<R: Read + Seek> Seek for RangedFile<R> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let new_pos = match pos {
            std::io::SeekFrom::Start(n) => self.range.start + n,
            std::io::SeekFrom::End(n) => (self.range.end as i64 + n) as u64,
            std::io::SeekFrom::Current(n) => (self.pos as i64 + n) as u64,
        };

        if new_pos > self.range.end || new_pos < self.range.start {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Seek beyond range boundaries",
            ));
        }

        self.inner.seek(std::io::SeekFrom::Start(new_pos))?;
        self.pos = new_pos;
        Ok(new_pos - self.range.start)
    }
}
