// rugmat-io.rs: file I/O and checksum for RugMat
use crate::RugMat;
use crate::float_serializer::{read_float, write_float};
use rug::Float;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

const RUGMAT_MAGIC: &[u8; 6] = b"RUGMAT";
const RUGMAT_VERSION: u8 = 1;

impl RugMat {
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(RUGMAT_MAGIC)?;
        writer.write_all(&[RUGMAT_VERSION])?;
        writer.write_all(&(self.rows as u64).to_le_bytes())?;
        writer.write_all(&(self.cols as u64).to_le_bytes())?;

        let mut hasher = blake3::Hasher::new();
        let mut temp_buf = Vec::new();

        for f in &self.data {
            temp_buf.clear();
            write_float(&mut temp_buf, f)?;
            hasher.update(&temp_buf);
            writer.write_all(&(temp_buf.len() as u64).to_le_bytes())?;
            writer.write_all(&temp_buf)?;
        }

        let checksum = hasher.finalize();
        writer.write_all(checksum.as_bytes())?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic)?;
        if &magic != RUGMAT_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Bad magic header",
            ));
        }

        let mut version = [0u8; 1];
        reader.read_exact(&mut version)?;
        if version[0] != RUGMAT_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Unsupported version",
            ));
        }

        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let rows = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?;
        let cols = u64::from_le_bytes(buf8) as usize;

        let mut data = Vec::with_capacity(rows * cols);
        let mut hasher = blake3::Hasher::new();

        for _ in 0..(rows * cols) {
            reader.read_exact(&mut buf8)?;
            let len = u64::from_le_bytes(buf8) as usize;
            let mut temp_buf = vec![0u8; len];
            reader.read_exact(&mut temp_buf)?;
            hasher.update(&temp_buf);
            let f = read_float(&mut &temp_buf[..])?;
            data.push(f);
        }

        let mut checksum_buf = [0u8; 32];
        reader.read_exact(&mut checksum_buf)?;
        let checksum_expected = blake3::Hash::from(checksum_buf);
        let checksum_actual = hasher.finalize();
        if checksum_actual != checksum_expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Checksum mismatch",
            ));
        }

        Ok(RugMat { data, rows, cols })
    }
}
