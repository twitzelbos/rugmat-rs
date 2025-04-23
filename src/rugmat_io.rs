// rugmat-io.rs: file I/O and checksum for RugMat
use crate::RugMat;
use rug::{Float, float::Round};
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
        writer.write_all(&(self.data[0].precision() as u32).to_le_bytes())?;

        let mut hasher = blake3::Hasher::new();

        for f in &self.data {
            let bytes = f.to_digits::<u8>(Round::Nearest).1;
            writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
            writer.write_all(&bytes)?;
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(&bytes);
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
        let mut buf4 = [0u8; 4];

        reader.read_exact(&mut buf8)?;
        let rows = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?;
        let cols = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf4)?;
        let precision = u32::from_le_bytes(buf4);

        let mut data = Vec::with_capacity(rows * cols);
        let mut hasher = blake3::Hasher::new();

        for _ in 0..(rows * cols) {
            reader.read_exact(&mut buf8)?;
            let len = u64::from_le_bytes(buf8) as usize;
            let mut bytes = vec![0u8; len];
            reader.read_exact(&mut bytes)?;
            hasher.update(&buf8);
            hasher.update(&bytes);
            let f = Float::with_val(precision, rug::float::Parse::from_digits(&bytes, 256));
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
