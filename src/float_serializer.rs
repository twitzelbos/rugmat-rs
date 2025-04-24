use gmp_mpfr_sys::mpfr;
use rug::Float;
use std::io::Cursor;
use std::io::{Read, Write};
use std::slice;

/// Serialize a Float by accessing internal MPFR data.
pub fn write_float<W: Write>(writer: &mut W, f: &Float) -> std::io::Result<()> {
    let raw = unsafe { &*f.as_raw() }; // raw: *const __mpfr_struct
    let prec = raw.prec as u32;
    let exp = raw.exp as i64;
    let sign = if f.is_sign_positive() { 1i8 } else { -1i8 };

    // Each GMP limb is u64 (assumed here)
    let n_limbs = ((prec + 63) / 64) as usize;
    let limbs = unsafe { slice::from_raw_parts(raw.d.as_ptr(), n_limbs) };

    writer.write_all(&prec.to_le_bytes())?;
    writer.write_all(&sign.to_le_bytes())?;
    writer.write_all(&exp.to_le_bytes())?;
    writer.write_all(&(n_limbs as u64).to_le_bytes())?;

    for &limb in limbs {
        writer.write_all(&limb.to_le_bytes())?;
    }

    Ok(())
}

/// Deserialize a Float by restoring precision, sign, exponent, and limb data.
pub fn read_float<R: Read>(reader: &mut R) -> std::io::Result<Float> {
    let mut buf4 = [0u8; 4];
    let mut buf1 = [0u8; 1];
    let mut buf8 = [0u8; 8];

    reader.read_exact(&mut buf4)?;
    let prec = u32::from_le_bytes(buf4);

    reader.read_exact(&mut buf1)?;
    let sign = i8::from_le_bytes(buf1);

    reader.read_exact(&mut buf8)?;
    let exp = i64::from_le_bytes(buf8);

    reader.read_exact(&mut buf8)?;
    let n_limbs = u64::from_le_bytes(buf8) as usize;

    let mut limbs = vec![0u64; n_limbs];
    for i in 0..n_limbs {
        reader.read_exact(&mut buf8)?;
        limbs[i] = u64::from_le_bytes(buf8);
    }

    // Allocate a new Float and replace its internal data
    let mut f = Float::new(prec);
    let raw = unsafe { &mut *f.as_raw_mut() };

    raw.prec = prec as mpfr::prec_t;
    raw.exp = exp;

    unsafe {
        std::ptr::copy_nonoverlapping(limbs.as_ptr(), raw.d.as_ptr(), n_limbs);
    }

    if sign < 0 {
        f = -f;
    }

    Ok(f)
}

#[test]
fn round_trip_float_precision() {
    let original = Float::with_val(
        256,
        Float::parse("3.14159265358979323846264338327950288419716939937510").unwrap(),
    );

    let mut buf = Vec::new();
    write_float(&mut buf, &original).expect("write failed");

    let mut cursor = Cursor::new(buf);
    let restored = read_float(&mut cursor).expect("read failed");

    assert_eq!(original.prec(), restored.prec(), "Precision mismatch");
    assert!(
        original == restored,
        "Float value mismatch\noriginal: {}\nrestored: {}",
        original,
        restored
    );
}
