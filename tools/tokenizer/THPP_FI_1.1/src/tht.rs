//! Formato binario de cinta `.tht` y escritura/lectura `.npy` (u32).
//!
//! Cabecera .tht (32 bytes, little endian):
//! ```text
//! 0  magic        b"THPT"
//! 4  format       u16   (1)
//! 6  spec         u16   (11 = cinta 1.1)
//! 8  flags        u32   (Config::flags; bit 5 = varios documentos)
//! 12 lang_ll      u32   (0 = varios idiomas)
//! 16 list_sum     u64   (lists_checksum())
//! 24 count        u64   (número de ids)
//! 32 cuerpo       count × u32 LE
//! ```
use std::io::{self, Read, Write};

pub const MAGIC: &[u8; 4] = b"THPT";
pub const FORMAT: u16 = 1;
pub const SPEC: u16 = 11;
pub const HEADER_LEN: usize = 32;
pub const FLAG_MULTI_DOC: u32 = 1 << 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub format: u16,
    pub spec: u16,
    pub flags: u32,
    pub lang_ll: u32,
    pub list_checksum: u64,
    pub count: u64,
}

impl Header {
    pub fn new(flags: u32, lang_ll: u32, count: u64) -> Self {
        Self { format: FORMAT, spec: SPEC, flags, lang_ll, list_checksum: crate::lists_checksum(), count }
    }
    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0..4].copy_from_slice(MAGIC);
        b[4..6].copy_from_slice(&self.format.to_le_bytes());
        b[6..8].copy_from_slice(&self.spec.to_le_bytes());
        b[8..12].copy_from_slice(&self.flags.to_le_bytes());
        b[12..16].copy_from_slice(&self.lang_ll.to_le_bytes());
        b[16..24].copy_from_slice(&self.list_checksum.to_le_bytes());
        b[24..32].copy_from_slice(&self.count.to_le_bytes());
        b
    }
    pub fn from_bytes(b: &[u8]) -> io::Result<Self> {
        if b.len() < HEADER_LEN || &b[0..4] != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "no es .tht (magic THPT)"));
        }
        Ok(Self {
            format: u16::from_le_bytes(b[4..6].try_into().unwrap()),
            spec: u16::from_le_bytes(b[6..8].try_into().unwrap()),
            flags: u32::from_le_bytes(b[8..12].try_into().unwrap()),
            lang_ll: u32::from_le_bytes(b[12..16].try_into().unwrap()),
            list_checksum: u64::from_le_bytes(b[16..24].try_into().unwrap()),
            count: u64::from_le_bytes(b[24..32].try_into().unwrap()),
        })
    }
}

pub fn is_tht(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[0..4] == MAGIC
}

pub fn is_npy(bytes: &[u8]) -> bool {
    bytes.len() >= 6 && &bytes[0..6] == b"\x93NUMPY"
}

fn write_u32s(w: &mut impl Write, ids: &[u32]) -> io::Result<()> {
    if cfg!(target_endian = "little") {
        // SAFETY: u32 sin padding; en little endian los bytes ya son LE.
        let b = unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const u8, ids.len() * 4) };
        w.write_all(b)
    } else {
        for &x in ids {
            w.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    }
}

fn read_u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn write_tht(w: &mut impl Write, header: &Header, ids: &[u32]) -> io::Result<()> {
    let mut h = *header;
    h.count = ids.len() as u64;
    w.write_all(&h.to_bytes())?;
    write_u32s(w, ids)
}

pub fn read_tht(bytes: &[u8]) -> io::Result<(Header, Vec<u32>)> {
    let h = Header::from_bytes(bytes)?;
    let body = &bytes[HEADER_LEN..];
    let n = h.count as usize;
    if body.len() < n * 4 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, ".tht truncado"));
    }
    Ok((h, read_u32s(&body[..n * 4])))
}

pub fn read_tht_file(path: &str) -> io::Result<(Header, Vec<u32>)> {
    let mut b = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut b)?;
    read_tht(&b)
}

pub fn write_tht_file(path: &str, header: &Header, ids: &[u32]) -> io::Result<()> {
    let mut w = io::BufWriter::new(std::fs::File::create(path)?);
    write_tht(&mut w, header, ids)?;
    w.flush()
}

/// .npy v1.0, dtype `<u4`, forma `shape` (C order). numpy.load lo lee directo.
pub fn write_npy_u32(w: &mut impl Write, data: &[u32], shape: &[usize]) -> io::Result<()> {
    let shape_s = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut hdr = format!("{{'descr': '<u4', 'fortran_order': False, 'shape': {shape_s}, }}");
    let total = 10 + hdr.len() + 1;
    let pad = (64 - total % 64) % 64;
    hdr.push_str(&" ".repeat(pad));
    hdr.push('\n');
    w.write_all(b"\x93NUMPY\x01\x00")?;
    w.write_all(&(hdr.len() as u16).to_le_bytes())?;
    w.write_all(hdr.as_bytes())?;
    write_u32s(w, data)
}

pub fn write_npy_u32_file(path: &str, data: &[u32], shape: &[usize]) -> io::Result<()> {
    let mut w = io::BufWriter::new(std::fs::File::create(path)?);
    write_npy_u32(&mut w, data, shape)?;
    w.flush()
}

/// Lee un .npy `<u4` (cualquier forma, se devuelve plano).
pub fn read_npy_u32(bytes: &[u8]) -> io::Result<Vec<u32>> {
    if !is_npy(bytes) || bytes.len() < 10 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "no es .npy"));
    }
    let (hlen, start) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12)
    };
    let hdr = std::str::from_utf8(&bytes[start..start + hlen]).unwrap_or("");
    if !hdr.contains("<u4") {
        return Err(io::Error::new(io::ErrorKind::InvalidData, ".npy no es <u4"));
    }
    Ok(read_u32s(&bytes[start + hlen..]))
}
