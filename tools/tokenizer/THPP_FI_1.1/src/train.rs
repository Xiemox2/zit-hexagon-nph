//! API para entrenar modelos sobre cintas THPP:
//! - `classify` / `factorize`: (tipo, idioma, cubo SS, slot BBB) de cualquier id.
//! - `DenseVocab`: ids dispersos (hasta ~3.1e8) → índices contiguos 0..N y vuelta.
use crate::*;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Kind {
    /// SEP, NL, TAB, PAD, SPACE, TITLE, UPPER, MIXED, TITLE_L, UPPER_L, FORM_*
    Control = 0,
    Lang = 1,
    Version = 2,
    /// CCC corto (dígitos, puntuación) < 1000
    Char = 3,
    /// letra LL*10000+CCC
    Letter = 4,
    Case = 5,
    CodePoint = 6,
    Byte = 7,
    Lexeme = 8,
    Unknown = 9,
}

pub fn classify(t: u32) -> Kind {
    match t {
        SEP | NL | TAB | PAD | SPACE | TITLE | UPPER | MIXED | TITLE_L | UPPER_L | FORM_NFC | FORM_NFD => Kind::Control,
        1201..=1299 => Kind::Lang,
        VER_BASE..=1399 => Kind::Version,
        1..=999 => Kind::Char,
        CASE_BASE..=CASE_END => Kind::Case,
        CP_BASE..=CP_END => Kind::CodePoint,
        BYTE_BASE..=BYTE_END => Kind::Byte,
        LETTER_MIN..=319_999 => Kind::Letter,
        _ if lex_parts(t).is_some() => Kind::Lexeme,
        _ => Kind::Unknown,
    }
}

/// Descomposición factorizada de un id. Para lexemas: lang=LL, bucket=SS-10, slot=BBB
/// (y `index` = (SS-10)*1000+BBB, posición en la lista). Para el resto ver `factorize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Factors {
    pub kind: Kind,
    pub lang: u32,
    pub bucket: u32,
    pub slot: u32,
}

impl Factors {
    pub fn as_array(&self) -> [u32; 4] {
        [self.kind as u32, self.lang, self.bucket, self.slot]
    }
}

/// - Lexeme: (LL, SS-10, BBB)
/// - Letter: (LL, 0, CCC)
/// - Char: (0, 0, CCC)
/// - Lang: (LL, 0, 0); Version: (0, 0, v)
/// - Control: (0, 0, id)
/// - CodePoint: (0, cp >> 10, cp & 1023); Case: (0, m >> 10, m & 1023); Byte: (0, 0, b)
pub fn factorize(t: u32) -> Factors {
    let k = classify(t);
    let (lang, bucket, slot) = match k {
        Kind::Lexeme => {
            let (ll, ss, bbb) = lex_parts(t).unwrap();
            (ll, ss - SS0, bbb)
        }
        Kind::Letter => (t / 10_000, 0, t % 10_000),
        Kind::Char => (0, 0, t),
        Kind::Lang => (t - LANG_BASE, 0, 0),
        Kind::Version => (0, 0, t - VER_BASE),
        Kind::Control => (0, 0, t),
        Kind::CodePoint => {
            let cp = t - CP_BASE;
            (0, cp >> 10, cp & 1023)
        }
        Kind::Case => {
            let m = t - CASE_BASE;
            (0, m >> 10, m & 1023)
        }
        Kind::Byte => (0, 0, t - BYTE_BASE),
        Kind::Unknown => (0, 0, 0),
    };
    Factors { kind: k, lang, bucket, slot }
}

/// Ids que siempre van primero en el vocab denso (índices estables entre corpus).
pub fn reserved_ids() -> Vec<u32> {
    let mut v = vec![PAD, SEP, NL, TAB, SPACE, TITLE, UPPER, MIXED, TITLE_L, UPPER_L, FORM_NFC, FORM_NFD, VER_1_1];
    v.extend(LANGS.iter().map(|l| l.lang_id));
    v
}

/// Vocabulario denso: índice 0 = PAD, 1 = UNK (ids no vistos), luego reservados, luego por frecuencia.
#[derive(Clone, Debug, Default)]
pub struct DenseVocab {
    ids: Vec<u32>,
    map: HashMap<u32, u32>,
}

pub const DENSE_PAD: u32 = 0;
pub const DENSE_UNK: u32 = 1;
/// id "virtual" para UNK en la tabla densa (no aparece en cintas).
pub const UNK_ID: u32 = u32::MAX;

pub fn count_ids(counts: &mut HashMap<u32, u64>, ids: &[u32]) {
    for &t in ids {
        *counts.entry(t).or_insert(0) += 1;
    }
}

impl DenseVocab {
    /// `min_count`: frecuencia mínima; `max_size`: tope de filas (incluye reservados).
    pub fn from_counts(counts: &HashMap<u32, u64>, min_count: u64, max_size: Option<usize>) -> Self {
        let mut ids = vec![PAD, UNK_ID];
        for r in reserved_ids() {
            if r != PAD {
                ids.push(r);
            }
        }
        let mut rest: Vec<(u32, u64)> =
            counts.iter().filter(|(id, c)| **c >= min_count && !ids.contains(id)).map(|(a, b)| (*a, *b)).collect();
        rest.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (id, _) in rest {
            if let Some(m) = max_size {
                if ids.len() >= m {
                    break;
                }
            }
            ids.push(id);
        }
        Self::from_ids(ids)
    }

    pub fn from_ids(ids: Vec<u32>) -> Self {
        let map = ids.iter().enumerate().map(|(i, &id)| (id, i as u32)).collect();
        Self { ids, map }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn to_dense(&self, id: u32) -> u32 {
        *self.map.get(&id).unwrap_or(&DENSE_UNK)
    }
    pub fn to_id(&self, dense: u32) -> u32 {
        match self.ids.get(dense as usize) {
            Some(&UNK_ID) | None => PAD,
            Some(&id) => id,
        }
    }
    pub fn encode_dense(&self, ids: &[u32]) -> Vec<u32> {
        ids.iter().map(|&t| self.to_dense(t)).collect()
    }
    pub fn decode_dense(&self, dense: &[u32]) -> Vec<u32> {
        dense.iter().map(|&d| self.to_id(d)).collect()
    }

    /// Texto: `THPP-VOCAB 1 <n>` y luego un id por línea (línea k = índice denso k).
    pub fn save(&self, w: &mut impl Write) -> io::Result<()> {
        writeln!(w, "THPP-VOCAB 1 {} lists={:016x}", self.ids.len(), lists_checksum())?;
        for &id in &self.ids {
            writeln!(w, "{id}")?;
        }
        Ok(())
    }
    pub fn load(r: impl BufRead) -> io::Result<Self> {
        let mut lines = r.lines();
        let head = lines.next().transpose()?.unwrap_or_default();
        if !head.starts_with("THPP-VOCAB 1") {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "no es THPP-VOCAB"));
        }
        let mut ids = Vec::new();
        for l in lines {
            let l = l?;
            if !l.trim().is_empty() {
                ids.push(l.trim().parse::<u32>().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?);
            }
        }
        Ok(Self::from_ids(ids))
    }
    pub fn save_file(&self, path: &str) -> io::Result<()> {
        let mut w = io::BufWriter::new(std::fs::File::create(path)?);
        self.save(&mut w)?;
        w.flush()
    }
    pub fn load_file(path: &str) -> io::Result<Self> {
        Self::load(io::BufReader::new(std::fs::File::open(path)?))
    }
}
