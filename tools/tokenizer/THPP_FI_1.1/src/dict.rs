//! Diccionario de un idioma: tabla hash abierta sobre un arena (como 1.0) + rev por índice.
use crate::{chars, is_cjk, lex_index, pack_lex, PAD, SS0};
use unicode_normalization::{is_nfc_quick, IsNormalized, UnicodeNormalization};

pub struct Dict {
    pub ll: u32,
    pub lang_id: u32,
    arena: Vec<u8>,
    off: Vec<u32>,
    len: Vec<u32>,
    id: Vec<u32>,
    n_primary: usize,
    slot: Vec<u32>,
    mask: usize,
    pub ascii: [u32; 128],
    /// Largo (en glifos) de la entrada CJK/Thai más larga.
    pub max_cjk: usize,
}

fn mix(h: u64, b: u8) -> u64 {
    h.wrapping_mul(0x517cc1b727220a95) ^ b as u64
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0x517cc1b727220a95u64;
    for &b in bytes {
        h = mix(h, b);
    }
    h
}

impl Dict {
    /// Mismo recorrido que `Map::build` de 1.0: split_whitespace, minúsculas, saltar repetidas,
    /// `tid = pack_lex(LL, i % 1000, 10 + i/1000)`. Luego alias NFC (mismo id) para entradas no NFC.
    pub fn load(list: &str, ll: u32, lang_id: u32) -> Self {
        let nwords = list.split_whitespace().count();
        let mut nslots = 16usize;
        while nslots < nwords * 3 + 32 {
            nslots <<= 1;
        }
        let mut d = Self {
            ll,
            lang_id,
            arena: Vec::with_capacity(list.len()),
            off: Vec::with_capacity(nwords),
            len: Vec::with_capacity(nwords),
            id: Vec::with_capacity(nwords),
            n_primary: 0,
            slot: vec![0; nslots],
            mask: nslots - 1,
            ascii: [PAD; 128],
            max_cjk: 1,
        };
        let mut i = 0u32;
        let mut key = String::new();
        for w in list.split_whitespace() {
            key.clear();
            for c in w.chars() {
                key.extend(c.to_lowercase());
            }
            if key.is_empty() || d.get(key.as_bytes()).is_some() {
                continue;
            }
            let tid = pack_lex(ll, i % 1000, SS0 + i / 1000);
            d.insert(key.as_bytes(), tid);
            if key.chars().all(is_cjk) {
                d.max_cjk = d.max_cjk.max(key.chars().count());
            }
            i += 1;
        }
        d.n_primary = d.off.len();
        // alias NFC → mismo id (vi trae entradas en NFD; texto normal llega en NFC)
        for k in 0..d.n_primary {
            let w = d.entry(k);
            if w.is_ascii() || is_nfc_quick(w.chars()) == IsNormalized::Yes {
                continue;
            }
            let n: String = w.nfc().collect();
            if n != w && d.get(n.as_bytes()).is_none() {
                let tid = d.id[k];
                d.insert(n.as_bytes(), tid);
            }
        }
        for b in 0..128u8 {
            if let Some(id) = chars::enc(b as char) {
                d.ascii[b as usize] = id;
            }
        }
        d
    }

    fn insert(&mut self, key: &[u8], tid: u32) {
        let off = self.arena.len() as u32;
        self.arena.extend_from_slice(key);
        let idx = self.off.len() as u32;
        self.off.push(off);
        self.len.push(key.len() as u32);
        self.id.push(tid);
        let mut p = hash_bytes(key) as usize & self.mask;
        while self.slot[p] != 0 {
            p = (p + 1) & self.mask;
        }
        self.slot[p] = idx + 1;
    }

    fn entry(&self, k: usize) -> &str {
        let o = self.off[k] as usize;
        let n = self.len[k] as usize;
        // SAFETY: el arena solo recibe bytes de &str completos.
        unsafe { std::str::from_utf8_unchecked(&self.arena[o..o + n]) }
    }

    pub fn n_primary(&self) -> usize {
        self.n_primary
    }

    /// Palabra i (orden del txt), para decode.
    pub fn word(&self, i: usize) -> Option<&str> {
        if i < self.n_primary {
            Some(self.entry(i))
        } else {
            None
        }
    }

    pub fn word_by_id(&self, tid: u32) -> Option<&str> {
        let (ll, i) = lex_index(tid)?;
        if ll != self.ll {
            return None;
        }
        self.word(i)
    }

    pub fn get(&self, key: &[u8]) -> Option<u32> {
        let mut p = hash_bytes(key) as usize & self.mask;
        loop {
            let s = self.slot[p];
            if s == 0 {
                return None;
            }
            let i = (s - 1) as usize;
            let o = self.off[i] as usize;
            let n = self.len[i] as usize;
            if &self.arena[o..o + n] == key {
                return Some(self.id[i]);
            }
            p = (p + 1) & self.mask;
        }
    }

    pub fn get_ascii_lower(&self, raw: &[u8]) -> Option<u32> {
        let mut h = 0x517cc1b727220a95u64;
        for &b in raw {
            h = mix(h, b.to_ascii_lowercase());
        }
        let mut p = h as usize & self.mask;
        loop {
            let s = self.slot[p];
            if s == 0 {
                return None;
            }
            let i = (s - 1) as usize;
            let o = self.off[i] as usize;
            let n = self.len[i] as usize;
            if n == raw.len() && raw.iter().zip(&self.arena[o..o + n]).all(|(&a, &b)| a.to_ascii_lowercase() == b) {
                return Some(self.id[i]);
            }
            p = (p + 1) & self.mask;
        }
    }
}
