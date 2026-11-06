//! THPP 1.0. Solo es/en. IDs de cinta sin cambios respecto a 0.9.
//! SEP 0, NL 11, TAB 110, PAD 111, SPACE 1001, TITLE 1101, UPPER 1102.
//! LANG es 1201, en 1202. Lexema = LL*10_000_000 + 1_000_000 + SS*1_000 + BBB.

mod chars;

use std::collections::HashMap;

pub const SPACE: u32 = 1001;
pub const SEP: u32 = 0;
pub const PAD: u32 = 111;
pub const NL: u32 = 11;
pub const TAB: u32 = 110;
pub const TITLE: u32 = 1101;
pub const UPPER: u32 = 1102;
pub const FF: u32 = 10;
pub const SS0: u32 = 10;

#[derive(Clone, Copy)]
pub struct Lang {
    pub code: &'static str,
    pub ll: u32,
    pub lang_id: u32,
}

pub const LANGS: &[Lang] = &[
    Lang { code: "es", ll: 1, lang_id: 1201 },
    Lang { code: "en", ll: 2, lang_id: 1202 },
];

pub fn pack_lex(ll: u32, bbb: u32, ss: u32) -> u32 {
    ll * 10_000_000 + FF * 100_000 + ss * 1_000 + bbb
}

pub fn pack_letter(ll: u32, ch: char) -> u32 {
    match chars::enc(ch) {
        Some(core) => ll * 10_000 + core,
        None => PAD,
    }
}

pub fn unpack_letter(t: u32) -> Option<char> {
    if t < 10_000 || t >= 10_000_000 {
        return None;
    }
    chars::dec_core(t % 10_000)
}

struct Map {
    arena: Vec<u8>,
    off: Vec<u32>,
    len: Vec<u32>,
    id: Vec<u32>,
    slot: Vec<u32>,
    mask: usize,
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

impl Map {
    // FASE 3 — aquí se expande el diccionario directo.
    // Una línea de data/hppt_*.txt -> una clave en minúsculas -> un id pack_lex.
    // El arena guarda los bytes; slot[] es la tabla abierta. No vuelve a crecer en encode.
    fn build(list: &str, ll: u32) -> Self {
        let mut nslots = 16usize;
        let approx = list.len() / 6;
        while nslots < approx * 2 + 32 {
            nslots <<= 1;
        }
        let mut m = Self {
            arena: Vec::with_capacity(list.len()),
            off: Vec::with_capacity(approx),
            len: Vec::with_capacity(approx),
            id: Vec::with_capacity(approx),
            slot: vec![0; nslots],
            mask: nslots - 1,
        };
        let mut i = 0u32;
        for w in list.split_whitespace() {
            if w.is_empty() {
                continue;
            }
            let mut key = String::with_capacity(w.len());
            for c in w.chars() {
                for l in c.to_lowercase() {
                    key.push(l);
                }
            }
            if key.is_empty() || m.get(key.as_bytes()).is_some() {
                continue;
            }
            let tid = pack_lex(ll, i % 1000, SS0 + i / 1000);
            let off = m.arena.len() as u32;
            m.arena.extend(key.as_bytes());
            let idx = m.off.len() as u32;
            m.off.push(off);
            m.len.push(key.len() as u32);
            m.id.push(tid);
            let mut p = hash_bytes(key.as_bytes()) as usize & m.mask;
            loop {
                if m.slot[p] == 0 {
                    m.slot[p] = idx + 1;
                    break;
                }
                p = (p + 1) & m.mask;
            }
            i += 1;
        }
        m
    }

    fn get(&self, key: &[u8]) -> Option<u32> {
        if self.slot.is_empty() {
            return None;
        }
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

    fn get_ascii_lower(&self, raw: &[u8]) -> Option<u32> {
        let mut h = 0x517cc1b727220a95u64;
        for &b in raw {
            let l = if b.is_ascii_uppercase() { b + 32 } else { b };
            h = mix(h, l);
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
            if n == raw.len() && raw.iter().zip(&self.arena[o..o + n]).all(|(&a, &b)| {
                let l = if a.is_ascii_uppercase() { a + 32 } else { a };
                l == b
            }) {
                return Some(self.id[i]);
            }
            p = (p + 1) & self.mask;
        }
    }
}

struct Dict {
    ll: u32,
    lang_id: u32,
    fwd: Map,
    rev: HashMap<u32, String>,
    ascii: [u32; 128],
}

impl Dict {
    // FASE 3b — load llama a build (directo) y luego arma rev (inverso) con el mismo i.
    fn load(list: &str, ll: u32, lang_id: u32) -> Self {
        let mut rev = HashMap::new();
        let mut i = 0u32;
        for w in list.split_whitespace() {
            let k: String = w.chars().flat_map(|c| c.to_lowercase()).collect();
            if k.is_empty() || rev.values().any(|v| v == &k) {
                continue;
            }
            // rev filled after map to avoid O(n^2). rebuilt below.
            let _ = i;
            i += 1;
            let _ = k;
            break;
        }
        let fwd = Map::build(list, ll);
        rev.clear();
        // rebuild rev from the same walk so ids match the map
        let mut i = 0u32;
        let mut seen = HashMap::new();
        for w in list.split_whitespace() {
            let k: String = w.chars().flat_map(|c| c.to_lowercase()).collect();
            if k.is_empty() || seen.contains_key(&k) {
                continue;
            }
            let tid = pack_lex(ll, i % 1000, SS0 + i / 1000);
            seen.insert(k.clone(), tid);
            rev.insert(tid, k);
            i += 1;
        }
        let mut ascii = [PAD; 128];
        for b in 0..128u8 {
            if let Some(id) = chars::enc(b as char) {
                ascii[b as usize] = id;
            }
        }
        Self { ll, lang_id, fwd, rev, ascii }
    }
}

pub struct Thpp {
    es: Dict,
    en: Dict,
}

impl Thpp {
    // FASE 3c — THPP expande es y en aquí, en new(), antes del primer encode.
    pub fn new() -> Self {
        Self {
            es: Dict::load(include_str!("../data/hppt_es.txt"), 1, 1201),
            en: Dict::load(include_str!("../data/hppt_en.txt"), 2, 1202),
        }
    }

    fn dict(&self, lang: &str) -> &Dict {
        match lang {
            "es" => &self.es,
            "en" => &self.en,
            _ => panic!("THPP solo es/en, llegó {lang}"),
        }
    }

    // FASE 4 — cinta. Ya no expande el dict: solo lookup.
    pub fn encode(&self, text: &str, lang: &str) -> Vec<u32> {
        let d = self.dict(lang);
        let b = text.as_bytes();
        let mut out = Vec::with_capacity(b.len() / 3 + 4);
        out.push(d.lang_id);
        let mut i = 0;
        let mut scratch = String::new();
        while i < b.len() {
            let c = b[i];
            if c == b' ' {
                out.push(SPACE);
                i += 1;
                continue;
            }
            if c == b'\n' {
                out.push(NL);
                i += 1;
                continue;
            }
            if c == b'\t' {
                out.push(TAB);
                i += 1;
                continue;
            }
            if c.is_ascii_alphabetic() {
                let s0 = i;
                i += 1;
                while i < b.len() && b[i].is_ascii_alphabetic() {
                    i += 1;
                }
                if i == b.len() || b[i] < 128 {
                    push_ascii_word(&mut out, d, &b[s0..i]);
                    continue;
                }
                i = s0;
                i = push_uni_word(&mut out, d, text, i, &mut scratch, false);
                continue;
            }
            if c.is_ascii_digit() {
                while i < b.len() && b[i].is_ascii_digit() {
                    out.push(d.ascii[b[i] as usize]);
                    i += 1;
                }
                continue;
            }
            if c < 128 {
                out.push(d.ascii[c as usize]);
                i += 1;
                continue;
            }
            i = push_uni_word(&mut out, d, text, i, &mut scratch, false);
        }
        out
    }

    // FASE 5 — roundtrip. Ignora LANG y SEP. SPACE/NL/TAB vuelven a carácter.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::with_capacity(ids.len() * 3);
        let mut case = 0u8;
        for &t in ids {
            if t == 1201 || t == 1202 {
                continue;
            }
            if t == TITLE {
                case = 1;
                continue;
            }
            if t == UPPER {
                case = 2;
                continue;
            }
            if t == SEP {
                continue;
            }
            if t == PAD {
                s.push('\u{FFFD}');
                continue;
            }
            if t == SPACE {
                s.push(' ');
                case = 0;
                continue;
            }
            if t == NL {
                s.push('\n');
                case = 0;
                continue;
            }
            if t == TAB {
                s.push('\t');
                case = 0;
                continue;
            }
            if let Some(w) = self.es.rev.get(&t).or_else(|| self.en.rev.get(&t)) {
                push_cased(&mut s, w, case);
                case = 0;
                continue;
            }
            if let Some(ch) = unpack_letter(t).or_else(|| chars::dec_core(t)) {
                s.push(ch);
                case = 0;
                continue;
            }
            s.push('\u{FFFD}');
            case = 0;
        }
        s
    }

    pub fn size(&self, lang: &str) -> usize {
        self.dict(lang).fwd.id.len()
    }
}

impl Default for Thpp {
    fn default() -> Self {
        Self::new()
    }
}

fn push_cased(s: &mut String, w: &str, case: u8) {
    match case {
        2 => s.extend(w.chars().flat_map(|c| c.to_uppercase())),
        1 => {
            let mut it = w.chars();
            if let Some(f) = it.next() {
                s.extend(f.to_uppercase());
                s.push_str(it.as_str());
            }
        }
        _ => s.push_str(w),
    }
}

fn push_ascii_word(out: &mut Vec<u32>, d: &Dict, w: &[u8]) {
    if let Some(tid) = d.fwd.get_ascii_lower(w) {
        if w.len() > 1 && w.iter().all(|x| x.is_ascii_uppercase()) {
            out.push(UPPER);
        } else if w[0].is_ascii_uppercase() {
            out.push(TITLE);
        }
        out.push(tid);
    } else {
        out.push(SEP);
        for &ch in w {
            out.push(d.ll * 10_000 + d.ascii[ch as usize]);
        }
    }
}

fn push_uni_word(out: &mut Vec<u32>, d: &Dict, text: &str, i: usize, scratch: &mut String, cjk: bool) -> usize {
    let rest = &text[i..];
    let ch0 = rest.chars().next().unwrap();
    if cjk && is_cjk(ch0) {
        scratch.clear();
        scratch.push(ch0);
        if let Some(tid) = d.fwd.get(scratch.as_bytes()) {
            out.push(tid);
        } else {
            out.push(SEP);
            out.push(PAD);
        }
        return i + ch0.len_utf8();
    }
    if !ch0.is_alphabetic() {
        out.push(chars::enc(ch0).unwrap_or(PAD));
        return i + ch0.len_utf8();
    }
    let mut end = 0;
    scratch.clear();
    let mut nch = 0u32;
    let mut all_upper = true;
    let mut first_upper = false;
    for ch in rest.chars() {
        if ch.is_alphabetic() && !(cjk && is_cjk(ch)) {
            if nch == 0 {
                first_upper = ch.is_uppercase();
            }
            if !ch.is_uppercase() {
                all_upper = false;
            }
            scratch.push(ch);
            end += ch.len_utf8();
            nch += 1;
        } else {
            break;
        }
    }
    let key: String = scratch.chars().flat_map(|c| c.to_lowercase()).collect();
    if let Some(tid) = d.fwd.get(key.as_bytes()) {
        if nch > 1 && all_upper {
            out.push(UPPER);
        } else if first_upper {
            out.push(TITLE);
        }
        out.push(tid);
    } else {
        out.push(SEP);
        for ch in scratch.chars() {
            out.push(pack_letter(d.ll, ch));
        }
    }
    i + end
}

fn is_cjk(c: char) -> bool {
    let u = c as u32;
    (0x3400..=0x9FFF).contains(&u)
        || (0xF900..=0xFAFF).contains(&u)
        || (0x20000..=0x2FA1F).contains(&u)
        || (0x3040..=0x30FF).contains(&u)
        || (0xAC00..=0xD7AF).contains(&u)
        || (0x0E00..=0x0E7F).contains(&u)
}
