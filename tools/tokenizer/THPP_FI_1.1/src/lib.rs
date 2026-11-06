//! THPP 1.1 — HPPT 1.0 + THPP 1.0 unificados en un crate.
//!
//! Cinta: `LANG [VER] [FORM] [TITLE|UPPER|TITLE_L|UPPER_L|MIXED CASE..] LEXEMA SPACE ... [SEP LETRA..] PUNCT`.
//! Todos los ids de 0.9 / 0.3 / 1.0 siguen iguales. Los ids nuevos viven solo en rangos que antes no se usaban:
//!
//! | id                    | nombre    | uso                                                   |
//! |-----------------------|-----------|-------------------------------------------------------|
//! | 1103                  | MIXED     | caso mixto; le siguen 1+ CASE y luego el lexema       |
//! | 1104 / 1105           | TITLE_L / UPPER_L | mayúsculas con reglas del idioma (tr: i→İ, ı→I) |
//! | 1106 / 1107           | FORM_NFC / FORM_NFD | el siguiente lexema se escribe en NFC / NFD  |
//! | 1300 + v              | VER       | versión opcional de la cinta (1.1 → 1311)             |
//! | 4_000_000 + m16       | CASE      | 16 bits de máscara de mayúsculas por id               |
//! | 5_000_000 + cp        | CP        | escape crudo de un code point Unicode (sin pérdida)   |
//! | 6_200_000 + b         | BYTE      | byte suelto de UTF-8 inválido (solo encode_bytes)     |

pub mod chars;
mod casing;
mod dict;
pub mod tht;
pub mod train;

use std::sync::OnceLock;
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::{is_nfc_quick, IsNormalized, UnicodeNormalization};

use dict::Dict;

// ---- ids fijos 1.0 (sin cambios) ----
pub const SPACE: u32 = 1001;
pub const SEP: u32 = 0;
pub const PAD: u32 = 111;
pub const NL: u32 = 11;
pub const TAB: u32 = 110;
pub const TITLE: u32 = 1101;
pub const UPPER: u32 = 1102;
pub const FF: u32 = 10;
pub const SS0: u32 = 10;
pub const LANG_BASE: u32 = 1200;

// ---- ids nuevos 1.1 (rangos antes libres) ----
pub const MIXED: u32 = 1103;
pub const TITLE_L: u32 = 1104;
pub const UPPER_L: u32 = 1105;
pub const FORM_NFC: u32 = 1106;
pub const FORM_NFD: u32 = 1107;
pub const VER_BASE: u32 = 1300;
pub const VER_1_1: u32 = 1311;
pub const CASE_BASE: u32 = 4_000_000;
pub const CASE_END: u32 = CASE_BASE + 0xFFFF;
pub const CP_BASE: u32 = 5_000_000;
pub const CP_END: u32 = CP_BASE + 0x10FFFF;
pub const BYTE_BASE: u32 = 6_200_000;
pub const BYTE_END: u32 = BYTE_BASE + 255;
/// Ids de letra de 1.0: LL*10_000 + CCC. Se aceptan en decode en [10_000, CASE_BASE).
pub const LETTER_MIN: u32 = 10_000;
pub const LEX_MIN: u32 = 10_000_000;

pub const VERSION: &str = "1.1.0";

#[derive(Clone, Copy, Debug)]
pub struct Lang {
    pub code: &'static str,
    pub ll: u32,
    pub lang_id: u32,
}

pub const LANGS: &[Lang] = &[
    Lang { code: "es", ll: 1, lang_id: 1201 },
    Lang { code: "en", ll: 2, lang_id: 1202 },
    Lang { code: "fr", ll: 3, lang_id: 1203 },
    Lang { code: "pt", ll: 4, lang_id: 1204 },
    Lang { code: "it", ll: 5, lang_id: 1205 },
    Lang { code: "de", ll: 6, lang_id: 1206 },
    Lang { code: "ca", ll: 7, lang_id: 1207 },
    Lang { code: "ro", ll: 8, lang_id: 1208 },
    Lang { code: "nl", ll: 9, lang_id: 1209 },
    Lang { code: "gl", ll: 10, lang_id: 1210 },
    Lang { code: "pl", ll: 11, lang_id: 1211 },
    Lang { code: "sv", ll: 12, lang_id: 1212 },
    Lang { code: "id", ll: 13, lang_id: 1213 },
    Lang { code: "tr", ll: 14, lang_id: 1214 },
    Lang { code: "vi", ll: 15, lang_id: 1215 },
    Lang { code: "sw", ll: 16, lang_id: 1216 },
    Lang { code: "af", ll: 17, lang_id: 1217 },
    Lang { code: "fi", ll: 18, lang_id: 1218 },
    Lang { code: "cs", ll: 19, lang_id: 1219 },
    Lang { code: "hu", ll: 20, lang_id: 1220 },
    Lang { code: "ru", ll: 21, lang_id: 1221 },
    Lang { code: "ar", ll: 22, lang_id: 1222 },
    Lang { code: "zh", ll: 23, lang_id: 1223 },
    Lang { code: "el", ll: 24, lang_id: 1224 },
    Lang { code: "he", ll: 25, lang_id: 1225 },
    Lang { code: "hi", ll: 26, lang_id: 1226 },
    Lang { code: "ja", ll: 27, lang_id: 1227 },
    Lang { code: "ko", ll: 28, lang_id: 1228 },
    Lang { code: "th", ll: 29, lang_id: 1229 },
    Lang { code: "fa", ll: 30, lang_id: 1230 },
    Lang { code: "bn", ll: 31, lang_id: 1231 },
];

/// Listas congeladas. El orden de cada txt define los ids: NO editar, solo anexar al final.
pub fn list_of(code: &str) -> &'static str {
    match code {
        "es" => include_str!("../data/hppt_es.txt"),
        "en" => include_str!("../data/hppt_en.txt"),
        "fr" => include_str!("../data/hppt_fr.txt"),
        "pt" => include_str!("../data/hppt_pt.txt"),
        "it" => include_str!("../data/hppt_it.txt"),
        "de" => include_str!("../data/hppt_de.txt"),
        "ca" => include_str!("../data/hppt_ca.txt"),
        "ro" => include_str!("../data/hppt_ro.txt"),
        "nl" => include_str!("../data/hppt_nl.txt"),
        "gl" => include_str!("../data/hppt_gl.txt"),
        "pl" => include_str!("../data/hppt_pl.txt"),
        "sv" => include_str!("../data/hppt_sv.txt"),
        "id" => include_str!("../data/hppt_id.txt"),
        "tr" => include_str!("../data/hppt_tr.txt"),
        "vi" => include_str!("../data/hppt_vi.txt"),
        "sw" => include_str!("../data/hppt_sw.txt"),
        "af" => include_str!("../data/hppt_af.txt"),
        "fi" => include_str!("../data/hppt_fi.txt"),
        "cs" => include_str!("../data/hppt_cs.txt"),
        "hu" => include_str!("../data/hppt_hu.txt"),
        "ru" => include_str!("../data/hppt_ru.txt"),
        "ar" => include_str!("../data/hppt_ar.txt"),
        "zh" => include_str!("../data/hppt_zh.txt"),
        "el" => include_str!("../data/hppt_el.txt"),
        "he" => include_str!("../data/hppt_he.txt"),
        "hi" => include_str!("../data/hppt_hi.txt"),
        "ja" => include_str!("../data/hppt_ja.txt"),
        "ko" => include_str!("../data/hppt_ko.txt"),
        "th" => include_str!("../data/hppt_th.txt"),
        "fa" => include_str!("../data/hppt_fa.txt"),
        "bn" => include_str!("../data/hppt_bn.txt"),
        _ => "",
    }
}

/// FNV-1a 64 de los bytes de un txt de lista.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn list_checksum(code: &str) -> u64 {
    fnv1a64(list_of(code).as_bytes())
}

/// Checksum combinado de las 31 listas (en el orden de LANGS). Va en la cabecera .tht.
pub fn lists_checksum() -> u64 {
    let mut buf = Vec::with_capacity(31 * 11);
    for l in LANGS {
        buf.extend_from_slice(l.code.as_bytes());
        buf.extend_from_slice(&list_checksum(l.code).to_le_bytes());
    }
    fnv1a64(&buf)
}

/// Checksums congelados (FNV-1a 64) de data/hppt_<ll>.txt. Si uno cambia, los ids cambian.
pub const FROZEN_LIST_CHECKSUMS: &[(&str, u64)] = &include!("checksums.in");

pub fn lang_by_code(code: &str) -> Option<Lang> {
    LANGS.iter().copied().find(|l| l.code == code)
}

pub fn lang_by_ll(ll: u32) -> Option<Lang> {
    if ll >= 1 && (ll as usize) <= LANGS.len() {
        Some(LANGS[ll as usize - 1])
    } else {
        None
    }
}

pub fn pack_lex(ll: u32, bbb: u32, ss: u32) -> u32 {
    ll * 10_000_000 + FF * 100_000 + ss * 1_000 + bbb
}

/// Igual que 1.0: solo la tabla `enc` (latina). PAD si no hay CCC.
pub fn pack_letter(ll: u32, ch: char) -> u32 {
    match chars::enc(ch) {
        Some(core) => ll * 10_000 + core,
        None => PAD,
    }
}

/// 1.1: `enc`, luego `enc_ext` (cirílico/árabe), luego escape CP. Nunca PAD.
pub fn pack_letter_ext(ll: u32, ch: char) -> u32 {
    match chars::enc(ch).or_else(|| chars::enc_ext(ch)) {
        Some(core) => ll * 10_000 + core,
        None => CP_BASE + ch as u32,
    }
}

pub fn unpack_letter(t: u32) -> Option<char> {
    if !(LETTER_MIN..CASE_BASE).contains(&t) {
        return None;
    }
    let c = t % 10_000;
    chars::dec_core(c).or_else(|| chars::dec_ext(c))
}

/// (LL, SS, BBB) de un id de lexema. None si no es lexema.
pub fn lex_parts(t: u32) -> Option<(u32, u32, u32)> {
    if t < LEX_MIN {
        return None;
    }
    let ll = t / 10_000_000;
    let r = t % 10_000_000;
    if r < FF * 100_000 {
        return None;
    }
    let r = r - FF * 100_000;
    let ss = r / 1000;
    if ss < SS0 {
        return None;
    }
    Some((ll, ss, r % 1000))
}

/// Índice i en la lista (orden del txt) de un id de lexema.
pub fn lex_index(t: u32) -> Option<(u32, usize)> {
    lex_parts(t).map(|(ll, ss, bbb)| (ll, ((ss - SS0) * 1000 + bbb) as usize))
}

pub fn is_cjk(c: char) -> bool {
    let u = c as u32;
    (0x3400..=0x9FFF).contains(&u)
        || (0xF900..=0xFAFF).contains(&u)
        || (0x20000..=0x2FA1F).contains(&u)
        || (0x3040..=0x30FF).contains(&u)
        || (0xAC00..=0xD7AF).contains(&u)
        || (0x0E00..=0x0E7F).contains(&u)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Arma los dicts en `new()` (THPP 1.0: es/en).
    Eager,
    /// Arma cada dict la primera vez que se pide (HPPT 1.0).
    Lazy,
}

/// Opciones de encode. `Config::compat()` (por defecto) da exactamente la cinta 1.0
/// para todo texto que 1.0 ya codificaba sin pérdida. `Config::v11()` activa extras.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// NFC del texto antes de codificar (cinta canónica; el roundtrip devuelve el NFC).
    pub nfc_input: bool,
    /// Busca entradas con guion/apóstrofo de la lista (est-ce, good-bye) como un solo lexema.
    pub compounds: bool,
    /// CJK/Thai: busca la entrada más larga de la lista en vez de 1 glifo.
    pub cjk_longest: bool,
    /// tr: prueba primero minúsculas turcas (I→ı, İ→i).
    pub locale_first: bool,
    /// Empuja VER_1_1 después de LANG.
    pub emit_version: bool,
}

impl Config {
    pub const fn compat() -> Self {
        Self { nfc_input: false, compounds: false, cjk_longest: false, locale_first: false, emit_version: false }
    }
    pub const fn v11() -> Self {
        Self { nfc_input: false, compounds: true, cjk_longest: true, locale_first: true, emit_version: true }
    }
    pub fn flags(&self) -> u32 {
        (self.emit_version as u32)
            | (self.nfc_input as u32) << 1
            | (self.compounds as u32) << 2
            | (self.cjk_longest as u32) << 3
            | (self.locale_first as u32) << 4
    }
    pub fn from_flags(f: u32) -> Self {
        Self {
            emit_version: f & 1 != 0,
            nfc_input: f & 2 != 0,
            compounds: f & 4 != 0,
            cjk_longest: f & 8 != 0,
            locale_first: f & 16 != 0,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::compat()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    UnknownLang(String),
    LangNotEnabled(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::UnknownLang(l) => write!(f, "idioma desconocido: {l}"),
            Error::LangNotEnabled(l) => write!(f, "idioma no habilitado en esta instancia: {l}"),
        }
    }
}
impl std::error::Error for Error {}

pub struct Thpp {
    dicts: Vec<OnceLock<Dict>>,
    enabled: Vec<bool>,
    cfg: Config,
    mode: Mode,
}

impl Default for Thpp {
    fn default() -> Self {
        Self::new()
    }
}

impl Thpp {
    /// Lazy, 31 idiomas, Config::compat() (lo que hacía HPPT 1.0).
    pub fn new() -> Self {
        Self::lazy()
    }

    pub fn lazy() -> Self {
        Self::build(Mode::Lazy, None, Config::compat()).unwrap()
    }

    /// Eager es/en (lo que hacía THPP 1.0). Otros idiomas: error en encode.
    pub fn eager() -> Self {
        Self::build(Mode::Eager, Some(&["es", "en"]), Config::compat()).unwrap()
    }

    /// `langs = None` → los 31. En Eager se arman todos los `langs` ya.
    pub fn build(mode: Mode, langs: Option<&[&str]>, cfg: Config) -> Result<Self, Error> {
        let mut enabled = vec![langs.is_none(); LANGS.len()];
        if let Some(ls) = langs {
            for l in ls {
                let lg = lang_by_code(l).ok_or_else(|| Error::UnknownLang(l.to_string()))?;
                enabled[lg.ll as usize - 1] = true;
            }
        }
        let t = Self { dicts: (0..LANGS.len()).map(|_| OnceLock::new()).collect(), enabled, cfg, mode };
        if mode == Mode::Eager {
            for (i, l) in LANGS.iter().enumerate() {
                if t.enabled[i] {
                    t.dict_ll(l.ll);
                }
            }
        }
        Ok(t)
    }

    pub fn with_config(mut self, cfg: Config) -> Self {
        self.cfg = cfg;
        self
    }

    pub fn config(&self) -> Config {
        self.cfg
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn is_enabled(&self, lang: &str) -> bool {
        lang_by_code(lang).map(|l| self.enabled[l.ll as usize - 1]).unwrap_or(false)
    }

    fn dict_ll(&self, ll: u32) -> Option<&Dict> {
        let l = lang_by_ll(ll)?;
        Some(self.dicts[ll as usize - 1].get_or_init(|| Dict::load(list_of(l.code), l.ll, l.lang_id)))
    }

    fn dict_checked(&self, lang: &str) -> Result<&Dict, Error> {
        let l = lang_by_code(lang).ok_or_else(|| Error::UnknownLang(lang.to_string()))?;
        if !self.enabled[l.ll as usize - 1] {
            return Err(Error::LangNotEnabled(lang.to_string()));
        }
        Ok(self.dict_ll(l.ll).unwrap())
    }

    /// Arma el dict de `lang` ahora (lo que pedía tool.rs 1.0).
    pub fn prepare(&self, lang: &str) -> Result<(), Error> {
        self.dict_checked(lang).map(|_| ())
    }

    pub fn prepare_all(&self) {
        for (i, l) in LANGS.iter().enumerate() {
            if self.enabled[i] {
                self.dict_ll(l.ll);
            }
        }
    }

    pub fn is_loaded(&self, lang: &str) -> bool {
        lang_by_code(lang).map(|l| self.dicts[l.ll as usize - 1].get().is_some()).unwrap_or(false)
    }

    /// Número de lexemas (entradas únicas) de la lista.
    pub fn size(&self, lang: &str) -> usize {
        lang_by_code(lang).and_then(|l| self.dict_ll(l.ll)).map(|d| d.n_primary()).unwrap_or(0)
    }

    /// Texto del lexema `id` (forma guardada en la lista, minúsculas).
    pub fn word(&self, id: u32) -> Option<&str> {
        let (ll, i) = lex_index(id)?;
        self.dict_ll(ll)?.word(i)
    }

    /// Id de lexema de una palabra (clave en minúsculas, NFC o tal cual).
    pub fn lookup(&self, word: &str, lang: &str) -> Option<u32> {
        let d = self.dict_checked(lang).ok()?;
        let k = casing::lower_generic(word);
        d.get(k.as_bytes()).or_else(|| d.get(k.nfc().collect::<String>().as_bytes()))
    }

    pub fn encode(&self, text: &str, lang: &str) -> Vec<u32> {
        self.try_encode(text, lang).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_encode(&self, text: &str, lang: &str) -> Result<Vec<u32>, Error> {
        let mut out = Vec::with_capacity(text.len() / 3 + 4);
        self.encode_into(text, lang, &mut out)?;
        Ok(out)
    }

    /// Añade a `out` la cinta completa (con LANG) de `text`.
    pub fn encode_into(&self, text: &str, lang: &str, out: &mut Vec<u32>) -> Result<(), Error> {
        let d = self.dict_checked(lang)?;
        out.push(d.lang_id);
        if self.cfg.emit_version {
            out.push(VER_1_1);
        }
        self.encode_body(text, d, out);
        Ok(())
    }

    /// Bytes arbitrarios: UTF-8 válido igual que `encode`; bytes inválidos → BYTE_BASE+b.
    pub fn encode_bytes(&self, bytes: &[u8], lang: &str) -> Result<Vec<u32>, Error> {
        let d = self.dict_checked(lang)?;
        let mut out = Vec::with_capacity(bytes.len() / 3 + 4);
        out.push(d.lang_id);
        if self.cfg.emit_version {
            out.push(VER_1_1);
        }
        let mut rest = bytes;
        loop {
            match std::str::from_utf8(rest) {
                Ok(s) => {
                    self.encode_body(s, d, &mut out);
                    break;
                }
                Err(e) => {
                    let ok = e.valid_up_to();
                    // SAFETY: from_utf8 garantiza que rest[..ok] es UTF-8 válido.
                    let s = unsafe { std::str::from_utf8_unchecked(&rest[..ok]) };
                    self.encode_body(s, d, &mut out);
                    let bad = e.error_len().unwrap_or(rest.len() - ok);
                    for &b in &rest[ok..ok + bad] {
                        out.push(BYTE_BASE + b as u32);
                    }
                    rest = &rest[ok + bad..];
                    if rest.is_empty() {
                        break;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Igual que `encode` pero parte el texto en trozos en saltos de línea y los codifica en
    /// `threads` hilos. La cinta es idéntica a la secuencial (ningún token cruza un '\n').
    pub fn encode_parallel(&self, text: &str, lang: &str, threads: usize) -> Result<Vec<u32>, Error> {
        let d = self.dict_checked(lang)?;
        let threads = threads.max(1);
        if threads == 1 || text.len() < 1 << 20 {
            return self.try_encode(text, lang);
        }
        let target = text.len() / (threads * 4) + 1;
        let b = text.as_bytes();
        let mut cuts = vec![0usize];
        let mut p = 0;
        while p < b.len() {
            let mut q = (p + target).min(b.len());
            while q < b.len() && b[q - 1] != b'\n' {
                q += 1;
            }
            cuts.push(q);
            p = q;
        }
        let pieces: Vec<&str> = cuts.windows(2).map(|w| &text[w[0]..w[1]]).collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut parts: Vec<Vec<u32>> = vec![Vec::new(); pieces.len()];
        let slots: Vec<std::sync::Mutex<Vec<u32>>> = (0..pieces.len()).map(|_| std::sync::Mutex::new(Vec::new())).collect();
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| loop {
                    let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if k >= pieces.len() {
                        break;
                    }
                    let mut v = Vec::with_capacity(pieces[k].len() / 3 + 4);
                    self.encode_body(pieces[k], d, &mut v);
                    *slots[k].lock().unwrap() = v;
                });
            }
        });
        for (k, m) in slots.into_iter().enumerate() {
            parts[k] = m.into_inner().unwrap();
        }
        let n: usize = parts.iter().map(|p| p.len()).sum();
        let mut out = Vec::with_capacity(n + 2);
        out.push(d.lang_id);
        if self.cfg.emit_version {
            out.push(VER_1_1);
        }
        for p in parts {
            out.extend_from_slice(&p);
        }
        Ok(out)
    }

    fn encode_body(&self, text: &str, d: &Dict, out: &mut Vec<u32>) {
        if self.cfg.nfc_input && is_nfc_quick(text.chars()) != IsNormalized::Yes {
            let n: String = text.nfc().collect();
            if n != text {
                return self.encode_body_raw(&n, d, out);
            }
        }
        self.encode_body_raw(text, d, out)
    }

    fn encode_body_raw(&self, text: &str, d: &Dict, out: &mut Vec<u32>) {
        let b = text.as_bytes();
        let cfg = &self.cfg;
        let turkic = casing::is_turkic(d.ll);
        let mut i = 0;
        let mut sc = Scratch::default();
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
                let joiner_next = cfg.compounds && i < b.len() && casing::is_joiner(b[i] as char);
                let tr_slow = turkic && cfg.locale_first && b[s0..i].iter().any(|&x| x == b'I');
                if (i == b.len() || b[i] < 128) && !joiner_next && !tr_slow {
                    push_ascii_word(out, d, &b[s0..i]);
                    continue;
                }
                i = self.push_uni(out, d, text, s0, &mut sc);
                continue;
            }
            if c.is_ascii_digit() {
                out.push(d.ascii[c as usize]);
                i += 1;
                continue;
            }
            if c < 128 {
                let id = d.ascii[c as usize];
                out.push(if id == PAD { CP_BASE + c as u32 } else { id });
                i += 1;
                continue;
            }
            i = self.push_uni(out, d, text, i, &mut sc);
        }
    }

    fn push_uni(&self, out: &mut Vec<u32>, d: &Dict, text: &str, i: usize, sc: &mut Scratch) -> usize {
        let rest = &text[i..];
        let ch0 = rest.chars().next().unwrap();
        if is_cjk(ch0) {
            if self.cfg.cjk_longest && d.max_cjk > 1 {
                // ventana de hasta max_cjk glifos CJK consecutivos, del más largo al más corto
                let mut ends = Vec::with_capacity(d.max_cjk);
                let mut e = 0;
                for ch in rest.chars().take(d.max_cjk) {
                    if !is_cjk(ch) {
                        break;
                    }
                    e += ch.len_utf8();
                    ends.push(e);
                }
                for &e in ends.iter().rev() {
                    if let Some(tid) = d.get(rest[..e].as_bytes()) {
                        out.push(tid);
                        return i + e;
                    }
                }
            } else {
                let n = ch0.len_utf8();
                if let Some(tid) = d.get(rest[..n].as_bytes()) {
                    out.push(tid);
                    return i + n;
                }
            }
            out.push(SEP);
            out.push(pack_letter_ext(d.ll, ch0));
            return i + ch0.len_utf8();
        }
        if !ch0.is_alphabetic() {
            out.push(chars::enc(ch0).or_else(|| chars::enc_ext(ch0)).unwrap_or(CP_BASE + ch0 as u32));
            return i + ch0.len_utf8();
        }
        let end = word_end(rest);
        if self.cfg.compounds {
            // probar cadenas a-b(-c...) completas, de la más larga a la más corta
            let mut ends = vec![];
            let mut e = end;
            loop {
                let mut it = rest[e..].chars();
                match (it.next(), it.next()) {
                    (Some(j), Some(n)) if casing::is_joiner(j) && n.is_alphabetic() && !is_cjk(n) => {
                        e = e + j.len_utf8() + word_end(&rest[e + j.len_utf8()..]);
                        ends.push(e);
                    }
                    _ => break,
                }
                if ends.len() >= 4 {
                    break;
                }
            }
            for &e in ends.iter().rev() {
                if self.try_word_lexeme(out, d, &rest[..e], sc) {
                    return i + e;
                }
            }
        }
        let w = &rest[..end];
        if !self.try_word_lexeme(out, d, w, sc) {
            out.push(SEP);
            for ch in w.chars() {
                out.push(pack_letter_ext(d.ll, ch));
            }
        }
        i + end
    }

    /// Busca `w` como lexema y emite marcas + id si alguna forma reproduce `w` exacto.
    fn try_word_lexeme(&self, out: &mut Vec<u32>, d: &Dict, w: &str, sc: &mut Scratch) -> bool {
        let turkic = casing::is_turkic(d.ll);
        // camino rápido (= 1.0): clave genérica + marca de 1.0, sin reservar memoria
        if !(turkic && self.cfg.locale_first) {
            sc.key.clear();
            for c in w.chars() {
                sc.key.extend(c.to_lowercase());
            }
            match d.get(sc.key.as_bytes()) {
                Some(tid) => {
                    let stored = d.word_by_id(tid).unwrap();
                    let m = casing::marker_10(w);
                    let hit = if m == 0 { stored == w } else { casing::render_into(&mut sc.buf, stored, m, d.ll, &[], 0) && sc.buf == w };
                    if hit {
                        if m != 0 {
                            out.push(m);
                        }
                        out.push(tid);
                        return true;
                    }
                }
                None => {
                    if !turkic && is_nfc_quick(sc.key.chars()) == IsNormalized::Yes {
                        return false;
                    }
                }
            }
        }
        let k_gen = casing::lower_generic(w);
        let mut keys: Vec<String> = Vec::with_capacity(4);
        let push_key = |keys: &mut Vec<String>, k: String| {
            if !k.is_empty() && !keys.contains(&k) {
                keys.push(k);
            }
        };
        if turkic && self.cfg.locale_first {
            let k = casing::lower_turkic(w);
            let n: String = k.nfc().collect();
            push_key(&mut keys, k);
            push_key(&mut keys, n);
        }
        push_key(&mut keys, k_gen.clone());
        if is_nfc_quick(k_gen.chars()) != IsNormalized::Yes {
            push_key(&mut keys, k_gen.nfc().collect());
        }
        if turkic && !self.cfg.locale_first {
            let k = casing::lower_turkic(w);
            let n: String = k.nfc().collect();
            push_key(&mut keys, k);
            push_key(&mut keys, n);
        }
        let w_nfc = unicode_normalization::is_nfc(w);
        let w_nfd = unicode_normalization::is_nfd(w);
        let mut tried: Vec<u32> = Vec::new();
        for k in keys.iter() {
            let Some(tid) = d.get(k.as_bytes()) else { continue };
            if tried.contains(&tid) {
                continue;
            }
            tried.push(tid);
            let stored = d.word_by_id(tid).unwrap();
            // 1) la elección exacta de 1.0 (solo para la clave genérica: compat byte a byte)
            if *k == k_gen {
                let m = casing::marker_10(w);
                if casing::render_into(&mut sc.buf, stored, m, d.ll, &[], 0) && sc.buf == w {
                    if m != 0 {
                        out.push(m);
                    }
                    out.push(tid);
                    return true;
                }
            }
            // 2) resto de variantes: forma × caso
            for form in [0u32, FORM_NFC, FORM_NFD] {
                // una forma solo puede reproducir `w` si `w` ya está en esa forma
                if (form == FORM_NFC && !w_nfc) || (form == FORM_NFD && !w_nfd) {
                    continue;
                }
                let mut cands: Vec<u32> = vec![0, TITLE, UPPER];
                if turkic {
                    cands.push(TITLE_L);
                    cands.push(UPPER_L);
                }
                for m in cands {
                    if casing::render_into(&mut sc.buf, stored, m, d.ll, &[], form) && sc.buf == w {
                        if form != 0 {
                            out.push(form);
                        }
                        if m != 0 {
                            out.push(m);
                        }
                        out.push(tid);
                        return true;
                    }
                }
                // MIXED: máscara por carácter del lexema (ya en la forma `form`)
                if let Some(mask) = casing::mixed_mask(stored, w, form) {
                    if casing::render_into(&mut sc.buf, stored, MIXED, d.ll, &mask, form) && sc.buf == w {
                        if form != 0 {
                            out.push(form);
                        }
                        out.push(MIXED);
                        for m in &mask {
                            out.push(CASE_BASE + *m as u32);
                        }
                        out.push(tid);
                        return true;
                    }
                }
            }
        }
        false
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let v = self.decode_impl(ids, false);
        String::from_utf8(v).expect("decode produce UTF-8")
    }

    /// Como `decode`, pero los ids BYTE vuelven a ser bytes crudos (roundtrip de bytes no UTF-8).
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        self.decode_impl(ids, true)
    }

    fn decode_impl(&self, ids: &[u32], raw_bytes: bool) -> Vec<u8> {
        let mut s: Vec<u8> = Vec::with_capacity(ids.len() * 4);
        let mut case = 0u32;
        let mut form = 0u32;
        let mut mask: Vec<u16> = Vec::new();
        let mut buf = String::new();
        let mut tmp = [0u8; 4];
        for &t in ids {
            if t >= LEX_MIN {
                let w = lex_index(t).and_then(|(ll, i)| self.dict_ll(ll).and_then(|d| d.word(i).map(|w| (w, ll))));
                match w {
                    Some((w, ll)) => {
                        if case == 0 && form == 0 {
                            s.extend_from_slice(w.as_bytes());
                        } else {
                            casing::render_into(&mut buf, w, case, ll, &mask, form);
                            s.extend_from_slice(buf.as_bytes());
                        }
                    }
                    None => s.extend_from_slice("\u{FFFD}".as_bytes()),
                }
                case = 0;
                form = 0;
                mask.clear();
                continue;
            }
            match t {
                SPACE => {
                    s.push(b' ');
                    case = 0;
                    form = 0;
                }
                NL => {
                    s.push(b'\n');
                    case = 0;
                    form = 0;
                }
                TAB => {
                    s.push(b'\t');
                    case = 0;
                    form = 0;
                }
                SEP => {}
                PAD => s.extend_from_slice("\u{FFFD}".as_bytes()),
                TITLE | UPPER | TITLE_L | UPPER_L => case = t,
                MIXED => {
                    case = MIXED;
                    mask.clear();
                }
                FORM_NFC | FORM_NFD => form = t,
                1201..=1299 | VER_BASE..=1399 => {}
                CASE_BASE..=CASE_END => mask.push((t - CASE_BASE) as u16),
                CP_BASE..=CP_END => {
                    let ch = char::from_u32(t - CP_BASE).unwrap_or('\u{FFFD}');
                    s.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                    case = 0;
                }
                BYTE_BASE..=BYTE_END => {
                    if raw_bytes {
                        s.push((t - BYTE_BASE) as u8);
                    } else {
                        s.extend_from_slice("\u{FFFD}".as_bytes());
                    }
                }
                _ => {
                    let ch = if t < 1000 {
                        chars::dec_core(t).or_else(|| chars::dec_ext(t))
                    } else {
                        unpack_letter(t)
                    };
                    let ch = ch.unwrap_or('\u{FFFD}');
                    s.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                    case = 0;
                }
            }
        }
        s
    }
}

#[derive(Default)]
struct Scratch {
    buf: String,
    key: String,
}

/// Fin (bytes) de la racha de letras al inicio de `rest`: alfabéticos no-CJK + marcas combinantes.
fn word_end(rest: &str) -> usize {
    let mut end = 0;
    for ch in rest.chars() {
        if (ch.is_alphabetic() || (end > 0 && is_combining_mark(ch))) && !is_cjk(ch) {
            end += ch.len_utf8();
        } else {
            break;
        }
    }
    end
}

fn push_ascii_word(out: &mut Vec<u32>, d: &Dict, w: &[u8]) {
    if let Some(tid) = d.get_ascii_lower(w) {
        let all_upper = w.iter().all(|x| x.is_ascii_uppercase());
        let rest_lower = w[1..].iter().all(|x| x.is_ascii_lowercase());
        if w.len() > 1 && all_upper {
            out.push(UPPER);
        } else if w[0].is_ascii_uppercase() && rest_lower {
            out.push(TITLE);
        } else if !(w[0].is_ascii_lowercase() && rest_lower) {
            // caso mixto (iPhone, McDonald): máscara de 16 bits por id
            out.push(MIXED);
            for chunk in w.chunks(16) {
                let mut m = 0u32;
                for (k, &x) in chunk.iter().enumerate() {
                    if x.is_ascii_uppercase() {
                        m |= 1 << k;
                    }
                }
                out.push(CASE_BASE + m);
            }
        }
        out.push(tid);
    } else {
        out.push(SEP);
        for &ch in w {
            out.push(d.ll * 10_000 + d.ascii[ch as usize]);
        }
    }
}
