//! Mayúsculas/minúsculas y formas (NFC/NFD) usadas por encode y decode.
use crate::{FORM_NFC, FORM_NFD, MIXED, TITLE, TITLE_L, UPPER, UPPER_L};
use unicode_normalization::UnicodeNormalization;

/// tr (LL=14). az no tiene LL en 1.1; si se añade, va aquí.
pub fn is_turkic(ll: u32) -> bool {
    ll == 14
}

pub fn is_joiner(c: char) -> bool {
    c == '-' || c == '\'' || c == '`'
}

/// Minúsculas de 1.0 (`char::to_lowercase`, sin locale).
pub fn lower_generic(w: &str) -> String {
    let mut k = String::with_capacity(w.len());
    for c in w.chars() {
        k.extend(c.to_lowercase());
    }
    k
}

pub fn lower_turkic(w: &str) -> String {
    let mut k = String::with_capacity(w.len());
    for c in w.chars() {
        match c {
            'I' => k.push('ı'),
            'İ' => k.push('i'),
            _ => k.extend(c.to_lowercase()),
        }
    }
    k
}

fn push_upper(s: &mut String, c: char, turkic: bool) {
    if turkic {
        match c {
            'i' => return s.push('İ'),
            'ı' => return s.push('I'),
            _ => {}
        }
    }
    s.extend(c.to_uppercase());
}

/// Marca que elegía 1.0 para una palabra encontrada en la lista.
pub fn marker_10(w: &str) -> u32 {
    let mut n = 0;
    let mut all_upper = true;
    let mut first_upper = false;
    for c in w.chars() {
        if n == 0 {
            first_upper = c.is_uppercase();
        }
        if !c.is_uppercase() {
            all_upper = false;
        }
        n += 1;
    }
    if n > 1 && all_upper {
        UPPER
    } else if first_upper {
        TITLE
    } else {
        0
    }
}

/// Escribe en `buf` la forma de superficie de `stored` con la marca de caso y la forma dadas.
pub fn render_into(buf: &mut String, stored: &str, marker: u32, ll: u32, mask: &[u16], form: u32) -> bool {
    buf.clear();
    let tk = is_turkic(ll);
    match marker {
        TITLE | TITLE_L => {
            let mut it = stored.chars();
            if let Some(f) = it.next() {
                push_upper(buf, f, marker == TITLE_L && tk);
                buf.push_str(it.as_str());
            }
        }
        UPPER | UPPER_L => {
            for c in stored.chars() {
                push_upper(buf, c, marker == UPPER_L && tk);
            }
        }
        MIXED => {
            for (k, c) in stored.chars().enumerate() {
                let bit = mask.get(k / 16).map(|m| (m >> (k % 16)) & 1 == 1).unwrap_or(false);
                if bit {
                    push_upper(buf, c, false);
                } else {
                    buf.push(c);
                }
            }
        }
        _ => buf.push_str(stored),
    }
    match form {
        FORM_NFC => {
            let n: String = buf.nfc().collect();
            *buf = n;
        }
        FORM_NFD => {
            let n: String = buf.nfd().collect();
            *buf = n;
        }
        _ => {}
    }
    true
}

/// Máscara de mayúsculas (16 bits por id) que lleva `stored` a `w`, si existe.
pub fn mixed_mask(stored: &str, w: &str, form: u32) -> Option<Vec<u16>> {
    let n = stored.chars().count();
    let cands: Vec<String> = match form {
        0 => vec![w.to_string()],
        _ => vec![w.nfc().collect(), w.nfd().collect()],
    };
    'c: for x in cands {
        if x.chars().count() != n {
            continue;
        }
        let mut mask = vec![0u16; n.div_ceil(16).max(1)];
        let mut any = false;
        for (k, (s, t)) in stored.chars().zip(x.chars()).enumerate() {
            if s == t {
                continue;
            }
            let mut up = s.to_uppercase();
            if up.next() == Some(t) && up.next().is_none() {
                mask[k / 16] |= 1 << (k % 16);
                any = true;
            } else {
                continue 'c;
            }
        }
        if any {
            return Some(mask);
        }
    }
    None
}
