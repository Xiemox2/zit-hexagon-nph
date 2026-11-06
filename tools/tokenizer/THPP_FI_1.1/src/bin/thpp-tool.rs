//! thpp-tool 1.1 — contrato 1.0 (ids, encode, decode, stdin '-', file-encode, file-decode)
//! + .tht/.npy + verify + stats + corpus + vocab + dense + factors + bench.
use std::collections::HashMap;
use std::env;
use std::io::{self, BufWriter, Read, Write};
use std::time::Instant;
use thpp::train::{count_ids, factorize, DenseVocab, Kind};
use thpp::{tht, Config, Mode, Thpp, LANGS};

fn usage() {
    eprintln!(
        "thpp-tool 1.1\n\
         \n  thpp-tool ids\
         \n  thpp-tool encode <lang> <texto>        (stdin: thpp-tool encode <lang> -)\
         \n  thpp-tool decode <lang> <id id ...>    (stdin: thpp-tool decode <lang> -; lang se ignora: el id dice el idioma)\
         \n  thpp-tool file-encode <lang> <in.txt> <out.ids|out.tht|out.npy>\
         \n  thpp-tool file-decode <lang> <in.ids|in.tht|in.npy> <out.txt>\
         \n  thpp-tool verify <lang> <in.txt> [...]   roundtrip byte a byte\
         \n  thpp-tool stats <lang> <in.txt|in.tht>\
         \n  thpp-tool corpus <lang> <out.tht|out.npy> <in1.txt> [in2.txt ...]   (+ out.offsets.npy)\
         \n  thpp-tool vocab <out.vocab> <in.tht|.npy|.ids> [...] [--min N] [--max N]\
         \n  thpp-tool dense <vocab> <in.tht|.npy|.ids> <out.npy>\
         \n  thpp-tool undense <vocab> <in.npy> <out.tht>\
         \n  thpp-tool factors <in.tht|.npy|.ids> <out.npy>     (N,4): kind, lang, bucket, slot\
         \n  thpp-tool bench [lang] [in.txt]\
         \n\nopciones: --v11 (compounds, cjk_longest, locale_first, VER) --nfc --eager --threads N\
         \nlangs: {}",
        LANGS.iter().map(|l| l.code).collect::<Vec<_>>().join(" ")
    );
}

struct Opts {
    cfg: Config,
    eager: bool,
    threads: usize,
    min: u64,
    max: Option<usize>,
}

fn parse_opts(args: &mut Vec<String>) -> Opts {
    let mut o = Opts {
        cfg: Config::compat(),
        eager: false,
        threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        min: 1,
        max: None,
    };
    let mut i = 0;
    while i < args.len() {
        let a = args[i].clone();
        let take = |args: &mut Vec<String>, i: usize| -> String {
            let v = args.get(i + 1).cloned().unwrap_or_default();
            args.drain(i..=i + 1);
            v
        };
        match a.as_str() {
            "--v11" => {
                o.cfg = Config { nfc_input: o.cfg.nfc_input, ..Config::v11() };
                args.remove(i);
            }
            "--nfc" => {
                o.cfg.nfc_input = true;
                args.remove(i);
            }
            "--eager" => {
                o.eager = true;
                args.remove(i);
            }
            "--threads" => o.threads = take(args, i).parse().unwrap_or(1),
            "--min" => o.min = take(args, i).parse().unwrap_or(1),
            "--max" => o.max = take(args, i).parse().ok(),
            _ => i += 1,
        }
    }
    o
}

fn tok(o: &Opts) -> Thpp {
    let mode = if o.eager { Mode::Eager } else { Mode::Lazy };
    let langs: Option<&[&str]> = if o.eager { Some(&["es", "en"]) } else { None };
    Thpp::build(mode, langs, o.cfg).unwrap()
}

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2)
}

fn parse_ids(s: &str) -> Vec<u32> {
    s.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|x| !x.is_empty())
        .map(|x| x.parse::<u32>().unwrap_or_else(|_| die(format!("id inválido: {x}"))))
        .collect()
}

fn write_ids(ids: &[u32], w: &mut impl Write) -> io::Result<()> {
    let mut buf = [0u8; 10];
    for (i, &id) in ids.iter().enumerate() {
        if i > 0 {
            w.write_all(b" ")?;
        }
        let mut n = id;
        let mut j = 10;
        if n == 0 {
            w.write_all(b"0")?;
            continue;
        }
        while n > 0 {
            j -= 1;
            buf[j] = b'0' + (n % 10) as u8;
            n /= 10;
        }
        w.write_all(&buf[j..])?;
    }
    Ok(())
}

/// Lee ids de .tht, .npy o texto.
fn read_ids_file(path: &str) -> Vec<u32> {
    let b = std::fs::read(path).unwrap_or_else(|e| die(format!("{path}: {e}")));
    if tht::is_tht(&b) {
        let (h, ids) = tht::read_tht(&b).unwrap_or_else(|e| die(e));
        if h.list_checksum != thpp::lists_checksum() {
            eprintln!("aviso: {path} se hizo con otras listas (checksum {:016x})", h.list_checksum);
        }
        ids
    } else if tht::is_npy(&b) {
        tht::read_npy_u32(&b).unwrap_or_else(|e| die(e))
    } else {
        parse_ids(&String::from_utf8_lossy(&b))
    }
}

fn write_ids_file(path: &str, ids: &[u32], o: &Opts, ll: u32) {
    let r = if path.ends_with(".tht") {
        tht::write_tht_file(path, &tht::Header::new(o.cfg.flags(), ll, 0), ids)
    } else if path.ends_with(".npy") {
        tht::write_npy_u32_file(path, ids, &[ids.len()])
    } else {
        std::fs::File::create(path).and_then(|f| {
            let mut w = BufWriter::new(f);
            write_ids(ids, &mut w)?;
            w.flush()
        })
    };
    r.unwrap_or_else(|e| die(format!("{path}: {e}")));
}

fn ll_of(lang: &str) -> u32 {
    thpp::lang_by_code(lang).map(|l| l.ll).unwrap_or_else(|| die(format!("idioma {lang}")))
}

fn encode_file(t: &Thpp, lang: &str, raw: &[u8], threads: usize) -> Vec<u32> {
    match std::str::from_utf8(raw) {
        Ok(s) => t.encode_parallel(s, lang, threads).unwrap_or_else(|e| die(e)),
        Err(_) => t.encode_bytes(raw, lang).unwrap_or_else(|e| die(e)),
    }
}

fn rss_kb() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let f = |k: &str| {
        s.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|x| x.parse().ok()).unwrap_or(0)
    };
    (f("VmRSS:"), f("VmHWM:"))
}

fn stats(t: &Thpp, ids: &[u32], bytes: Option<usize>) {
    let mut k: HashMap<Kind, usize> = HashMap::new();
    let mut words = 0usize;
    let mut oov = 0usize;
    let mut uniq = std::collections::HashSet::new();
    let mut prev_sep = false;
    for &id in ids {
        let kind = factorize(id).kind;
        *k.entry(kind).or_insert(0) += 1;
        if kind == Kind::Lexeme {
            words += 1;
            uniq.insert(id);
        }
        if id == thpp::SEP {
            oov += 1;
        }
        prev_sep = id == thpp::SEP || (prev_sep && matches!(kind, Kind::Letter | Kind::CodePoint));
    }
    let _ = prev_sep;
    println!("ids            {}", ids.len());
    if let Some(b) = bytes {
        println!("bytes          {b}  ({:.2} bytes/id)", b as f64 / ids.len().max(1) as f64);
    }
    println!("lexemas        {words}  (únicos {})", uniq.len());
    println!("palabras OOV   {oov}  (SEP; tasa OOV {:.2}%)", 100.0 * oov as f64 / (words + oov).max(1) as f64);
    let mut kv: Vec<_> = k.into_iter().collect();
    kv.sort_by_key(|(a, _)| *a as u8);
    for (kind, n) in kv {
        println!("  {:<10} {n}", format!("{kind:?}"));
    }
    let pads = ids.iter().filter(|&&x| x == thpp::PAD).count();
    println!("PAD            {pads}");
    let _ = t;
}

fn synthetic(t: &Thpp, lang: &str, target: usize) -> String {
    let words: Vec<&str> = thpp::list_of(lang).split_whitespace().collect();
    let _ = t;
    let mut s = String::with_capacity(target + 64);
    let mut x: u64 = 12345;
    let mut n = 0;
    while s.len() < target {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let w = words[(x >> 33) as usize % words.len()];
        n += 1;
        if n % 12 == 0 {
            let mut c = w.chars();
            if let Some(f) = c.next() {
                s.extend(f.to_uppercase());
                s.push_str(c.as_str());
            }
        } else {
            s.push_str(w);
        }
        s.push_str(match (x >> 20) % 20 {
            0 => ", ",
            1 => ". ",
            2 => "\n",
            3 => " zqxj ",
            _ => " ",
        });
    }
    s
}

fn main() {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let o = parse_opts(&mut args);
    if args.is_empty() {
        usage();
        return;
    }
    let cmd = args.remove(0);
    let out = io::stdout();
    match cmd.as_str() {
        "ids" => {
            use thpp::*;
            println!("SPACE={SPACE} SEP={SEP} PAD={PAD} NL={NL} TAB={TAB} TITLE={TITLE} UPPER={UPPER}");
            println!("1.1: MIXED={MIXED} TITLE_L={TITLE_L} UPPER_L={UPPER_L} FORM_NFC={FORM_NFC} FORM_NFD={FORM_NFD} VER_1_1={VER_1_1}");
            println!("1.1: CASE={CASE_BASE}+mask16 CP={CP_BASE}+codepoint BYTE={BYTE_BASE}+byte");
            println!("pack_letter es a {}", pack_letter(1, 'a'));
            println!("pack_lex es0 {}", pack_lex(1, 0, 10));
            for l in LANGS {
                println!("lang {} ll={} id={} list_fnv={:016x}", l.code, l.ll, l.lang_id, list_checksum(l.code));
            }
            println!("lists_checksum {:016x}", lists_checksum());
        }
        "encode" => {
            if args.len() < 2 {
                return usage();
            }
            let t = tok(&o);
            t.prepare(&args[0]).unwrap_or_else(|e| die(e));
            let ids = if args[1] == "-" {
                let mut b = Vec::new();
                io::stdin().read_to_end(&mut b).unwrap();
                encode_file(&t, &args[0], &b, o.threads)
            } else {
                t.encode(&args[1..].join(" "), &args[0])
            };
            let mut w = BufWriter::new(out.lock());
            write_ids(&ids, &mut w).unwrap();
            w.write_all(b"\n").unwrap();
        }
        "decode" => {
            if args.len() < 2 {
                return usage();
            }
            let ids = if args[1] == "-" {
                let mut b = String::new();
                io::stdin().read_to_string(&mut b).unwrap();
                parse_ids(&b)
            } else {
                parse_ids(&args[1..].join(" "))
            };
            let t = tok(&o);
            let mut w = BufWriter::new(out.lock());
            w.write_all(&t.decode_bytes(&ids)).unwrap();
        }
        "file-encode" => {
            if args.len() != 3 {
                return usage();
            }
            let t = tok(&o);
            let raw = std::fs::read(&args[1]).unwrap_or_else(|e| die(format!("{}: {e}", args[1])));
            let ids = encode_file(&t, &args[0], &raw, o.threads);
            write_ids_file(&args[2], &ids, &o, ll_of(&args[0]));
            eprintln!("tokens={}", ids.len());
        }
        "file-decode" => {
            if args.len() != 3 {
                return usage();
            }
            let ids = read_ids_file(&args[1]);
            let t = tok(&o);
            std::fs::write(&args[2], t.decode_bytes(&ids)).unwrap_or_else(|e| die(e));
        }
        "verify" => {
            if args.len() < 2 {
                return usage();
            }
            let t = tok(&o);
            let mut bad = 0;
            for p in &args[1..] {
                let raw = std::fs::read(p).unwrap_or_else(|e| die(format!("{p}: {e}")));
                let ids = encode_file(&t, &args[0], &raw, o.threads);
                let back = t.decode_bytes(&ids);
                let nfc_ok = o.cfg.nfc_input;
                if back == raw {
                    println!("OK   {p}  bytes={} ids={} roundtrip byte-exacto", raw.len(), ids.len());
                } else if nfc_ok {
                    use unicode_normalization::UnicodeNormalization;
                    let n: String = String::from_utf8_lossy(&raw).nfc().collect();
                    if n.as_bytes() == back {
                        println!("OK   {p}  roundtrip exacto respecto al NFC de la entrada (--nfc)");
                    } else {
                        bad += 1;
                        println!("FAIL {p}");
                    }
                } else {
                    bad += 1;
                    let pos = back.iter().zip(&raw).position(|(a, b)| a != b).unwrap_or(back.len().min(raw.len()));
                    println!("FAIL {p}  primer byte distinto en {pos}");
                }
            }
            if bad > 0 {
                std::process::exit(1);
            }
        }
        "stats" => {
            if args.len() != 2 {
                return usage();
            }
            let t = tok(&o);
            let raw = std::fs::read(&args[1]).unwrap_or_else(|e| die(e));
            if tht::is_tht(&raw) || tht::is_npy(&raw) {
                stats(&t, &read_ids_file(&args[1]), None);
            } else {
                let t0 = Instant::now();
                let ids = encode_file(&t, &args[0], &raw, o.threads);
                let el = t0.elapsed();
                stats(&t, &ids, Some(raw.len()));
                println!("encode         {:?} ({:.1} MB/s, {} hilos)", el, raw.len() as f64 / 1e6 / el.as_secs_f64(), o.threads);
            }
        }
        "corpus" => {
            if args.len() < 3 {
                return usage();
            }
            let lang = args[0].clone();
            let outp = args[1].clone();
            let files = &args[2..];
            let t = tok(&o);
            t.prepare(&lang).unwrap_or_else(|e| die(e));
            let t0 = Instant::now();
            let next = std::sync::atomic::AtomicUsize::new(0);
            let res: Vec<std::sync::Mutex<Vec<u32>>> = files.iter().map(|_| std::sync::Mutex::new(vec![])).collect();
            let total_bytes = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..o.threads.min(files.len()).max(1) {
                    s.spawn(|| loop {
                        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if k >= files.len() {
                            break;
                        }
                        let raw = std::fs::read(&files[k]).unwrap_or_else(|e| die(format!("{}: {e}", files[k])));
                        total_bytes.fetch_add(raw.len(), std::sync::atomic::Ordering::Relaxed);
                        let ids = encode_file(&t, &lang, &raw, 1);
                        *res[k].lock().unwrap() = ids;
                    });
                }
            });
            let mut all = Vec::new();
            let mut offs = Vec::with_capacity(files.len() + 1);
            for r in res {
                offs.push(all.len() as u32);
                all.extend(r.into_inner().unwrap());
            }
            offs.push(all.len() as u32);
            let mut oo = o.cfg.flags();
            if files.len() > 1 {
                oo |= tht::FLAG_MULTI_DOC;
            }
            if outp.ends_with(".tht") {
                tht::write_tht_file(&outp, &tht::Header::new(oo, ll_of(&lang), 0), &all).unwrap_or_else(|e| die(e));
            } else {
                tht::write_npy_u32_file(&outp, &all, &[all.len()]).unwrap_or_else(|e| die(e));
            }
            let offp = format!("{}.offsets.npy", outp.trim_end_matches(".tht").trim_end_matches(".npy"));
            tht::write_npy_u32_file(&offp, &offs, &[offs.len()]).unwrap_or_else(|e| die(e));
            let el = t0.elapsed();
            let tb = total_bytes.into_inner();
            eprintln!(
                "docs={} ids={} bytes={} en {:?} ({:.1} MB/s) -> {outp} + {offp}",
                files.len(),
                all.len(),
                tb,
                el,
                tb as f64 / 1e6 / el.as_secs_f64()
            );
        }
        "vocab" => {
            if args.len() < 2 {
                return usage();
            }
            let mut counts = HashMap::new();
            for p in &args[1..] {
                count_ids(&mut counts, &read_ids_file(p));
            }
            let v = DenseVocab::from_counts(&counts, o.min, o.max);
            v.save_file(&args[0]).unwrap_or_else(|e| die(e));
            eprintln!("ids distintos={} vocab denso={} -> {}", counts.len(), v.len(), args[0]);
        }
        "dense" => {
            if args.len() != 3 {
                return usage();
            }
            let v = DenseVocab::load_file(&args[0]).unwrap_or_else(|e| die(e));
            let ids = read_ids_file(&args[1]);
            let d = v.encode_dense(&ids);
            let unk = d.iter().filter(|&&x| x == thpp::train::DENSE_UNK).count();
            tht::write_npy_u32_file(&args[2], &d, &[d.len()]).unwrap_or_else(|e| die(e));
            eprintln!("n={} unk={} vocab={} -> {}", d.len(), unk, v.len(), args[2]);
        }
        "undense" => {
            if args.len() != 3 {
                return usage();
            }
            let v = DenseVocab::load_file(&args[0]).unwrap_or_else(|e| die(e));
            let d = read_ids_file(&args[1]);
            let ids = v.decode_dense(&d);
            write_ids_file(&args[2], &ids, &o, 0);
        }
        "factors" => {
            if args.len() != 2 {
                return usage();
            }
            let ids = read_ids_file(&args[0]);
            let mut f = Vec::with_capacity(ids.len() * 4);
            for &id in &ids {
                f.extend_from_slice(&factorize(id).as_array());
            }
            tht::write_npy_u32_file(&args[1], &f, &[ids.len(), 4]).unwrap_or_else(|e| die(e));
        }
        "bench" => {
            let lang = args.first().cloned().unwrap_or_else(|| "es".into());
            let r0 = rss_kb();
            let t0 = Instant::now();
            let e = Thpp::eager();
            let le = t0.elapsed();
            let r1 = rss_kb();
            let t0 = Instant::now();
            let l = Thpp::lazy().with_config(o.cfg);
            l.prepare_all();
            let la = t0.elapsed();
            let r2 = rss_kb();
            println!("load eager es/en {:?} (+{} KB RSS); lazy 31 idiomas {:?} (+{} KB RSS)", le, r1.0 - r0.0, la, r2.0 - r1.0);
            let text = match args.get(1) {
                Some(p) => std::fs::read_to_string(p).unwrap_or_else(|e| die(e)),
                None => synthetic(&l, &lang, 8_000_000),
            };
            let _ = e;
            for th in [1usize, o.threads] {
                let t0 = Instant::now();
                let ids = l.encode_parallel(&text, &lang, th).unwrap();
                let ee = t0.elapsed();
                let t0 = Instant::now();
                let back = l.decode(&ids);
                let de = t0.elapsed();
                let mb = text.len() as f64 / 1e6;
                println!(
                    "[{lang}] {mb:.1} MB, hilos={th}: encode {:.0} MB/s, decode {:.0} MB/s, ids={} ({:.2} bytes/id), roundtrip={}",
                    mb / ee.as_secs_f64(),
                    mb / de.as_secs_f64(),
                    ids.len(),
                    text.len() as f64 / ids.len() as f64,
                    back == text
                );
            }
            println!("RSS {} KB, pico {} KB", rss_kb().0, rss_kb().1);
        }
        _ => usage(),
    }
}
