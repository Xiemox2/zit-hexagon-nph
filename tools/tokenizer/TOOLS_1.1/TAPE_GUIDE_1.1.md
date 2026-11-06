# THPP 1.1 — tape guide (English)

THPP 1.1 merges HPPT 1.0 and THPP 1.0 into one crate, `thpp`, with one type, `Thpp`, and two modes:

| Constructor | Behaviour |
|---|---|
| `Thpp::eager()` | THPP 1.0: es/en built in `new`, only es/en enabled |
| `Thpp::lazy()` / `Thpp::new()` | HPPT 1.0: 31 languages, each dictionary built on first use |

The tape is the same as 0.9 / 0.3 / 1.0. **No existing id changed.** The word lists are the same files. They are frozen with FNV-1a-64 checksums (`data/CHECKSUMS.txt`, `FROZEN_LIST_CHECKSUMS`, `lists_checksum()`). New ids live only in ranges that 1.0 never used.

`Config::compat()` is the default. For any input that 1.0 encoded losslessly, the 1.1 tape is identical to the 1.0 tape, id for id. Where 1.0 lost information (PAD, '�', mixed case, NFD, `\r`, emoji, …), 1.1 uses the new ids and round-trips exactly.

`Config::v11()` adds:
- hyphen/apostrophe compounds,
- longest-match CJK,
- Turkish-first casing,
- a VER id after LANG.

It changes tapes, so it is opt-in. `nfc_input` is a separate opt-in.

## 1. Tape

```
LANG [VER] [FORM] [TITLE|UPPER|TITLE_L|UPPER_L | MIXED CASE…] LEXEME
     SPACE LEXEME [SEP LETTER LETTER …] PUNCT [CP] [BYTE]
```

### Fixed ids (unchanged)

| Name | Id |
|---|---|
| SEP | 0 |
| NL | 11 |
| TAB | 110 |
| PAD | 111 (never emitted by 1.1; still decoded as '�') |
| SPACE | 1001 |
| TITLE | 1101 |
| UPPER | 1102 |
| LANG | 1200+LL (es 1201 … bn 1231) |

### New ids in 1.1

| Name | Id / range | Meaning |
|---|---|---|
| MIXED | 1103 | followed by CASE ids, then the lexeme (iPhone, McDonald) |
| TITLE_L | 1104 | Title-case with Turkish rules (i→İ); tr only (LL 14) |
| UPPER_L | 1105 | UPPER with Turkish rules |
| FORM_NFC | 1106 | next lexeme is rendered NFC |
| FORM_NFD | 1107 | next lexeme is rendered NFD |
| VER | 1300+v (1.1 = 1311) | optional, right after LANG |
| CASE | 4,000,000 + mask16 | bit k = char k uppercase; one id per 16 chars |
| CP | 5,000,000 + codepoint (…6,114,111) | raw codepoint: emoji, `\r`, NBSP, symbols, letters with no CCC |
| BYTE | 6,200,000 + byte (…6,200,255) | invalid UTF-8 byte; only `encode_bytes` / `decode_bytes` |

Letters are unchanged: `LL*10000 + CCC`. CCC 300–439 (Cyrillic, Arabic, CJK punctuation; `。` = 436) are now emitted by encode and decoded in every language.

Lexemes are unchanged: `LL*10^7 + 10*10^5 + SS*1000 + BBB`, with `SS = 10 + i/1000` and `BBB = i % 1000`.

## 2. Dictionary

`src/dict.rs` `Dict::load` uses the same algorithm as 1.0 `Map::build`:
1. `split_whitespace`.
2. Lowercase.
3. Skip duplicates.
4. Assign `pack_lex(LL, i%1000, 10+i/1000)`.

Non-NFC entries also get an NFC alias with the **same id**, used only for lookup. The reverse lookup uses index `i = (SS-10)*1000 + BBB`, so it needs no HashMap.

In lazy mode, a language is built on its first `encode`, `decode`, or `prepare(lang)`. `prepare_all()` builds all 31.

`decode` infers the language from each lexeme id and builds that dictionary on demand. HPPT 1.0 instead printed '�' for languages it had not loaded.

## 3. Encode

1. Pure-ASCII runs use the 1.0 fast path. A hit gives TITLE/UPPER, or MIXED+CASE for mixed-case words.
2. CJK uses one glyph per lookup. With v11 it tries the longest listed run first. A miss gives SEP + letter, or CP; never PAD.
3. Alphabetic runs (letters plus combining marks) are lowercased and looked up. A lexeme is emitted **only if rendering it reproduces the input exactly**. The order tried is:
   1. the 1.0 marker (keeps 1.0 tapes identical);
   2. FORM × {none, TITLE, UPPER, TITLE_L, UPPER_L};
   3. MIXED+CASE.

   If none of these work, the run becomes SEP + letters, and any letter without a CCC becomes CP.
4. Turkish tries i/İ and ı/I as well. v11 tries them first. v11 also tries compounds (`a-b`, `l'x`, up to 4 joins, longest first).
5. Digits and punctuation use short CCC ids, otherwise CP.
6. `encode_bytes` sends invalid UTF-8 bytes to BYTE.
7. `encode_parallel(text, lang, threads)` splits at `\n` and gives the same tape as `encode`.

## 4. Decode

- LANG and VER are ignored.
- Markers apply to the next lexeme.
- CP gives that codepoint. BYTE gives the byte with `decode_bytes`, or '�' with `decode`.

Guarantees:
- `decode(encode(s)) == s` byte-for-byte for every `&str`.
- `decode_bytes(encode_bytes(b)) == b` for every byte string.
- With `nfc_input`, the round trip is exact only up to NFC.

## 5. Phases (`fases/`)

These are Rust examples: `cargo run --release --example fase01_ids` … `fase09_denso`.

| Example | What it shows |
|---|---|
| 01–05 | the 1.0 phases (ids, letters, lexeme buckets, tape, round trip) |
| 06 | new markers |
| 07 | CP/BYTE |
| 08 | `.tht`/`.npy` |
| 09 | dense vocab and factors |

## 6. Tools

`thpp-tool` keeps the 1.0 contract exactly:
- `ids`
- `encode`
- `decode`
- `encode -`
- `decode -`
- `file-encode`
- `file-decode`

New commands and options:

| Command / option | What it does |
|---|---|
| `.tht`/`.npy` input and output | format chosen by file extension |
| `verify` | byte-exact round trip check |
| `stats` | tape statistics |
| `corpus` | many documents in parallel, plus `out.offsets.npy` |
| `vocab`, `dense`, `undense` | contiguous indices for training |
| `factors` | (N,4) npy of kind, lang, bucket, slot |
| `bench` | encode/decode benchmark |
| `--v11`, `--nfc`, `--eager`, `--threads N` | options |

### `.tht` header (32 bytes, little endian), then u32 LE ids

| Offset | Field |
|---|---|
| 0 | magic `THPT` |
| 4 | format u16 = 1 |
| 6 | spec u16 = 11 |
| 8 | flags u32: bit0 VER, bit1 nfc_input, bit2 compounds, bit3 cjk_longest, bit4 locale_first, bit5 multi-doc |
| 12 | LL u32 |
| 16 | lists checksum u64 |
| 24 | count u64 |

## 7. Training API (`thpp::train`)

- `classify`, `factorize(id) -> Factors{kind, lang, bucket, slot}`.
- `DenseVocab`:
  - Layout: 0 = PAD, 1 = UNK, then all reserved ids, then corpus ids by frequency.
  - Methods: `from_counts`, `to_dense`, `to_id`, `encode_dense`, `decode_dense`, `save`/`load`.

## 8. Caveats

- 1.0 decoders do not understand the new ids; they would print '�'.
- The list entries `²` and `nº` can never be matched (they are not alphabetic runs).
- hi/ko/th lists are still seeds.
- No 0.9/0.3 tapes were available. Compatibility is tested against tapes produced by the 1.0 crates (`tests/golden/`).
