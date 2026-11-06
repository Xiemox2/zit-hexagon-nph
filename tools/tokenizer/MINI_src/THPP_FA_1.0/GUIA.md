# THPP_FA 1.0 — fases

Solo es/en. Los IDs son los de 0.9. Esta guía dice qué es cada pieza y dónde se expande el diccionario.

## Fase 1 — IDs

Archivo: `src/lib.rs`, constantes de arriba.

| id | rol |
|---|---|
| 0 | SEP, marca de deletreo. Decode no escribe |
| 11 | NL |
| 110 | TAB |
| 111 | PAD |
| 1001 | SPACE, un token por cada espacio |
| 1101 | TITLE, mayúscula inicial del siguiente lexema |
| 1102 | UPPER, todo el lexema en mayúscula |
| 1201 | LANG es |
| 1202 | LANG en |

Letra: `id = LL * 10000 + CCC`. `a` es CCC 10, así que `a` es = 10010. El 11 está reservado a NL, por eso `b` es 12.

Lexema: `id = LL * 10_000_000 + 1_000_000 + SS * 1_000 + BBB`, con SS = 10 + i/1000 y BBB = i % 1000.
Primer lema es = 11010000. El i=1000 salta a 11011000.

Comprobar: `python3 fases/01_ids.py`

## Fase 2 — letra

Archivo: `src/chars.rs`, `enc` y `dec_core`.
Latino, dígitos y puntuación corta viven ahí. THPP no tiene `enc_ext`.
OOV no busca lexema: empuja SEP y un `pack_letter` por glifo.

Comprobar: `python3 fases/02_letra.py`

## Fase 3 — aquí se expande el diccionario

No se expande en encode. Se expande al construir.

1. `Thpp::new()` en `src/lib.rs` (fase 3c). Carga ya es y en. No es perezoso.
2. `Dict::load` (fase 3b) llama a `Map::build` y luego recorre otra vez el txt para el mapa inverso `rev`.
3. `Map::build` (fase 3) es el sitio exacto:
   - lee `data/hppt_es.txt` o `data/hppt_en.txt`
   - `split_whitespace`, una palabra por línea
   - pasa a minúsculas
   - si ya está, la salta
   - asigna `pack_lex(ll, i % 1000, 10 + i/1000)` y sube `i`
   - mete los bytes en `arena` y el índice en `slot[]`

Después de `new()`, el dict no crece. Encode solo hace lookup.

es = 49901 lemas. en = 49039.

Comprobar: `python3 fases/03_lexema.py`

## Fase 4 — cinta

`Thpp::encode` en `src/lib.rs`.
Orden: LANG, luego el texto.

- ` ` -> 1001, uno por carácter
- `\n` -> 11
- `\t` -> 110
- run ASCII alfabético: lookup en minúsculas. Hit: TITLE o UPPER si toca, luego el id. Miss: SEP + letras
- si el run ASCII sigue en UTF-8 (`niño`), se junta el lexema entero antes del lookup
- dígito o punct: CCC de `chars::enc`, sin LL

`encode("el la", "es")` = `[1201, 11010000, 1001, 11010001]`

Comprobar: `python3 fases/04_cinta.py`

## Fase 5 — decode

`Thpp::decode`. Ignora 1201/1202 y SEP. SPACE/NL/TAB vuelven a carácter. 8 dígitos buscan en `rev`. 5 dígitos son letra.

Comprobar: `python3 fases/05_roundtrip.py`

## Qué no es

No es tiktoken. No parte subpalabras. No entrena. FI es el mismo código sin esta guía ni `fases/`.
