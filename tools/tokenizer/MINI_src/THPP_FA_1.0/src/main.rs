use thpp::{pack_letter, pack_lex, Thpp, SEP, SPACE, PAD, NL, TAB, TITLE, UPPER};

fn main() {
    let t = Thpp::new();
    println!("THPP 1.0  es={} en={}", t.size("es"), t.size("en"));
    println!("SPACE={SPACE} SEP={SEP} PAD={PAD} NL={NL} TAB={TAB} TITLE={TITLE} UPPER={UPPER}");
    println!("pack_letter es a {}", pack_letter(1, 'a'));
    println!("pack_lex es0 {}", pack_lex(1, 0, 10));
    let ids = t.encode("el la", "es");
    println!("el la -> {ids:?}");
    println!("decode -> {:?}", t.decode(&ids));
}
