use thpp::{pack_letter, pack_lex, Thpp, MIXED, NL, PAD, SEP, SPACE, TAB, TITLE, UPPER, VER_1_1};

fn main() {
    let t = Thpp::eager();
    println!("THPP {} (eager es/en)  es={} en={}", thpp::VERSION, t.size("es"), t.size("en"));
    println!("SPACE={SPACE} SEP={SEP} PAD={PAD} NL={NL} TAB={TAB} TITLE={TITLE} UPPER={UPPER} MIXED={MIXED} VER={VER_1_1}");
    println!("pack_letter es a {}", pack_letter(1, 'a'));
    println!("pack_lex es0 {}", pack_lex(1, 0, 10));
    println!("pack_lex gl0 {}", pack_lex(10, 0, 10));
    let ids = t.encode("el la", "es");
    println!("el la -> {ids:?}");
    println!("decode -> {:?}", t.decode(&ids));
    let l = Thpp::lazy();
    let ids = l.encode("iPhone 😀 привет。", "ru");
    println!("lazy ru 'iPhone 😀 привет。' -> {ids:?} -> {:?}", l.decode(&ids));
}
