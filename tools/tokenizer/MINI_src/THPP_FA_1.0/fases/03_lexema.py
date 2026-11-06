# fase 3 — cubo SS/BBB y sitio de expansión
# La expansión no está en este script. Está en src/lib.rs:
#   Thpp::new          fase 3c  dispara es y en ya
#   Dict::load         fase 3b  directo + rev
#   Map::build         fase 3   data/hppt_es.txt | hppt_en.txt -> arena + slot

def pack(ll, i):
    ss = 10 + i // 1000
    bbb = i % 1000
    return ll * 10_000_000 + 10 * 100_000 + ss * 1000 + bbb

assert pack(1, 0) == 11010000
assert pack(1, 1000) == 11011000
assert pack(2, 0) == 21010000
print("i=0", pack(1, 0), "i=1000", pack(1, 1000))
print("expansion: src/lib.rs Map::build, disparada por Thpp::new")
print("fase3 ok")
