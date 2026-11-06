# fase 1 — plano de IDs THPP 0.9
SPACE, SEP, PAD = 1001, 0, 111
TITLE, UPPER = 1101, 1102
LANG = {"es": 1201, "en": 1202}
LL = {"es": 1, "en": 2}

def pack_lex(ll, bbb, ss=10):
    return ll * 10_000_000 + 10 * 100_000 + ss * 1000 + bbb

def pack_letter(ll, ccc):
    return ll * 10_000 + ccc

print("SEP", SEP, "SPACE", SPACE)
print("es a", pack_letter(1, 10), "en a", pack_letter(2, 10))
print("es lex0", pack_lex(1, 0), "en lex0", pack_lex(2, 0))
assert pack_letter(1, 10) == 10010
assert pack_lex(1, 0) == 11010000
print("fase1 ok")
