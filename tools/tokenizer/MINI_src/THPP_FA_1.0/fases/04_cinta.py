# fase 4 — encode de "el la" con dict fijo
SPACE, LANG_ES = 1001, 1201
FWD = {"el": 11010000, "la": 11010001}

def encode(text):
    out = [LANG_ES]
    for w in text.split(" "):
        if out[-1] != LANG_ES:
            out.append(SPACE)
        out.append(FWD[w])
    return out

ids = encode("el la")
print(ids)
assert ids == [1201, 11010000, 1001, 11010001]
print("fase4 ok")
