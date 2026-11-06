# fase 5 — decode
SPACE, LANG_ES = 1001, 1201
REV = {11010000: "el", 11010001: "la"}

def decode(ids):
    s = []
    for t in ids:
        if t == LANG_ES:
            continue
        if t == SPACE:
            s.append(" ")
            continue
        s.append(REV[t])
    return "".join(s)

ids = [1201, 11010000, 1001, 11010001]
back = decode(ids)
print(back)
assert back == "el la"
print("fase5 ok")
