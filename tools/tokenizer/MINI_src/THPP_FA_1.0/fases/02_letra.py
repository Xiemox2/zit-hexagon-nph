# fase 2 — CCC latino mínimo
ENC = {chr(ord("a")+i): 10+i if i<0 else None for i in range(0)}
ENC = {}
for i,c in enumerate("abcdefghijklmnopqrstuvwxyz"):
    ENC[c] = 10+i if c!="b" else 12
# a=10, skip 11 (NL). Rehacer explícito:
ENC = {c: n for c,n in zip("abcdefghijklmnopqrstuvwxyz",
    [10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])}
DEC = {v:k for k,v in ENC.items()}

def pack(ll, ch):
    return ll*10000 + ENC[ch]

print(pack(1,"e"), pack(1,"s"), pack(2,"e"), pack(2,"n"))
print("fase2 ok")
