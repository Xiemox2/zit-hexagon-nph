#!/usr/bin/env python3
# Rebuild ggml-htp skels (v79/v81) with the exact flags of the original build-android trees
# (compile_commands.json + link rule), into sdbench/build-skel/<arch>/. MIT (project glue).
import json, os, subprocess, sys, concurrent.futures as cf
ROOT="./build-android/sdcpp/ggml/src/ggml-hexagon"
OUT="./build-skel"
T="${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools/21.0.04/Tools"
archs=sys.argv[1:] or ["v79","v81"]
for a in archs:
    src=f"{ROOT}/htp-{a}-prefix/src/htp-{a}-build"
    od=f"{OUT}/{a}"; os.makedirs(od+"/obj",exist_ok=True)
    cmds=json.load(open(src+"/compile_commands.json"))
    objs=[]; jobs=[]
    for c in cmds:
        f=c["file"]; o=f"{od}/obj/"+os.path.basename(f)+".obj"; objs.append(o)
        cmd=c["command"]
        i=cmd.index(" -o "); j=cmd.index(" -c ")
        cmd=cmd[:i]+" -o "+o+cmd[j:]
        jobs.append(cmd)
    def run(cmd):
        r=subprocess.run(cmd,shell=True,cwd=src,capture_output=True,text=True)
        return r.returncode,cmd,r.stderr
    with cf.ThreadPoolExecutor(8) as ex:
        for rc,cmd,err in ex.map(run,jobs):
            if rc: print(err[-3000:]); sys.exit(1)
    so=f"{od}/libggml-htp-{a}.so"
    link=(f"{T}/bin/hexagon-clang -mcpu={a} -m{a} -mhvx={a} -mhmx -G0 -fpic -Wl,-Bsymbolic "
          f"-Wl,-L{T}/target/hexagon/lib/{a}/G0/pic -Wl,-L{T}/target/hexagon/lib/ -Wl,--no-threads "
          "-Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign "
          f"-shared -o {so} -Wl,-soname,libggml-htp-{a}.so -shared -Wl,--start-group "+" ".join(objs)+" -Wl,--end-group -lc")
    r=subprocess.run(link,shell=True,cwd=src,capture_output=True,text=True)
    if r.returncode: print(r.stderr[-3000:]); sys.exit(1)
    print(a, so, os.path.getsize(so))
