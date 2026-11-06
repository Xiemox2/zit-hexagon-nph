import numpy as np, pyopencl as cl, ml_dtypes, sys
src = open('./native/npg_kernels.cl').read()
ctx = cl.create_some_context(interactive=False); q = cl.CommandQueue(ctx)
print(ctx.devices[0].extensions[:200])
rng = np.random.default_rng(0)
M,N,K = 64,128,256
X = rng.standard_normal((M,K)).astype(np.float32)
Wf = np.clip(rng.standard_normal((N,K))*0.06,-1,1).astype(np.float32)
W8 = Wf.astype(ml_dtypes.float8_e4m3fn); Wd = W8.astype(np.float64)
ref = X.astype(np.float64) @ Wd.T
mf = cl.mem_flags
for wt, wdata, alpha, refm in [('WT_F8', W8.view(np.uint8), 256.0, ref), ('WT_F16', Wd.astype(np.float16), 1.0, ref), ('WT_F32', Wf, 1.0, X.astype(np.float64)@Wf.astype(np.float64).T)]:
  for tm,tn in [(4,4),(8,4),(4,8)]:
    prg = cl.Program(ctx, src).build(options=f'-D{wt} -DTM={tm} -DTN={tn}')
    bx = cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=X); bw = cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(wdata))
    by = cl.Buffer(ctx, mf.READ_WRITE, M*N*4)
    for kn in ['gemm','gemm_tiled']:
      k = getattr(prg, kn)
      args=(bx,bw,by,np.int32(M),np.int32(N),np.int32(K),np.float32(alpha),np.int32(0),np.int32(0))
      if kn=='gemm': k(q,(N//tn,M//tm),None,*args)
      else: k(q,(((N+16*tn-1)//(16*tn))*16,((M+16*tm-1)//(16*tm))*16),(16,16),*args)
      Y=np.empty((M,N),np.float32); cl.enqueue_copy(q,Y,by)
      print(wt,tm,tn,kn,'relL2',np.linalg.norm(Y-refm)/np.linalg.norm(refm))
# accumulate
prg = cl.Program(ctx, src).build(options='-DWT_F16')
H,S=2,64
Qh=rng.standard_normal((H,S,128)).astype(np.float32); Kh=rng.standard_normal((H,S,128)).astype(np.float16); Vh=rng.standard_normal((H,S,128)).astype(np.float16)
sc=1/np.sqrt(128)
bq=cl.Buffer(ctx,mf.COPY_HOST_PTR,hostbuf=Qh); bk=cl.Buffer(ctx,mf.COPY_HOST_PTR,hostbuf=Kh); bv=cl.Buffer(ctx,mf.COPY_HOST_PTR,hostbuf=Vh); bo=cl.Buffer(ctx,mf.READ_WRITE,Qh.nbytes)
prg.flash_attn(q,(S//16*64,H),(64,1),bq,bk,bv,bo,np.int32(S),np.float32(sc))
O=np.empty_like(Qh); cl.enqueue_copy(q,O,bo)
s=np.einsum('hqd,hkd->hqk',Qh.astype(np.float64),Kh.astype(np.float64))*sc; p=np.exp(s-s.max(-1,keepdims=True)); p/=p.sum(-1,keepdims=True)
Or=np.einsum('hqk,hkd->hqd',p,Vh.astype(np.float64)); print('FA relL2',np.linalg.norm(O-Or)/np.linalg.norm(Or))
C=512; R=8
Xr=rng.standard_normal((R,C)).astype(np.float32); w=rng.standard_normal(C).astype(np.float32); s1=rng.standard_normal(C).astype(np.float32)*.1; s2=rng.standard_normal(C).astype(np.float32)
bufs=[cl.Buffer(ctx,mf.COPY_HOST_PTR,hostbuf=a) for a in (Xr,w,s1,s2)]; by=cl.Buffer(ctx,mf.READ_WRITE,Xr.nbytes)
prg.rmsnorm_mod(q,(R*256,),(256,),*bufs,by,np.int32(C),np.float32(1e-6)); Y=np.empty_like(Xr); cl.enqueue_copy(q,Y,by)
Yr=Xr/np.sqrt((Xr**2).mean(-1,keepdims=True)+1e-6)*w*(1+s1)+s2; print('rms relL2',np.linalg.norm(Y-Yr)/np.linalg.norm(Yr))
