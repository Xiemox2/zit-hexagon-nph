#!/bin/sh
cd "$(dirname "$0")"
{ echo '// generated from npg_kernels.cl by gen_kernels.sh — MIT'; echo 'namespace npg { extern const char * npg_kernel_src; const char * npg_kernel_src = R"NPGCL('; cat npg_kernels.cl; echo ')NPGCL"; }'; } > npg_kernels_src.cpp
