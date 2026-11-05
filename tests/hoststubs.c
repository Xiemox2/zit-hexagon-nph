/* static bionic has no dladdr; ggml only uses it for backtraces */
int dladdr(const void* a, void* b) { (void)a; (void)b; return 0; }
