// NPG bench — weight-buffer split planning (pure, unit-tested on the host). MIT License.
#pragma once
#include <cstddef>
#include <vector>

namespace npg {

struct Chunk { size_t begin = 0, end = 0, bytes = 0; };   // tensors [begin, end), total padded bytes

// Greedy split of an ordered list of (already aligned) tensor sizes into consecutive chunks of <= max_bytes each,
// the same policy as ggml_backend_alloc_ctx_tensors_from_buft. Returns false (and *bad = index) if a single
// tensor is larger than max_bytes (it can never fit one backend buffer).
inline bool plan_chunks(const std::vector<size_t> & sizes, size_t max_bytes, std::vector<Chunk> & out, size_t * bad = nullptr) {
    out.clear();
    if (max_bytes == 0) { if (bad) *bad = 0; return false; }
    Chunk cur;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] > max_bytes) { if (bad) *bad = i; out.clear(); return false; }
        if (cur.end > cur.begin && cur.bytes + sizes[i] > max_bytes) {
            out.push_back(cur);
            cur = Chunk();
            cur.begin = cur.end = i;
        }
        cur.end = i + 1;
        cur.bytes += sizes[i];
    }
    if (cur.end > cur.begin) out.push_back(cur);
    return true;
}

inline size_t pad_to(size_t x, size_t a) { return a ? (x + a - 1) / a * a : x; }

}  // namespace npg
