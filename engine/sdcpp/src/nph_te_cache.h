// NPH 0.29 text-encoder output cache (ZIT-HEXAGON, MIT).
// RAM: LRU of at most kRamEntries conditionings (exact key match, values copied => byte-identical).
// Disk (optional): one file per key under <NPH_CACHE_DIR>/te/, holding the full key (exact-match
// check), the serialized tensors and a checksum. Written atomically (tmp + rename); any mismatch or
// short read is a miss. Total size capped (oldest files removed first).
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "core/tensor.hpp"

namespace nph_te {

constexpr size_t kRamEntries = 4;
constexpr const char* kFormat = "NPHTE029";  // bump when the payload layout changes

// ---- generic LRU (RAM) ------------------------------------------------------------------------
template <class V>
class Lru {
public:
    explicit Lru(size_t cap = kRamEntries) : cap_(cap) {}
    bool get(const std::string& key, V& out) {
        for (size_t i = 0; i < items_.size(); ++i) {
            if (items_[i].first == key) {
                auto it = std::move(items_[i]);
                items_.erase(items_.begin() + i);
                items_.push_back(std::move(it));
                out = items_.back().second;
                return true;
            }
        }
        return false;
    }
    void put(const std::string& key, const V& v) {
        for (size_t i = 0; i < items_.size(); ++i) {
            if (items_[i].first == key) {
                items_.erase(items_.begin() + i);
                break;
            }
        }
        while (!items_.empty() && items_.size() >= cap_) items_.erase(items_.begin());
        if (cap_ > 0) items_.emplace_back(key, v);
    }
    void clear() { items_.clear(); }
    size_t size() const { return items_.size(); }
    bool contains(const std::string& key) const {
        for (const auto& it : items_)
            if (it.first == key) return true;
        return false;
    }

private:
    size_t cap_;
    std::vector<std::pair<std::string, V>> items_;
};

// ---- serialization ----------------------------------------------------------------------------
struct Writer {
    std::string buf;
    void u8(uint8_t v) { buf.push_back((char)v); }
    void u64(uint64_t v) { buf.append((const char*)&v, 8); }
    void raw(const void* p, size_t n) { buf.append((const char*)p, n); }
    template <class T>
    void tensor(const sd::Tensor<T>& t) {
        u64(t.shape().size());
        for (int64_t d : t.shape()) u64((uint64_t)d);
        u64((uint64_t)t.numel());
        if (t.numel() > 0) raw(t.data(), sizeof(T) * (size_t)t.numel());
    }
};

struct Reader {
    const uint8_t* p;
    size_t n;
    size_t off = 0;
    bool ok    = true;
    Reader(const void* d, size_t sz) : p((const uint8_t*)d), n(sz) {}
    bool take(void* dst, size_t k) {
        if (!ok || k > n - off) return ok = false;
        memcpy(dst, p + off, k);
        off += k;
        return true;
    }
    uint8_t u8() { uint8_t v = 0; take(&v, 1); return v; }
    uint64_t u64() { uint64_t v = 0; take(&v, 8); return v; }
    template <class T>
    bool tensor(sd::Tensor<T>& t) {
        uint64_t nd = u64();
        if (!ok || nd > 8) return ok = false;
        std::vector<int64_t> shape(nd);
        uint64_t prod = 1;
        for (auto& d : shape) {
            d = (int64_t)u64();
            if (d < 0) return ok = false;
            prod *= (uint64_t)d;
        }
        uint64_t ne = u64();
        if (!ok || (nd > 0 && ne != prod) || (nd == 0 && ne != 0) || ne > (n - off) / sizeof(T)) return ok = false;
        if (nd == 0) {
            t = sd::Tensor<T>();
            return true;
        }
        std::vector<T> data((size_t)ne);
        if (ne > 0 && !take(data.data(), sizeof(T) * (size_t)ne)) return false;
        t = sd::Tensor<T>(std::move(shape), std::move(data));
        return true;
    }
    bool done() const { return ok && off == n; }
};


// ---- condition / embeds (templated so the unit test can use a mock with the same fields) ------
template <class C>
void ser_cond(Writer& w, const C& c) {
    w.tensor(c.c_crossattn);
    w.tensor(c.c_vector);
    w.tensor(c.c_concat);
    w.tensor(c.c_t5_ids);
    w.tensor(c.c_t5_weights);
    w.tensor(c.c_input_ids);
    w.tensor(c.c_position_ids);
    w.tensor(c.c_token_types);
    w.tensor(c.c_vinput_mask);
    w.u64(c.c_image_embeds.size());
    for (const auto& ie : c.c_image_embeds) {
        w.u64((uint64_t)(int64_t)ie.first);
        w.tensor(ie.second);
    }
    w.u64(c.extra_c_crossattns.size());
    for (const auto& t : c.extra_c_crossattns) w.tensor(t);
}
template <class C>
bool deser_cond(Reader& r, C& c) {
    r.tensor(c.c_crossattn);
    r.tensor(c.c_vector);
    r.tensor(c.c_concat);
    r.tensor(c.c_t5_ids);
    r.tensor(c.c_t5_weights);
    r.tensor(c.c_input_ids);
    r.tensor(c.c_position_ids);
    r.tensor(c.c_token_types);
    r.tensor(c.c_vinput_mask);
    uint64_t n = r.u64();
    if (!r.ok || n > 64) return false;
    for (uint64_t i = 0; i < n; ++i) {
        int idx = (int)(int64_t)r.u64();
        sd::Tensor<float> t;
        if (!r.tensor(t)) return false;
        c.c_image_embeds.emplace_back(idx, std::move(t));
    }
    n = r.u64();
    if (!r.ok || n > 64) return false;
    for (uint64_t i = 0; i < n; ++i) {
        sd::Tensor<float> t;
        if (!r.tensor(t)) return false;
        c.extra_c_crossattns.push_back(std::move(t));
    }
    return r.ok;
}
// Conditions carrying reference images/audio/blocks are never written to disk.
template <class C>
bool cond_disk_ok(const C& c) {
    return c.c_ref_images.empty() && c.c_ref_audios.empty() && c.c_reference_blocks.empty();
}
template <class E>
bool serialize_embeds(const E& e, std::string& out) {
    if (!cond_disk_ok(e.cond) || !cond_disk_ok(e.uncond) || !cond_disk_ok(e.img_uncond)) return false;
    Writer w;
    w.raw(kFormat, 8);
    ser_cond(w, e.cond);
    ser_cond(w, e.uncond);
    ser_cond(w, e.img_uncond);
    out = std::move(w.buf);
    return true;
}
template <class E>
bool deserialize_embeds(const std::string& in, E& e) {
    Reader r(in.data(), in.size());
    char magic[8];
    if (!r.take(magic, 8) || memcmp(magic, kFormat, 8) != 0) return false;
    E out;
    if (!deser_cond(r, out.cond) || !deser_cond(r, out.uncond) || !deser_cond(r, out.img_uncond) || !r.done()) return false;
    e = std::move(out);
    return true;
}

uint64_t hash64(const void* data, size_t size, uint64_t seed);

// Identity of a model file: "size:mtime_ns:hash(first 4 MiB + last 1 MiB)". Memoized per
// (path,size,mtime) so repeated generations do not re-read the file. "" when it cannot be read.
std::string file_fingerprint(const std::string& path);

// Disk store. `dir` is the te/ directory (created on demand).
bool disk_load(const std::string& dir, const std::string& key, std::string& payload);
bool disk_store(const std::string& dir, const std::string& key, const std::string& payload, uint64_t cap_bytes);
std::string disk_path(const std::string& dir, const std::string& key);
uint64_t disk_size(const std::string& dir);
int disk_clear(const std::string& dir);

// Counters for the report.
struct Stats {
    uint64_t ram_hits = 0, disk_hits = 0, misses = 0, disk_stores = 0, disk_rejects = 0;
    uint64_t disk_load_us = 0, disk_store_us = 0;
    std::string last;  // last event, human readable
};
Stats stats();
void note(const char* what, uint64_t us = 0);

}  // namespace nph_te
