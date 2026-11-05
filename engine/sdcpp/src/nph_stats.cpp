#include "nph_stats.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>

static std::atomic<uint64_t> g_sd_ctr[NPH_SD_COUNT];
static std::atomic<nph_mark_cb_t> g_mark_cb{nullptr};
static std::atomic<const nph_hex_hooks*> g_hooks{nullptr};

uint64_t nph_now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void nph_sd_add(int idx, uint64_t v) {
    if (idx >= 0 && idx < NPH_SD_COUNT) {
        g_sd_ctr[idx].fetch_add(v, std::memory_order_relaxed);
    }
}

void nph_sd_counters(uint64_t* out, int n) {
    for (int i = 0; i < n && i < NPH_SD_COUNT; ++i) {
        out[i] = g_sd_ctr[i].load(std::memory_order_relaxed);
    }
}

void nph_set_mark_cb(nph_mark_cb_t cb) { g_mark_cb.store(cb); }

void nph_mark(const char* name) {
    auto cb = g_mark_cb.load();
    if (cb != nullptr) {
        cb(name);
    }
}

void nph_set_hex_hooks(const nph_hex_hooks* hooks) { g_hooks.store(hooks); }
const nph_hex_hooks* nph_get_hex_hooks() { return g_hooks.load(); }

bool nph_env_flag(const char* name, bool def) {
    const char* v = getenv(name);
    if (v == nullptr || *v == 0) {
        return def;
    }
    return !(strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "off") == 0);
}

#include <mutex>
#include <string>
#include <utility>
#include <vector>

static std::mutex g_npg_mtx;
static std::vector<std::pair<std::string, std::string>> g_npg_notes;

void nph_npg_note(const char* key, const char* value) {
    if (key == nullptr || value == nullptr) return;
    std::lock_guard<std::mutex> lk(g_npg_mtx);
    for (auto& kv : g_npg_notes) {
        if (kv.first == key) {
            kv.second = value;
            return;
        }
    }
    if (g_npg_notes.size() < 64) g_npg_notes.emplace_back(key, value);
}

void nph_npg_notes(char* out, size_t cap) {
    if (out == nullptr || cap == 0) return;
    std::string s;
    {
        std::lock_guard<std::mutex> lk(g_npg_mtx);
        for (auto& kv : g_npg_notes) s += kv.first + ": " + kv.second + "\n";
    }
    size_t n = s.size() < cap - 1 ? s.size() : cap - 1;
    memcpy(out, s.data(), n);
    out[n] = 0;
}

void nph_npg_notes_clear() {
    std::lock_guard<std::mutex> lk(g_npg_mtx);
    g_npg_notes.clear();
}
