// NPH 0.30: CombinedLoraAdapter - all standard LoRAs of one target Linear packed into one HMX matmul
// pair (see nph_lora_core.h for the math and the exactness rules).
#ifndef __SD_MODEL_ADAPTER_LORA_COMBINED_HPP__
#define __SD_MODEL_ADAPTER_LORA_COMBINED_HPP__

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <dirent.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include "model/adapter/lora.hpp"
#include "model/adapter/nph_lora_core.h"

struct CombinedLoraAdapter : public WeightAdapter {
    struct Entry {
        ggml_tensor* A = nullptr;
        ggml_tensor* B = nullptr;
        ggml_tensor* s = nullptr;
        int R = 0, Rp = 0;
        int gain = 0;  // pow2 gain for scaled Linears (see nph_lora::Combined::gain)
    };
    struct FileInfo {
        std::string name;
        std::string dtype;
        int targets = 0;
        int rank_min = 0, rank_max = 0;
        size_t fallback_tensors = 0;
        float multiplier = 1.f;
    };

    std::unordered_map<std::string, Entry> entries;  // key: model weight name
    std::vector<std::shared_ptr<LoraModel>> models;
    std::shared_ptr<MultiLoraAdapter> fallback;
    bool fallback_empty = true;
    std::vector<std::unordered_map<std::string, ggml_tensor*>> removed;  // per model
    ggml_context* ctx                 = nullptr;
    ggml_backend_buffer_t buffer      = nullptr;
    std::vector<FileInfo> files;
    nph_lora::Stats st;
    int64_t rank_total = 0, rank_padded = 0;
    int Rp_min = 0, Rp_max = 0;
    int64_t parts_combined = 0;  // (LoRA, part) pairs folded in = matmul pairs the per-LoRA path would run
    double build_ms = 0;
    std::string key;
    // 0.30.1 on-load safety check (combined vs per-LoRA delta*x, random x, FP64 on the packed data)
    int check_targets = 0, check_fallbacks = 0;
    double check_max_rel = 0;
    bool check_from_cache = false;
    // 0.30.1 disk cache of the packed A/B/s
    std::string cache_status = "off";
    double cache_ms = 0;
    double cache_mb = 0;

    static constexpr const char* kCacheMagic = "NPHLC002";
    static constexpr const char* kCacheEnd   = "NPHLCEND";

    static uint64_t fnv1a(const std::string& v) {
        uint64_t h = 1469598103934665603ull;
        for (unsigned char ch : v) { h ^= ch; h *= 1099511628211ull; }
        return h;
    }
    static std::string cache_dir() {
        const char* on = getenv("NPH_LORA_CACHE");
        if (on != nullptr && atoi(on) == 0) return "";
        const char* d = getenv("NPH_CACHE_DIR");
        if (d == nullptr || d[0] == 0) return "";
        std::string dir = std::string(d) + "/lora";
        mkdir(d, 0700);
        mkdir(dir.c_str(), 0700);
        struct stat sb;
        if (stat(dir.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) return "";
        return dir;
    }
    // Keep only the newest `keep` cache files (one per LoRA set + strengths).
    static void cache_evict(const std::string& dir, const std::string& keep_path) {
        int keep = 2;
        if (const char* e = getenv("NPH_LORA_CACHE_KEEP")) keep = std::max(1, atoi(e));
        DIR* dd = opendir(dir.c_str());
        if (!dd) return;
        std::vector<std::pair<time_t, std::string>> files;
        while (dirent* de = readdir(dd)) {
            std::string n = de->d_name;
            if (n.size() < 7 || (n.compare(n.size() - 6, 6, ".nphlc") != 0 && n.find(".nphlc.tmp") == std::string::npos)) continue;
            std::string full = dir + "/" + n;
            struct stat sb;
            if (stat(full.c_str(), &sb) != 0) continue;
            if (n.find(".tmp") != std::string::npos) { unlink(full.c_str()); continue; }  // interrupted write
            files.push_back({sb.st_mtime, full});
        }
        closedir(dd);
        std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        int others = 0;
        for (auto& f : files) {
            if (f.second == keep_path) continue;
            if (others < keep - 1) { others++; continue; }
            unlink(f.second.c_str());
        }
    }
    template <typename T> static bool wr(FILE* f, const T& v) { return fwrite(&v, sizeof(T), 1, f) == 1; }
    template <typename T> static bool rd(FILE* f, T& v) { return fread(&v, sizeof(T), 1, f) == 1; }
    static bool wr_str(FILE* f, const std::string& v) { uint32_t n = (uint32_t)v.size(); return wr(f, n) && (n == 0 || fwrite(v.data(), 1, n, f) == n); }
    static bool rd_str(FILE* f, std::string& v, uint32_t max_len) {
        uint32_t n = 0;
        if (!rd(f, n) || n > max_len) return false;
        v.resize(n);
        return n == 0 || fread(&v[0], 1, n, f) == n;
    }

    ~CombinedLoraAdapter() override {
        restore();
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }

    // Put the LoRA tensors we took over back into the LoraModels (they may be reused by a rebuild).
    void restore() {
        for (size_t i = 0; i < models.size() && i < removed.size(); i++) {
            for (auto& kv : removed[i]) models[i]->lora_tensors[kv.first] = kv.second;
            removed[i].clear();
        }
    }

    size_t buffer_bytes() const { return buffer ? ggml_backend_buffer_get_size(buffer) : 0; }

    static bool parse_index(const std::string& key, std::string& base, int& idx) {
        size_t dot = key.rfind('.');
        if (dot == std::string::npos || dot + 1 >= key.size()) return false;
        for (size_t i = dot + 1; i < key.size(); i++)
            if (key[i] < '0' || key[i] > '9') return false;
        base = key.substr(0, dot);
        idx  = atoi(key.c_str() + dot + 1);
        return true;
    }

    static bool write_entry_stats(FILE* f, const nph_lora::Stats& st) {
        return wr(f, st.values) && wr(f, st.inexact_values) && wr(f, st.max_abs_err) && wr(f, st.shifted_vectors) && wr(f, st.split_vectors);
    }
    static bool read_entry_stats(FILE* f, nph_lora::Stats& st) {
        return rd(f, st.values) && rd(f, st.inexact_values) && rd(f, st.max_abs_err) && rd(f, st.shifted_vectors) && rd(f, st.split_vectors);
    }
    bool write_cache(const std::string& path, const std::string& ckey,
                     const std::vector<std::pair<std::string, nph_lora::Combined>>& packed) const {
        const std::string tmp = path + ".tmp";
        FILE* f = fopen(tmp.c_str(), "wb");
        if (!f) return false;
        bool ok = fwrite(kCacheMagic, 1, 8, f) == 8 && wr_str(f, ckey) && wr(f, (uint32_t)packed.size()) &&
                  wr(f, (int32_t)check_targets) && wr(f, (int32_t)check_fallbacks) && wr(f, check_max_rel);
        for (auto& [name, c] : packed) {
            if (!ok) break;
            ok = wr_str(f, name) && wr(f, (int32_t)c.K) && wr(f, (int32_t)c.N) && wr(f, (int32_t)c.R) && wr(f, (int32_t)c.Rp) &&
                 wr(f, (int32_t)c.gain) && write_entry_stats(f, c.st) &&
                 fwrite(c.A.data(), sizeof(ggml_fp16_t), c.A.size(), f) == c.A.size() &&
                 fwrite(c.B.data(), sizeof(ggml_fp16_t), c.B.size(), f) == c.B.size() &&
                 fwrite(c.s.data(), sizeof(float), c.s.size(), f) == c.s.size();
        }
        ok = ok && fwrite(kCacheEnd, 1, 8, f) == 8;
        ok = (fflush(f) == 0) && ok;
        ok = (fclose(f) == 0) && ok;
        if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
            unlink(tmp.c_str());
            return false;
        }
        return true;
    }
    // Exact key match required; every entry must be a known target with the same K/N.
    template <typename TargetMap>
    bool read_cache(const std::string& path, const std::string& ckey, const TargetMap& targets,
                    std::vector<std::pair<std::string, nph_lora::Combined>>& packed) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        char mg[8];
        std::string k;
        uint32_t n = 0;
        int32_t ct = 0, cf = 0;
        double cm = 0;
        bool ok = fread(mg, 1, 8, f) == 8 && memcmp(mg, kCacheMagic, 8) == 0 && rd_str(f, k, 1u << 24) && k == ckey &&
                  rd(f, n) && n <= targets.size() && rd(f, ct) && rd(f, cf) && rd(f, cm);
        std::vector<std::pair<std::string, nph_lora::Combined>> out;
        for (uint32_t i = 0; ok && i < n; i++) {
            std::string name;
            int32_t K, N, R, Rp, gain;
            nph_lora::Combined c;
            ok = rd_str(f, name, 4096) && rd(f, K) && rd(f, N) && rd(f, R) && rd(f, Rp) && rd(f, gain) && read_entry_stats(f, c.st);
            if (!ok) break;
            auto t = targets.find(name);
            ok = t != targets.end() && t->second.K == K && t->second.N == N && Rp > 0 && Rp % 32 == 0 && R <= Rp && Rp <= 65536;
            if (!ok) break;
            c.K = K; c.N = N; c.R = R; c.Rp = Rp; c.gain = gain;
            c.A.resize((size_t)K * Rp);
            c.B.resize((size_t)Rp * N);
            c.s.resize((size_t)Rp);
            ok = fread(c.A.data(), sizeof(ggml_fp16_t), c.A.size(), f) == c.A.size() &&
                 fread(c.B.data(), sizeof(ggml_fp16_t), c.B.size(), f) == c.B.size() &&
                 fread(c.s.data(), sizeof(float), c.s.size(), f) == c.s.size();
            if (ok) out.emplace_back(name, std::move(c));
        }
        ok = ok && fread(mg, 1, 8, f) == 8 && memcmp(mg, kCacheEnd, 8) == 0;
        fclose(f);
        if (!ok) return false;
        packed           = std::move(out);
        check_targets    = ct;
        check_fallbacks  = cf;
        check_max_rel    = cm;
        check_from_cache = true;
        return true;
    }

    // Returns nullptr when nothing could be combined (caller keeps the plain MultiLoraAdapter).
    static std::shared_ptr<CombinedLoraAdapter> build(const std::vector<std::shared_ptr<LoraModel>>& lora_models,
                                                      const std::map<std::string, ggml_tensor*>& weights,
                                                      ggml_backend_t backend,
                                                      const ModelLoader& loader,
                                                      const std::string& cache_key) {
        auto t0      = std::chrono::steady_clock::now();
        int max_split = 3;
        if (const char* e = getenv("NPH_LORA_SPLIT")) max_split = std::max(1, std::min(6, atoi(e)));

        struct PartRef {
            int model;
            std::string down, up;  // lora_tensors keys
            std::string scale_key; // alpha or scale key ("" = none)
            ggml_tensor* tdown;
            ggml_tensor* tup;
            float scale;
            int r, n;
        };
        struct Target {
            std::string name;
            int K = 0, N = 0;
            std::vector<PartRef> parts;
        };
        std::map<std::string, Target> targets;
        auto self = std::make_shared<CombinedLoraAdapter>();
        self->models = lora_models;
        self->removed.resize(lora_models.size());
        self->key = cache_key;
        self->files.resize(lora_models.size());

        for (size_t mi = 0; mi < lora_models.size(); mi++) {
            auto& m = lora_models[mi];
            auto& fi = self->files[mi];
            fi.name = m->file_path.substr(m->file_path.find_last_of('/') == std::string::npos ? 0 : m->file_path.find_last_of('/') + 1);
            fi.multiplier = m->multiplier;
            std::set<std::string> seen;
            for (auto& kv : m->lora_tensors) {
                const std::string& n = kv.first;
                static const std::string pre = "lora.", suf = ".lora_down";
                if (n.size() <= pre.size() + suf.size() || n.compare(0, pre.size(), pre) != 0 ||
                    n.compare(n.size() - suf.size(), suf.size(), suf) != 0)
                    continue;
                std::string key = n.substr(pre.size(), n.size() - pre.size() - suf.size());
                std::string T   = key;
                int idx         = 0;
                if (weights.find(T) == weights.end()) {
                    std::string base;
                    if (!parse_index(key, base, idx) || weights.find(base) == weights.end()) continue;
                    T = base;
                }
                (void)idx;
                if (!seen.insert(T).second) continue;
                ggml_tensor* W = weights.at(T);
                if (ggml_n_dims(W) != 2) continue;
                const int K = (int)W->ne[0], N = (int)W->ne[1];
                // Walk parts exactly like LoraModel::get_out_diff (stops at first missing index).
                std::vector<PartRef> parts;
                bool ok = true;
                int rows = 0;
                for (int i = 0;; i++) {
                    std::string k  = i == 0 ? T : T + "." + std::to_string(i);
                    std::string dn = "lora." + k + ".lora_down", up = "lora." + k + ".lora_up";
                    auto lt        = [&](const std::string& s) { return m->lora_tensors.count("lora." + k + s) != 0; };
                    if (lt(".lokr_w1") || lt(".lokr_w1_a")) { ok = false; break; }  // LoKr part: whole target stays per-LoRA
                    auto d = m->lora_tensors.find(dn), u = m->lora_tensors.find(up);
                    if (d == m->lora_tensors.end() || u == m->lora_tensors.end()) break;
                    if (lt(".lora_mid")) { ok = false; break; }
                    ggml_tensor *td = d->second, *tu = u->second;
                    if (ggml_n_dims(td) > 2 || ggml_n_dims(tu) > 2 || td->ne[0] != K || tu->ne[0] != td->ne[1]) { ok = false; break; }
                    PartRef p;
                    p.model = (int)mi; p.down = dn; p.up = up; p.tdown = td; p.tup = tu;
                    p.r = (int)td->ne[1]; p.n = (int)tu->ne[1];
                    float sv = 1.f;
                    auto sc = m->lora_tensors.find("lora." + k + ".scale");
                    if (sc != m->lora_tensors.end()) { sv = m->scalar_value(sc->second); p.scale_key = sc->first; }
                    else {
                        auto al = m->lora_tensors.find("lora." + k + ".alpha");
                        if (al != m->lora_tensors.end()) { sv = m->scalar_value(al->second) / (float)p.r; p.scale_key = al->first; }
                    }
                    p.scale = sv * m->multiplier;
                    if (!m->nph_source_names.count(td) || !m->nph_source_names.count(tu)) { ok = false; break; }
                    rows += p.n;
                    parts.push_back(p);
                }
                if (!ok || parts.empty() || rows != N) continue;
                auto& tg = targets[T];
                tg.name = T; tg.K = K; tg.N = N;
                for (auto& p : parts) tg.parts.push_back(p);
            }
        }
        if (targets.empty()) return nullptr;

        // 0.30.1 disk cache: key = format/core version + split setting + every LoRA file (path, size,
        // mtime ns, strength) + the exact target/part plan (names, dims, ranks, alpha/rank*strength).
        std::vector<std::pair<std::string, nph_lora::Combined>> packed;
        std::string cache_path, ckey;
        bool from_cache = false;
        const std::string cdir = cache_dir();
        if (!cdir.empty()) {
            ckey = std::string(kCacheMagic) + "|nph-0.30.1|split=" + std::to_string(max_split) + "\n";
            bool stat_ok = true;
            for (auto& m : lora_models) {
                struct stat sb;
                if (stat(m->file_path.c_str(), &sb) != 0) { stat_ok = false; break; }
                char buf[160];
                snprintf(buf, sizeof(buf), "|%lld|%lld.%09ld|%a\n", (long long)sb.st_size, (long long)sb.st_mtim.tv_sec,
                         (long)sb.st_mtim.tv_nsec, (double)m->multiplier);
                ckey += m->file_path + buf;
            }
            for (auto& kv : targets) {
                ckey += kv.first + "|" + std::to_string(kv.second.K) + "x" + std::to_string(kv.second.N);
                for (auto& p : kv.second.parts) {
                    char buf[96];
                    snprintf(buf, sizeof(buf), "|%d:%d:%d:%a", p.model, p.r, p.n, (double)p.scale);
                    ckey += buf;
                }
                ckey += "\n";
            }
            if (stat_ok) {
                char hn[40];
                snprintf(hn, sizeof(hn), "/%016llx.nphlc", (unsigned long long)fnv1a(ckey));
                cache_path = cdir + hn;
                auto tc0   = std::chrono::steady_clock::now();
                if (self->read_cache(cache_path, ckey, targets, packed)) {
                    from_cache = true;
                    self->cache_status = "hit";
                    struct stat sb;
                    if (stat(cache_path.c_str(), &sb) == 0) self->cache_mb = sb.st_size / 1048576.0;
                    utimes(cache_path.c_str(), nullptr);  // LRU
                    // targets missing from the cache failed combine/check when it was written
                    std::set<std::string> have;
                    for (auto& e : packed) have.insert(e.first);
                    for (auto& kv : targets) if (!have.count(kv.first)) kv.second.parts.clear();
                } else {
                    packed.clear();
                    self->cache_status = "miss";
                }
                self->cache_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0).count();
            } else {
                self->cache_status = "off(stat)";
            }
        }

        // Load + pack in batches (bounded host RAM: only one batch of F32 LoRA data at a time).
        if (!from_cache) packed.reserve(targets.size());
        std::vector<std::string> order;
        if (!from_cache) for (auto& kv : targets) order.push_back(kv.first);
        const size_t batch = 48;
        // on-load safety check: ~10 targets spread over the model (+ every scaled-Linear-like w2)
        const size_t check_every = std::max<size_t>(1, order.size() / 10);
        double check_thr = 1e-4;
        if (const char* e = getenv("NPH_LORA_CHECK_TOL")) check_thr = atof(e);
        const bool check_on = getenv("NPH_LORA_CHECK") == nullptr || atoi(getenv("NPH_LORA_CHECK")) != 0;
        for (size_t b0 = 0; b0 < order.size(); b0 += batch) {
            size_t b1 = std::min(order.size(), b0 + batch);
            std::vector<std::unordered_map<std::string, std::vector<float>>> bufs(lora_models.size());
            std::vector<std::unordered_map<std::string, ggml_tensor*>> dsts(lora_models.size());
            std::vector<ggml_context*> tctx(lora_models.size(), nullptr);
            for (size_t mi = 0; mi < lora_models.size(); mi++) {
                ggml_init_params ip = {ggml_tensor_overhead() * 2 * batch * 8 + 1024, nullptr, true};
                tctx[mi]            = ggml_init(ip);
            }
            for (size_t t = b0; t < b1; t++) {
                for (auto& p : targets[order[t]].parts) {
                    auto& m = lora_models[p.model];
                    for (ggml_tensor* src : {p.tdown, p.tup}) {
                        const std::string& sname = m->nph_source_names.at(src);
                        if (dsts[p.model].count(sname)) continue;
                        auto& v = bufs[p.model][sname];
                        v.resize((size_t)ggml_nelements(src));
                        ggml_tensor* d = ggml_new_tensor(tctx[p.model], GGML_TYPE_F32, ggml_n_dims(src), src->ne);
                        d->data        = v.data();
                        dsts[p.model][sname] = d;
                    }
                }
            }
            bool load_ok = true;
            for (size_t mi = 0; mi < lora_models.size() && load_ok; mi++) {
                if (dsts[mi].empty()) continue;
                std::set<std::string> names;
                for (auto& kv : dsts[mi]) names.insert(kv.first);
                const auto& dm = dsts[mi];
                auto cb = [&dm](const TensorStorage& src, ggml_tensor** dst) {
                    auto f = dm.find(src.name);
                    *dst   = f == dm.end() ? nullptr : f->second;
                    return true;
                };
                load_ok = loader.load_file_tensors(lora_models[mi]->source_file, lora_models[mi]->source_version, cb, names);
            }
            if (!load_ok) {
                for (auto* c : tctx) ggml_free(c);
                LOG_WARN("nph-lora: combined build failed to read LoRA tensors; using per-LoRA path");
                return nullptr;
            }
            for (size_t t = b0; t < b1; t++) {
                auto& tg = targets[order[t]];
                std::vector<nph_lora::PartSrc> srcs;
                // row offsets per model restart at 0 (each model covers the whole N with its parts)
                std::vector<int> off(lora_models.size(), 0);
                for (auto& p : tg.parts) {
                    auto& m = lora_models[p.model];
                    nph_lora::PartSrc s;
                    s.K = tg.K; s.r = p.r; s.n = p.n; s.row_off = off[p.model];
                    off[p.model] += p.n;
                    s.down  = bufs[p.model][m->nph_source_names.at(p.tdown)].data();
                    s.up    = bufs[p.model][m->nph_source_names.at(p.tup)].data();
                    s.scale = p.scale;
                    srcs.push_back(s);
                }
                nph_lora::Combined c;
                std::string err;
                if (!nph_lora::combine(srcs, tg.K, tg.N, max_split, c, &err)) {
                    LOG_WARN("nph-lora: combine %s failed (%s); per-LoRA path for it", tg.name.c_str(), err.c_str());
                    tg.parts.clear();
                    continue;
                }
                if (check_on && (t % check_every == 0 || (tg.name.find("w2.") != std::string::npos && t % (check_every / 2 + 1) == 0))) {
                    const double rel = nph_lora::check_packed(srcs, c, 4, (uint32_t)(t * 2654435761u));
                    self->check_targets++;
                    self->check_max_rel = std::max(self->check_max_rel, std::isfinite(rel) ? rel : 1e30);
                    if (!(rel <= check_thr)) {
                        self->check_fallbacks++;
                        LOG_WARN("nph-lora-check: %s combined vs per-LoRA rel err %.3g > %.1g; per-LoRA path for it", tg.name.c_str(), rel, check_thr);
                        tg.parts.clear();
                        continue;
                    }
                }
                packed.emplace_back(tg.name, std::move(c));
            }
            for (auto* c : tctx) ggml_free(c);
        }
        if (packed.empty()) return nullptr;
        if (!from_cache && !cache_path.empty()) {
            auto tc0 = std::chrono::steady_clock::now();
            const bool wok = self->write_cache(cache_path, ckey, packed);
            self->cache_status = wok ? "miss, written" : "miss, write failed";
            self->cache_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0).count();
            struct stat sb;
            if (wok && stat(cache_path.c_str(), &sb) == 0) self->cache_mb = sb.st_size / 1048576.0;
            cache_evict(cdir, cache_path);
        }

        ggml_init_params ip = {ggml_tensor_overhead() * (packed.size() * 3 + 8), nullptr, true};
        self->ctx           = ggml_init(ip);
        for (auto& [name, c] : packed) {
            Entry e;
            e.A = ggml_new_tensor_2d(self->ctx, GGML_TYPE_F16, c.K, c.Rp);
            e.B = ggml_new_tensor_2d(self->ctx, GGML_TYPE_F16, c.Rp, c.N);
            e.s = ggml_new_tensor_1d(self->ctx, GGML_TYPE_F32, c.Rp);
            ggml_set_name(e.A, "nphL_wA");
            ggml_set_name(e.B, "nphL_wB");
            ggml_set_name(e.s, "nphL_ws");
            e.R = c.R; e.Rp = c.Rp; e.gain = c.gain;
            self->entries[name] = e;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        self->buffer = ggml_backend_alloc_ctx_tensors_from_buft(self->ctx, buft);
        if (self->buffer == nullptr) {
            LOG_WARN("nph-lora: combined buffer alloc failed; using per-LoRA path");
            self->entries.clear();
            return nullptr;
        }
        ggml_backend_buffer_set_usage(self->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto& [name, c] : packed) {
            auto& e = self->entries[name];
            ggml_backend_tensor_set(e.A, c.A.data(), 0, ggml_nbytes(e.A));
            ggml_backend_tensor_set(e.B, c.B.data(), 0, ggml_nbytes(e.B));
            ggml_backend_tensor_set(e.s, c.s.data(), 0, ggml_nbytes(e.s));
            self->st.values += c.st.values;
            self->st.inexact_values += c.st.inexact_values;
            self->st.max_abs_err = std::max(self->st.max_abs_err, c.st.max_abs_err);
            self->st.shifted_vectors += c.st.shifted_vectors;
            self->st.split_vectors += c.st.split_vectors;
            self->rank_total += c.R;
            self->rank_padded += c.Rp;
            self->Rp_min = self->Rp_min == 0 ? c.Rp : std::min(self->Rp_min, c.Rp);
            self->Rp_max = std::max(self->Rp_max, c.Rp);
            std::vector<ggml_fp16_t>().swap(c.A);
            std::vector<ggml_fp16_t>().swap(c.B);
        }
        // Take the combined LoRA tensors away from the per-LoRA path (restored on rebuild/destruct).
        for (auto& [name, c] : packed) {
            self->parts_combined += (int64_t)targets[name].parts.size();
            for (auto& p : targets[name].parts) {
                auto& m  = lora_models[p.model];
                auto& rm = self->removed[p.model];
                auto& fi = self->files[p.model];
                for (const std::string* k : {&p.down, &p.up, &p.scale_key}) {
                    if (k->empty()) continue;
                    auto f = m->lora_tensors.find(*k);
                    if (f == m->lora_tensors.end()) continue;
                    rm[*k] = f->second;
                    m->lora_tensors.erase(f);
                }
                if (p.down == "lora." + name + ".lora_down") {
                    fi.targets++;
                    fi.rank_min = fi.rank_min == 0 ? p.r : std::min(fi.rank_min, p.r);
                    fi.rank_max = std::max(fi.rank_max, p.r);
                    if (fi.dtype.empty()) fi.dtype = ggml_type_name(p.tdown->type);
                }
            }
        }
        std::vector<std::shared_ptr<LoraModel>> rest;
        for (size_t mi = 0; mi < lora_models.size(); mi++) {
            size_t left = 0;
            for (auto& kv : lora_models[mi]->lora_tensors) {
                const std::string& n = kv.first;
                if (n.size() > 6 && (n.compare(n.size() - 6, 6, ".alpha") == 0 || n.compare(n.size() - 6, 6, ".scale") == 0)) continue;
                left++;
            }
            self->files[mi].fallback_tensors = left;
            if (left > 0) rest.push_back(lora_models[mi]);
        }
        self->fallback       = std::make_shared<MultiLoraAdapter>(rest);
        self->fallback_empty = rest.empty();
        self->build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return self;
    }

    void log_summary() const {
        for (auto& f : files) {
            LOG_INFO("nph-lora-file: %s mult=%.3f dtype=%s rank=%d-%d targets=%d fallback_tensors=%zu", f.name.c_str(), f.multiplier,
                     f.dtype.empty() ? "?" : f.dtype.c_str(), f.rank_min, f.rank_max, f.targets, f.fallback_tensors);
        }
        LOG_INFO("nph-lora-combined: loras=%zu targets=%zu rank_sum=%lld rank_padded=%lld Rp=%d-%d buf_mb=%.1f shifted=%d split=%d inexact=%lld maxerr=%.3g build_ms=%.0f matmuls_per_dit_pass=%zu (per-LoRA path: %lld) fallback=%s",
                 files.size(), entries.size(), (long long)rank_total, (long long)rank_padded, Rp_min, Rp_max,
                 buffer_bytes() / 1048576.0, st.shifted_vectors, st.split_vectors, (long long)st.inexact_values, st.max_abs_err,
                 build_ms, entries.size() * 2, (long long)(parts_combined * 2), fallback_empty ? "none" : "per-lora");
        LOG_INFO("nph-lora-check: targets=%d max_rel_err=%.3g fallbacks=%d%s | disk-cache=%s %.0f MB %.0f ms",
                 check_targets, check_max_rel, check_fallbacks, check_from_cache ? " (checked when the cache was built)" : "",
                 cache_status.c_str(), cache_mb, cache_ms);
    }

    const Entry* find(const std::string& weight_name) const {
        auto f = entries.find(weight_name);
        return f == entries.end() ? nullptr : &f->second;
    }

    ggml_tensor* apply(ggml_context* c, ggml_tensor* x_s, const Entry& e, const ForwardParams& fp, ggml_tensor* out) {
        return nph_lora::graph_apply(c, x_s, e.A, e.B, e.s, fp.linear.scale, fp.linear.force_prec_f32, out, e.gain);
    }

    bool nph_prescaled(const std::string& prefix) override { return find(prefix + "weight") != nullptr; }

    int nph_defer_gain(const std::string& prefix) override {
        static const bool on = getenv("NPH_LORA_DEFER") == nullptr || atoi(getenv("NPH_LORA_DEFER")) != 0;
        if (!on || !fallback_empty) return -1;
        const Entry* e = find(prefix + "weight");
        return e ? e->gain : -1;
    }

    ggml_tensor* nph_add_deferred(ggml_context* c, ggml_tensor* x_s, ggml_tensor* output_scaled, const std::string& prefix,
                                  ForwardParams fp) override {
        const Entry* e = find(prefix + "weight");
        if (e == nullptr) return output_scaled;
        return nph_lora::graph_apply_deferred(c, x_s, e->A, e->B, e->s, fp.linear.force_prec_f32, output_scaled, e->gain);
    }

    ggml_tensor* nph_add_prescaled(ggml_context* c, ggml_backend_t backend, ggml_tensor* x, ggml_tensor* x_s, ggml_tensor* w,
                                   ggml_tensor* output, const std::string& prefix, ForwardParams fp) override {
        if (!fallback_empty) output = fallback->add_lora_to_output(c, backend, x, w, output, prefix, fp);
        const Entry* e = find(prefix + "weight");
        return e ? apply(c, x_s, *e, fp, output) : output;
    }

    ggml_tensor* patch_weight(ggml_context* c, ggml_backend_t backend, ggml_tensor* weight, const std::string& name) override {
        weight         = fallback->patch_weight(c, backend, weight, name);
        const Entry* e = find(name);
        if (e == nullptr) return weight;
        // Rare path (weight used outside Linear::forward): merge delta = (A*diag(s)) x B in FP32.
        ggml_tensor* As    = ggml_mul(c, ggml_ext_cast_f32(c, backend, e->A), ggml_reshape_2d(c, e->s, 1, e->Rp));
        ggml_tensor* Bf    = ggml_ext_cast_f32(c, backend, e->B);
        ggml_tensor* delta = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, As)), Bf);
        ggml_mul_mat_set_prec(delta, GGML_PREC_F32);
        if (weight->type != GGML_TYPE_F32) weight = ggml_ext_cast_f32(c, backend, weight);
        return ggml_add(c, weight, delta);
    }

    ggml_tensor* forward_with_lora(ggml_context* c, ggml_backend_t backend, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                                   const std::string& prefix, ForwardParams fp) override {
        const Entry* e = fp.op_type == ForwardParams::op_type_t::OP_LINEAR ? find(prefix + "weight") : nullptr;
        if (e == nullptr) return fallback->forward_with_lora(c, backend, x, w, b, prefix, fp);
        ggml_tensor* x_s = fp.linear.scale != 1.f ? ggml_ext_scale(c, x, fp.linear.scale) : x;
        ggml_tensor* out;
        if (fallback_empty) {
            if (b) b = fallback->patch_weight(c, backend, b, prefix + "bias");
            out = ggml_ext_linear_prescaled(c, x_s, w, b, fp.linear.force_prec_f32, fp.linear.scale);
        } else {
            out = fallback->forward_with_lora(c, backend, x, w, b, prefix, fp);
        }
        return apply(c, x_s, *e, fp, out);
    }

    ggml_tensor* add_lora_to_output(ggml_context* c, ggml_backend_t backend, ggml_tensor* x, ggml_tensor* w, ggml_tensor* output,
                                    const std::string& prefix, ForwardParams fp) override {
        output         = fallback->add_lora_to_output(c, backend, x, w, output, prefix, fp);
        const Entry* e = find(prefix + "weight");
        if (e == nullptr || fp.op_type != ForwardParams::op_type_t::OP_LINEAR) return output;
        ggml_tensor* x_s = fp.linear.scale != 1.f ? ggml_ext_scale(c, x, fp.linear.scale) : x;
        return apply(c, x_s, *e, fp, output);
    }

    size_t get_extra_graph_size() override { return fallback->get_extra_graph_size() + entries.size() * 12; }
};

#endif
