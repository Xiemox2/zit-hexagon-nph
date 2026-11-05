// Unit tests for the NPH 0.29 TE cache (host build, no model). ZIT-HEXAGON, MIT.
#include "nph_te_cache.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>

static int g_fail = 0;
#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            g_fail++;                                                  \
        }                                                              \
    } while (0)

struct MockBlock {};
struct Cond {
    sd::Tensor<float> c_crossattn, c_vector, c_concat;
    sd::Tensor<int32_t> c_t5_ids;
    sd::Tensor<float> c_t5_weights;
    sd::Tensor<int32_t> c_input_ids, c_position_ids, c_token_types, c_vinput_mask;
    std::vector<std::pair<int, sd::Tensor<float>>> c_image_embeds;
    std::vector<sd::Tensor<float>> c_ref_images, c_ref_audios;
    std::vector<MockBlock> c_reference_blocks;
    std::vector<sd::Tensor<float>> extra_c_crossattns;
};
struct Embeds {
    Cond cond, uncond, img_uncond;
};

template <class T>
static sd::Tensor<T> rnd(std::vector<int64_t> shape, std::mt19937& g) {
    sd::Tensor<T> t(shape);
    for (int64_t i = 0; i < t.numel(); ++i) {
        if constexpr (std::is_floating_point<T>::value) {
            uint32_t bits = g();  // arbitrary bit patterns incl. NaN payloads / denormals / -0
            memcpy(&t.data()[i], &bits, 4);
        } else {
            t.data()[i] = (T)g();
        }
    }
    return t;
}
template <class T>
static bool same(const sd::Tensor<T>& a, const sd::Tensor<T>& b) {
    return a.shape() == b.shape() && a.numel() == b.numel() &&
           (a.numel() == 0 || memcmp(a.data(), b.data(), sizeof(T) * a.numel()) == 0);
}
static bool same_cond(const Cond& a, const Cond& b) {
    bool ok = same(a.c_crossattn, b.c_crossattn) && same(a.c_vector, b.c_vector) && same(a.c_concat, b.c_concat) &&
              same(a.c_t5_ids, b.c_t5_ids) && same(a.c_t5_weights, b.c_t5_weights) && same(a.c_input_ids, b.c_input_ids) &&
              same(a.c_position_ids, b.c_position_ids) && same(a.c_token_types, b.c_token_types) &&
              same(a.c_vinput_mask, b.c_vinput_mask) && a.c_image_embeds.size() == b.c_image_embeds.size() &&
              a.extra_c_crossattns.size() == b.extra_c_crossattns.size();
    for (size_t i = 0; ok && i < a.c_image_embeds.size(); ++i)
        ok = a.c_image_embeds[i].first == b.c_image_embeds[i].first && same(a.c_image_embeds[i].second, b.c_image_embeds[i].second);
    for (size_t i = 0; ok && i < a.extra_c_crossattns.size(); ++i) ok = same(a.extra_c_crossattns[i], b.extra_c_crossattns[i]);
    return ok;
}

int main() {
    std::mt19937 g(1234);
    // 1) LRU: bounded to 4, LRU eviction, refresh on get, values copied.
    {
        nph_te::Lru<int> lru(4);
        for (int i = 0; i < 4; ++i) lru.put("k" + std::to_string(i), i);
        int v = -1;
        CHECK(lru.size() == 4);
        CHECK(lru.get("k0", v) && v == 0);  // k0 now most recent
        lru.put("k4", 4);                   // evicts k1
        CHECK(lru.size() == 4);
        CHECK(!lru.get("k1", v));
        CHECK(lru.get("k0", v) && v == 0);
        CHECK(lru.get("k4", v) && v == 4);
        lru.put("k4", 44);  // overwrite, no growth
        CHECK(lru.size() == 4 && lru.get("k4", v) && v == 44);
        for (int i = 0; i < 100; ++i) lru.put("x" + std::to_string(i), i);
        CHECK(lru.size() == 4);
        CHECK(nph_te::kRamEntries == 4);
    }
    // 2) Serialization round trip is byte-identical (Z-Image like: [1, 77..512, 2560] + empty fields).
    Embeds e;
    e.cond.c_crossattn = rnd<float>({1, 300, 2560}, g);
    e.cond.c_input_ids = rnd<int32_t>({1, 300}, g);
    e.cond.c_image_embeds.emplace_back(3, rnd<float>({2, 5}, g));
    e.cond.extra_c_crossattns.push_back(rnd<float>({1, 7, 9}, g));
    e.uncond.c_crossattn = rnd<float>({1, 12, 2560}, g);
    e.uncond.c_vector    = rnd<float>({1, 1, 0}, g);  // zero-size with shape
    std::string blob;
    CHECK(nph_te::serialize_embeds(e, blob));
    Embeds d;
    CHECK(nph_te::deserialize_embeds(blob, d));
    CHECK(same_cond(e.cond, d.cond) && same_cond(e.uncond, d.uncond) && same_cond(e.img_uncond, d.img_uncond));
    CHECK(d.uncond.c_vector.shape() == std::vector<int64_t>({1, 1, 0}));
    CHECK(d.img_uncond.c_crossattn.empty() && d.img_uncond.c_crossattn.shape().empty());
    // truncated / extended blobs are rejected
    {
        Embeds x;
        CHECK(!nph_te::deserialize_embeds(blob.substr(0, blob.size() - 1), x));
        CHECK(!nph_te::deserialize_embeds(blob + "z", x));
        CHECK(!nph_te::deserialize_embeds(std::string("garbage!") + blob.substr(8), x));
    }
    // ref images are never serialized
    {
        Embeds r = e;
        r.cond.c_ref_images.push_back(rnd<float>({1, 3}, g));
        std::string b2;
        CHECK(!nph_te::serialize_embeds(r, b2));
    }
    // 3) Disk store / load: exact key, corruption, cap eviction, clear.
    char tmpl[] = "/tmp/nphte_test_XXXXXX";
    std::string root = mkdtemp(tmpl);
    std::string dir  = root + "/te";
    const std::string key1 = "prompt A\x1fneg\x1f" "1024x1024|te#123";
    const std::string key2 = "prompt B\x1fneg\x1f" "1024x1024|te#123";
    CHECK(nph_te::disk_store(dir, key1, blob, 64ull << 20));
    std::string got;
    CHECK(nph_te::disk_load(dir, key1, got) && got == blob);
    CHECK(!nph_te::disk_load(dir, key2, got));
    {
        Embeds x;
        CHECK(nph_te::disk_load(dir, key1, got) && nph_te::deserialize_embeds(got, x) && same_cond(x.cond, e.cond));
    }
    // flip one payload byte -> rejected and removed
    {
        std::string p = nph_te::disk_path(dir, key1);
        FILE* f       = fopen(p.c_str(), "r+b");
        fseek(f, -100, SEEK_END);
        int c = fgetc(f);
        fseek(f, -100, SEEK_END);
        fputc(c ^ 0x40, f);
        fclose(f);
        CHECK(!nph_te::disk_load(dir, key1, got));
        struct stat st{};
        CHECK(stat(p.c_str(), &st) != 0);
        CHECK(nph_te::stats().disk_rejects == 1);
    }
    // truncated file -> rejected
    {
        CHECK(nph_te::disk_store(dir, key1, blob, 64ull << 20));
        std::string p = nph_te::disk_path(dir, key1);
        CHECK(truncate(p.c_str(), 1000) == 0);
        CHECK(!nph_te::disk_load(dir, key1, got));
    }
    // cap: 3 entries of ~3 MB in a 7 MB cap -> oldest evicted
    {
        CHECK(nph_te::disk_clear(dir) >= 0);
        std::vector<std::string> keys;
        for (int i = 0; i < 3; ++i) {
            keys.push_back("cap-key-" + std::to_string(i));
            CHECK(nph_te::disk_store(dir, keys.back(), blob, 7ull << 20));
            usleep(20000);
        }
        CHECK(nph_te::disk_size(dir) <= (7ull << 20));
        CHECK(!nph_te::disk_load(dir, keys[0], got));
        CHECK(nph_te::disk_load(dir, keys[2], got) && got == blob);
        CHECK(!nph_te::disk_store(dir, "too-big", blob, 1000));  // larger than cap: skipped
        CHECK(nph_te::disk_clear(dir) >= 1 && nph_te::disk_size(dir) == 0);
    }
    // 4) File fingerprint changes with content/mtime, stable otherwise.
    {
        std::string fpath = root + "/te_model.bin";
        FILE* f           = fopen(fpath.c_str(), "wb");
        std::vector<char> data(6 << 20, 1);
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
        std::string a = nph_te::file_fingerprint(fpath);
        CHECK(!a.empty() && a == nph_te::file_fingerprint(fpath));
        usleep(20000);
        f = fopen(fpath.c_str(), "r+b");
        fseek(f, 10, SEEK_SET);
        fputc(7, f);
        fclose(f);
        std::string b = nph_te::file_fingerprint(fpath);
        CHECK(!b.empty() && a != b);
        CHECK(nph_te::file_fingerprint(root + "/missing").empty());
        unlink(fpath.c_str());
    }
    rmdir(dir.c_str());
    rmdir(root.c_str());
    printf("%s (%d failures)\n", g_fail ? "FAILED" : "ALL OK", g_fail);
    return g_fail ? 1 : 0;
}
