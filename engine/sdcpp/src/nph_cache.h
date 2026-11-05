// NPH 0.28 on-disk cache of repacked (HMX-tiled) FP8 DiT weights. ZIT-HEXAGON, MIT.
// One cache file per model file, keyed by: model size + mtime + hash of the model header + engine
// version + tile layout. Entries are appended (data first, then a checksummed record header that
// acts as the commit marker); every hit is checksum-verified and falls back to the normal
// read+repack path on any mismatch.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

class NphRepackCache {
public:
    static constexpr const char* kEngineVersion = "nph-0.28.0/f8e4m3-tile1024-v1";

    // Opens (or creates) the cache for `model_path` in `cache_dir`. Returns nullptr when disabled or
    // when the directory is unusable. An existing cache with a different identity is replaced.
    static std::shared_ptr<NphRepackCache> open(const std::string& cache_dir, const std::string& model_path);
    // Env based: NPH_CACHE_DIR set and NPH_REPACK_CACHE != 0. Instances are shared per model path.
    static std::shared_ptr<NphRepackCache> open_from_env(const std::string& model_path);
    static void close_all();

    ~NphRepackCache();

    bool has(const std::string& key, size_t size) const;
    // Reads the entry into dst (exactly `size` bytes) and verifies its checksum.
    bool read_into(const std::string& key, void* dst, size_t size);
    // Appends an entry. Skipped (returns false) when it exists, storage is low or the cap is hit.
    bool store(const std::string& key, const void* data, size_t size);

    size_t entries() const;
    const std::string& path() const { return path_; }

    static uint64_t hash_bytes(const void* data, size_t size, uint64_t seed = 0);
    static uint64_t dir_size_bytes(const std::string& dir);
    static int clear_dir(const std::string& dir);  // removes *.nphc files; returns count
    static std::string find_any(const std::string& dir);  // path of an existing non-empty cache or ""

private:
    struct Entry {
        uint64_t data_off;
        uint64_t size;
        uint64_t checksum;
    };
    NphRepackCache() = default;
    bool init(const std::string& cache_dir, const std::string& model_path);

    std::string path_;
    std::string dir_;
    int fd_ = -1;
    mutable std::mutex mtx_;
    std::unordered_map<std::string, Entry> index_;
    uint64_t end_off_ = 0;
    bool write_disabled_ = false;
    uint64_t max_bytes_ = 0;
};
