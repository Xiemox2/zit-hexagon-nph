#include "nph_cache.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#include "nph_stats.h"

namespace {

constexpr uint64_t kFileMagic = 0x3130303043485048ull;  // "HPHC0001" little endian-ish tag
constexpr uint64_t kRecMagic  = 0x3130434552485048ull;
constexpr uint32_t kFormat    = 1;
constexpr size_t kBlock       = 4096;
constexpr size_t kMaxKey      = 2048;

inline uint64_t round_up(uint64_t x, uint64_t m) { return (x + m - 1) / m * m; }

struct FileHeader {
    uint64_t magic;
    uint32_t format;
    uint32_t tile;
    char engine[64];
    uint64_t model_size;
    int64_t model_mtime_ns;
    uint64_t model_header_hash;
    uint64_t self_hash;
};

struct RecHeader {
    uint64_t magic;
    uint64_t data_size;
    uint64_t checksum;
    uint32_t key_len;
    uint32_t reserved;
    uint64_t self_hash;  // hash of the fields above + key
    // key bytes follow
};

bool pread_all(int fd, void* buf, size_t n, uint64_t off) {
    char* p = (char*)buf;
    while (n > 0) {
        size_t chunk = std::min<size_t>(n, 16u << 20);
        ssize_t r    = pread(fd, p, chunk, (off_t)off);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        p += r;
        n -= (size_t)r;
        off += (uint64_t)r;
    }
    return true;
}

bool pwrite_all(int fd, const void* buf, size_t n, uint64_t off) {
    const char* p = (const char*)buf;
    while (n > 0) {
        size_t chunk = std::min<size_t>(n, 16u << 20);
        ssize_t r    = pwrite(fd, p, chunk, (off_t)off);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        p += r;
        n -= (size_t)r;
        off += (uint64_t)r;
    }
    return true;
}

uint64_t rec_self_hash(const RecHeader& h, const char* key) {
    RecHeader c = h;
    c.self_hash = 0;
    uint64_t x  = NphRepackCache::hash_bytes(&c, sizeof(c), 0x51ed);
    return NphRepackCache::hash_bytes(key, h.key_len, x);
}

bool model_identity(const std::string& model_path, uint64_t* size, int64_t* mtime_ns, uint64_t* hhash) {
    struct stat st{};
    if (stat(model_path.c_str(), &st) != 0) return false;
    *size     = (uint64_t)st.st_size;
    *mtime_ns = (int64_t)st.st_mtim.tv_sec * 1000000000ll + st.st_mtim.tv_nsec;
    int fd    = ::open(model_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    // safetensors: u64 header length + JSON header (names, dtypes, shapes, offsets). Otherwise 1 MiB.
    uint64_t hlen = 0;
    size_t want   = 1u << 20;
    if (pread_all(fd, &hlen, 8, 0) && hlen > 0 && hlen < (256ull << 20) && hlen + 8 <= *size) {
        want = (size_t)hlen + 8;
    }
    want = (size_t)std::min<uint64_t>(want, *size);
    std::vector<uint8_t> buf(want);
    bool ok = pread_all(fd, buf.data(), want, 0);
    ::close(fd);
    if (!ok) return false;
    *hhash = NphRepackCache::hash_bytes(buf.data(), buf.size(), 0x1d);
    return true;
}


std::mutex g_registry_mtx;
std::map<std::string, std::shared_ptr<NphRepackCache>> g_registry;

}  // namespace

static inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

uint64_t NphRepackCache::hash_bytes(const void* data, size_t size, uint64_t seed) {
    // 4-lane xxh64-style rounds; not cryptographic, detects corruption/truncation.
    const uint64_t P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full;
    const uint8_t* p  = (const uint8_t*)data;
    uint64_t v[4]     = {seed + P1 + P2, seed + P2, seed, seed - P1};
    size_t i          = 0;
    for (; i + 32 <= size; i += 32) {
        for (int l = 0; l < 4; ++l) {
            uint64_t w;
            memcpy(&w, p + i + 8 * l, 8);
            v[l] = rotl64(v[l] + w * P2, 31) * P1;
        }
    }
    uint64_t h = rotl64(v[0], 1) + rotl64(v[1], 7) + rotl64(v[2], 12) + rotl64(v[3], 18) + size;
    for (; i < size; ++i) {
        h ^= p[i] * 0x27D4EB2F165667C5ull;
        h = rotl64(h, 11) * P1;
    }
    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    return h;
}

std::shared_ptr<NphRepackCache> NphRepackCache::open(const std::string& cache_dir, const std::string& model_path) {
    std::shared_ptr<NphRepackCache> c(new NphRepackCache());
    if (!c->init(cache_dir, model_path)) return nullptr;
    return c;
}

std::shared_ptr<NphRepackCache> NphRepackCache::open_from_env(const std::string& model_path) {
    const char* dir = getenv("NPH_CACHE_DIR");
    if (dir == nullptr || *dir == 0 || !nph_env_flag("NPH_REPACK_CACHE", true)) return nullptr;
    // model paths may be /proc/self/fd/N (SAF), so key by file identity, not by path
    uint64_t msize = 0, hh = 0;
    int64_t mtime = 0;
    if (!model_identity(model_path, &msize, &mtime, &hh)) return nullptr;
    char idbuf[96];
    snprintf(idbuf, sizeof(idbuf), "%llu:%lld:%016llx", (unsigned long long)msize, (long long)mtime,
             (unsigned long long)hh);
    std::lock_guard<std::mutex> lock(g_registry_mtx);
    auto it = g_registry.find(idbuf);
    if (it != g_registry.end()) return it->second;
    auto c = open(dir, model_path);
    if (c) g_registry[idbuf] = c;
    return c;
}

void NphRepackCache::close_all() {
    std::lock_guard<std::mutex> lock(g_registry_mtx);
    g_registry.clear();
}

NphRepackCache::~NphRepackCache() {
    if (fd_ >= 0) ::close(fd_);
}

bool NphRepackCache::init(const std::string& cache_dir, const std::string& model_path) {
    dir_ = cache_dir;
    mkdir(cache_dir.c_str(), 0700);
    uint64_t msize = 0, hh = 0;
    int64_t mtime = 0;
    if (!model_identity(model_path, &msize, &mtime, &hh)) return false;
    char name[64];
    snprintf(name, sizeof(name), "dit-%016llx.nphc", (unsigned long long)(hh ^ (msize * 0x9E3779B97F4A7C15ull)));
    path_ = cache_dir + "/" + name;
    // keep a single model's cache: drop caches of other (old) model files
    if (DIR* d = opendir(cache_dir.c_str())) {
        std::vector<std::string> stale;
        while (auto* e = readdir(d)) {
            std::string n = e->d_name;
            if (n.size() > 5 && n.compare(n.size() - 5, 5, ".nphc") == 0 && n != name) stale.push_back(cache_dir + "/" + n);
        }
        closedir(d);
        for (auto& f : stale) unlink(f.c_str());
    }
    const char* mx   = getenv("NPH_CACHE_MAX_MB");
    max_bytes_       = (uint64_t)(mx && *mx ? atoll(mx) : 16384) << 20;

    FileHeader want{};
    want.magic  = kFileMagic;
    want.format = kFormat;
    want.tile   = 1024;
    snprintf(want.engine, sizeof(want.engine), "%s", kEngineVersion);
    want.model_size        = msize;
    want.model_mtime_ns    = mtime;
    want.model_header_hash = hh;
    want.self_hash         = 0;
    want.self_hash         = hash_bytes(&want, sizeof(want), 7);

    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd_ < 0) return false;
    FileHeader have{};
    struct stat st{};
    fstat(fd_, &st);
    bool reuse = st.st_size >= (off_t)kBlock && pread_all(fd_, &have, sizeof(have), 0) &&
                 memcmp(&have, &want, sizeof(want)) == 0;
    if (!reuse) {
        // different model/engine (or new file): start fresh
        if (ftruncate(fd_, 0) != 0) return false;
        std::vector<uint8_t> blk(kBlock, 0);
        memcpy(blk.data(), &want, sizeof(want));
        if (!pwrite_all(fd_, blk.data(), kBlock, 0)) return false;
        end_off_ = kBlock;
        return true;
    }
    // scan records; stop at the first torn/invalid one
    uint64_t off = kBlock;
    std::vector<uint8_t> blk(kBlock);
    while (off + kBlock <= (uint64_t)st.st_size) {
        if (!pread_all(fd_, blk.data(), kBlock, off)) break;
        RecHeader h;
        memcpy(&h, blk.data(), sizeof(h));
        if (h.magic != kRecMagic || h.key_len == 0 || h.key_len > kMaxKey) break;
        const char* key = (const char*)blk.data() + sizeof(h);
        if (rec_self_hash(h, key) != h.self_hash) break;
        const uint64_t data_off = off + kBlock;
        if (data_off + h.data_size > (uint64_t)st.st_size) break;
        index_[std::string(key, h.key_len)] = Entry{data_off, h.data_size, h.checksum};
        off = data_off + round_up(h.data_size, kBlock);
    }
    end_off_ = off;
    if ((uint64_t)st.st_size > off) {
        if (ftruncate(fd_, (off_t)off) != 0) { /* keep going, entries beyond off are ignored */ }
    }
    return true;
}

bool NphRepackCache::has(const std::string& key, size_t size) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = index_.find(key);
    return it != index_.end() && it->second.size == size;
}

size_t NphRepackCache::entries() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return index_.size();
}

bool NphRepackCache::read_into(const std::string& key, void* dst, size_t size) {
    Entry e{};
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = index_.find(key);
        if (it == index_.end()) return false;
        e = it->second;
    }
    if (e.size != size) {
        nph_sd_add(NPH_SD_CACHE_REJECT, 1);
        return false;
    }
    const uint64_t t0 = nph_now_ns();
#if defined(POSIX_FADV_SEQUENTIAL)
    posix_fadvise(fd_, (off_t)e.data_off, (off_t)size, POSIX_FADV_SEQUENTIAL);
#endif
    bool ok = pread_all(fd_, dst, size, e.data_off);
    if (ok && nph_env_flag("NPH_CACHE_VERIFY", true)) {
        ok = hash_bytes(dst, size, 0xC0FFEE) == e.checksum;
    }
    if (!ok) {
        nph_sd_add(NPH_SD_CACHE_REJECT, 1);
        std::lock_guard<std::mutex> lock(mtx_);
        index_.erase(key);  // never trust it again in this process
        return false;
    }
    nph_sd_add(NPH_SD_CACHE_HIT_BYTES, size);
    nph_sd_add(NPH_SD_CACHE_HIT_NS, nph_now_ns() - t0);
    return true;
}

bool NphRepackCache::store(const std::string& key, const void* data, size_t size) {
    if (key.empty() || key.size() > kMaxKey) return false;
    const uint64_t t0 = nph_now_ns();
    uint64_t off      = 0;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (write_disabled_ || index_.count(key)) return false;
        const uint64_t need = kBlock + round_up(size, kBlock);
        if (end_off_ + need > max_bytes_) {
            write_disabled_ = true;
            return false;
        }
        struct statvfs vfs{};
        if (statvfs(dir_.c_str(), &vfs) == 0) {
            const uint64_t free_b = (uint64_t)vfs.f_bavail * vfs.f_frsize;
            if (free_b < need + (512ull << 20)) {  // keep >= 512 MiB free for the system/app
                write_disabled_ = true;
                return false;
            }
        }
        off = end_off_;
        end_off_ += need;
    }
    RecHeader h{};
    h.magic     = kRecMagic;
    h.data_size = size;
    h.checksum  = hash_bytes(data, size, 0xC0FFEE);
    h.key_len   = (uint32_t)key.size();
    h.self_hash = rec_self_hash(h, key.data());
    std::vector<uint8_t> blk(kBlock, 0);
    memcpy(blk.data(), &h, sizeof(h));
    memcpy(blk.data() + sizeof(h), key.data(), key.size());
    // data first, then the header (commit marker); pad so the next record stays aligned
    bool ok = pwrite_all(fd_, data, size, off + kBlock);
    const uint64_t pad = round_up(size, kBlock) - size;
    if (ok && pad) {
        std::vector<uint8_t> z(pad, 0);
        ok = pwrite_all(fd_, z.data(), pad, off + kBlock + size);
    }
    ok = ok && pwrite_all(fd_, blk.data(), kBlock, off);
    std::lock_guard<std::mutex> lock(mtx_);
    if (!ok) {
        write_disabled_ = true;  // e.g. ENOSPC; later opens stop scanning at this hole
        return false;
    }
    index_[key] = Entry{off + kBlock, size, h.checksum};
    nph_sd_add(NPH_SD_CACHE_WRITE_BYTES, size);
    nph_sd_add(NPH_SD_CACHE_WRITE_NS, nph_now_ns() - t0);
    return true;
}

uint64_t NphRepackCache::dir_size_bytes(const std::string& dir) {
    uint64_t total = 0;
    DIR* d         = opendir(dir.c_str());
    if (!d) return 0;
    while (auto* e = readdir(d)) {
        std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        struct stat st{};
        if (stat((dir + "/" + n).c_str(), &st) == 0 && S_ISREG(st.st_mode)) total += (uint64_t)st.st_blocks * 512;
    }
    closedir(d);
    return total;
}

int NphRepackCache::clear_dir(const std::string& dir) {
    close_all();
    int n  = 0;
    DIR* d = opendir(dir.c_str());
    if (!d) return 0;
    std::vector<std::string> victims;
    while (auto* e = readdir(d)) {
        std::string name = e->d_name;
        if (name.size() > 5 && name.compare(name.size() - 5, 5, ".nphc") == 0) victims.push_back(dir + "/" + name);
    }
    closedir(d);
    for (auto& v : victims) n += unlink(v.c_str()) == 0;
    return n;
}

std::string NphRepackCache::find_any(const std::string& dir) {
    std::string best;
    DIR* d = opendir(dir.c_str());
    if (!d) return best;
    while (auto* e = readdir(d)) {
        std::string n = e->d_name;
        if (n.size() > 5 && n.compare(n.size() - 5, 5, ".nphc") == 0) {
            struct stat st{};
            if (stat((dir + "/" + n).c_str(), &st) == 0 && st.st_size > (off_t)(2 * kBlock)) best = dir + "/" + n;
        }
    }
    closedir(d);
    return best;
}
