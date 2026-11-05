// NPH 0.29 text-encoder output cache (ZIT-HEXAGON, MIT). See nph_te_cache.h.
#include "nph_te_cache.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <map>
#include <mutex>

namespace nph_te {
namespace {

inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

std::mutex g_mtx;
Stats g_stats;
std::map<std::string, std::string> g_fp_memo;

bool pread_full(int fd, void* dst, size_t n, uint64_t off) {
    uint8_t* p = (uint8_t*)dst;
    while (n > 0) {
        ssize_t r = ::pread(fd, p, n, (off_t)off);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        p += r;
        n -= (size_t)r;
        off += (uint64_t)r;
    }
    return true;
}

bool write_full(int fd, const void* src, size_t n) {
    const uint8_t* p = (const uint8_t*)src;
    while (n > 0) {
        ssize_t r = ::write(fd, p, n);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        p += r;
        n -= (size_t)r;
    }
    return true;
}

std::string hex64(uint64_t v) {
    char b[17];
    snprintf(b, sizeof(b), "%016llx", (unsigned long long)v);
    return b;
}

struct Header {
    char magic[8];
    uint64_t key_len;
    uint64_t payload_len;
    uint64_t payload_hash;
    uint64_t header_hash;  // over magic..payload_hash + key bytes
};

uint64_t header_hash(const Header& h, const std::string& key) {
    uint64_t x = hash64(&h, offsetof(Header, header_hash), 0x7e11);
    return hash64(key.data(), key.size(), x);
}

}  // namespace

uint64_t hash64(const void* data, size_t size, uint64_t seed) {
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

std::string file_fingerprint(const std::string& path) {
    if (path.empty()) return "";
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) return "";
    const int64_t mt = (int64_t)st.st_mtim.tv_sec * 1000000000ll + st.st_mtim.tv_nsec;
    const std::string memo_key = path + "|" + std::to_string((uint64_t)st.st_size) + "|" + std::to_string(mt);
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_fp_memo.find(memo_key);
        if (it != g_fp_memo.end()) return it->second;
    }
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return "";
    const uint64_t size = (uint64_t)st.st_size;
    const size_t head   = (size_t)std::min<uint64_t>(size, 4ull << 20);
    const size_t tail   = (size_t)std::min<uint64_t>(size - head, 1ull << 20);
    std::vector<uint8_t> buf(head + tail);
    bool ok = pread_full(fd, buf.data(), head, 0) && (tail == 0 || pread_full(fd, buf.data() + head, tail, size - tail));
    ::close(fd);
    if (!ok) return "";
    std::string fp = std::to_string(size) + ":" + std::to_string(mt) + ":" + hex64(hash64(buf.data(), buf.size(), 0xF1));
    std::lock_guard<std::mutex> lk(g_mtx);
    g_fp_memo[memo_key] = fp;
    return fp;
}

std::string disk_path(const std::string& dir, const std::string& key) {
    return dir + "/" + hex64(hash64(key.data(), key.size(), 0xA5)) + hex64(hash64(key.data(), key.size(), 0x5A3C)) + ".nphte";
}

bool disk_load(const std::string& dir, const std::string& key, std::string& payload) {
    const std::string path = disk_path(dir, key);
    int fd                 = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    Header h{};
    bool ok = pread_full(fd, &h, sizeof(h), 0) && memcmp(h.magic, kFormat, 8) == 0 && h.key_len == key.size() &&
              h.payload_len < (1ull << 31);
    std::string k;
    if (ok) {
        k.resize(h.key_len);
        ok = pread_full(fd, &k[0], k.size(), sizeof(h)) && k == key && header_hash(h, k) == h.header_hash;
    }
    if (ok) {
        payload.resize(h.payload_len);
        ok = (h.payload_len == 0 || pread_full(fd, &payload[0], payload.size(), sizeof(h) + h.key_len)) &&
             hash64(payload.data(), payload.size(), 0xDA7A) == h.payload_hash;
        // must be exactly the file length (no trailing garbage / truncation)
        struct stat st{};
        ok = ok && fstat(fd, &st) == 0 && (uint64_t)st.st_size == sizeof(h) + h.key_len + h.payload_len;
    }
    ::close(fd);
    if (!ok) {
        payload.clear();
        ::unlink(path.c_str());  // corrupt or foreign entry: drop it
        std::lock_guard<std::mutex> lk(g_mtx);
        g_stats.disk_rejects++;
        return false;
    }
    ::utimes(path.c_str(), nullptr);  // LRU touch
    return true;
}

uint64_t disk_size(const std::string& dir) {
    uint64_t total = 0;
    DIR* d         = opendir(dir.c_str());
    if (!d) return 0;
    while (dirent* e = readdir(d)) {
        std::string n = e->d_name;
        if (n.size() > 6 && n.compare(n.size() - 6, 6, ".nphte") == 0) {
            struct stat st{};
            if (stat((dir + "/" + n).c_str(), &st) == 0) total += (uint64_t)st.st_size;
        }
    }
    closedir(d);
    return total;
}

int disk_clear(const std::string& dir) {
    int count = 0;
    DIR* d    = opendir(dir.c_str());
    if (!d) return 0;
    while (dirent* e = readdir(d)) {
        std::string n = e->d_name;
        if ((n.size() > 6 && n.compare(n.size() - 6, 6, ".nphte") == 0) || n.find(".nphte.tmp") != std::string::npos) {
            if (::unlink((dir + "/" + n).c_str()) == 0) count++;
        }
    }
    closedir(d);
    return count;
}

bool disk_store(const std::string& dir, const std::string& key, const std::string& payload, uint64_t cap_bytes) {
    const uint64_t need = sizeof(Header) + key.size() + payload.size();
    if (need > cap_bytes) return false;
    ::mkdir(dir.c_str(), 0700);
    // evict oldest until the new entry fits
    struct F { std::string path; uint64_t size; int64_t mt; };
    std::vector<F> files;
    uint64_t total = 0;
    if (DIR* d = opendir(dir.c_str())) {
        while (dirent* e = readdir(d)) {
            std::string n = e->d_name;
            if (n.size() > 6 && n.compare(n.size() - 6, 6, ".nphte") == 0) {
                struct stat st{};
                std::string p = dir + "/" + n;
                if (stat(p.c_str(), &st) == 0) {
                    files.push_back({p, (uint64_t)st.st_size, (int64_t)st.st_mtim.tv_sec * 1000000000ll + st.st_mtim.tv_nsec});
                    total += (uint64_t)st.st_size;
                }
            }
        }
        closedir(d);
    } else {
        return false;
    }
    std::sort(files.begin(), files.end(), [](const F& a, const F& b) { return a.mt < b.mt; });
    const std::string path = disk_path(dir, key);
    for (const auto& f : files) {
        if (total + need <= cap_bytes) break;
        if (f.path == path) continue;
        if (::unlink(f.path.c_str()) == 0) total -= f.size;
    }
    if (total + need > cap_bytes) return false;

    Header h{};
    memcpy(h.magic, kFormat, 8);
    h.key_len      = key.size();
    h.payload_len  = payload.size();
    h.payload_hash = hash64(payload.data(), payload.size(), 0xDA7A);
    h.header_hash  = header_hash(h, key);
    const std::string tmp = path + ".nphte.tmp" + std::to_string((long)getpid());
    int fd                = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    bool ok = write_full(fd, &h, sizeof(h)) && write_full(fd, key.data(), key.size()) &&
              write_full(fd, payload.data(), payload.size()) && ::fsync(fd) == 0;
    ok = (::close(fd) == 0) && ok;
    if (!ok || ::rename(tmp.c_str(), path.c_str()) != 0) {
        ::unlink(tmp.c_str());
        return false;
    }
    return true;
}

Stats stats() {
    std::lock_guard<std::mutex> lk(g_mtx);
    return g_stats;
}

void note(const char* what, uint64_t us) {
    std::lock_guard<std::mutex> lk(g_mtx);
    std::string w = what;
    if (w == "ram_hit") g_stats.ram_hits++;
    else if (w == "disk_hit") { g_stats.disk_hits++; g_stats.disk_load_us += us; }
    else if (w == "miss") g_stats.misses++;
    else if (w == "disk_store") { g_stats.disk_stores++; g_stats.disk_store_us += us; }
    g_stats.last = w;
}

}  // namespace nph_te
