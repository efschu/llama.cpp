#pragma once

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>

using json = nlohmann::ordered_json;

// One entry in the disk KV-cache index.
struct KvDiskEntry {
    std::string file;         // absolute path to .kvc file
    std::string session_id;   // opaque session id from HTTP request
    int         slot_id;      // slot that wrote this (-1 after cold-start)
    size_t      file_bytes;   // file size in bytes
    int64_t     written_at;   // unix epoch seconds
    size_t      n_tokens;     // tokens restored (0 until restored)
};

// Manages disk-tier KV-cache: budget, LRU, session index, and index.json.
class KvDiskCache {
public:
    KvDiskCache(const std::string & path, float max_gb);

    // ── Startup ──────────────────────────────────────────────────────────
    //
    // 1. Create base_path_ if missing.
    // 2. Read index.json if present.
    // 3. Validate each entry: keep if .kvc exists, remove if missing.
    // 4. Rewrite index.json atomically.
    // 5. Log warm-start summary.
    void init();

    // ── File paths ───────────────────────────────────────────────────────
    std::string make_slot_path(int slot_id, const std::string & session_id) const;

    // ── Session lookup ───────────────────────────────────────────────────
    const KvDiskEntry * lookup_session(const std::string & session_id) const;

    // ── Accounting (called under caller's lock) ──────────────────────────
    void record_write(int slot_id, const std::string & session_id,
                      const std::string & file, size_t bytes, int64_t written_at);
    void record_delete(const std::string & session_id);
    void record_delete_by_path(const std::string & file_path);

    // ── Budget / LRU ─────────────────────────────────────────────────────
    bool would_exceed_budget(size_t needed_bytes) const;

    // session_id of the oldest ON_DISK entry; "" if none.
    std::string lru_session() const;

    size_t used_bytes()    const;
    float  max_gb()        const;
    size_t session_count() const;

private:
    // Atomic write: serialize to tmp, rename(). Must be called under mu_.
    void flush_index_locked();

    // Find the entry with the smallest written_at. Returns "" if empty.
    std::string lru_session_unlocked() const;

    std::string  base_path_;
    float        max_gb_;
    size_t       used_bytes_ = 0;

    // session_id -> entry
    std::unordered_map<std::string, KvDiskEntry> sessions_;
    mutable std::mutex mu_;
};

// ── Inline implementations ───────────────────────────────────────────────

inline KvDiskCache::KvDiskCache(const std::string & path, float max_gb)
    : base_path_(path), max_gb_(max_gb) {}

inline void KvDiskCache::init() {
    std::filesystem::create_directories(base_path_);

    std::string idx_path = base_path_ + "/index.json";
    std::string idx_path_tmp = base_path_ + "/index.json.tmp";

    // Load existing index
    std::unordered_map<std::string, KvDiskEntry> loaded;
    size_t loaded_bytes = 0;
    int removed = 0;

    if (std::filesystem::exists(idx_path)) {
        try {
            std::ifstream f(idx_path);
            json j = json::parse(f);
            for (const auto & e : j["entries"]) {
                KvDiskEntry entry;
                entry.session_id = e["session_id"].get<std::string>();
                entry.file       = e["file"].get<std::string>();
                entry.slot_id    = e.value("slot_id", -1);
                entry.file_bytes = e.value("file_bytes", (size_t)0);
                entry.written_at = e.value("written_at", (int64_t)0);
                entry.n_tokens   = e.value("n_tokens", (size_t)0);

                if (std::filesystem::exists(entry.file)) {
                    loaded[entry.session_id] = std::move(entry);
                    loaded_bytes += entry.file_bytes;
                } else {
                    removed++;
                }
            }
        } catch (...) {
            // Corrupt index — start fresh
            loaded.clear();
            loaded_bytes = 0;
        }
    }

    // Log stale files on disk not in index
    if (std::filesystem::exists(base_path_)) {
        for (const auto & p : std::filesystem::directory_iterator(base_path_)) {
            if (p.path().extension() == ".kvc") {
                bool found = false;
                for (const auto & kv : loaded) {
                    if (kv.second.file == p.path().string()) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    SRV_WRN("[disk-kv] stale file %s not in index — ignoring\n", p.path().string().c_str());
                }
            }
        }
    }

    {
        std::lock_guard lock(this->mu_);
        this->sessions_ = std::move(loaded);
        this->used_bytes_ = loaded_bytes;
    }

    // Rewrite index atomically
    {
        std::lock_guard lock(this->mu_);
        flush_index_locked();
    }

    double miB = (double)used_bytes_ / (1024.0 * 1024.0);
    SRV_INF("[disk-kv] warm-start: %zu session(s) available on disk (%.1f MiB)\n",
            session_count(), miB);
    if (removed > 0) {
        SRV_WRN("[disk-kv] warm-start: %d index entries had missing files — removed\n", removed);
    }
}

inline std::string KvDiskCache::make_slot_path(int slot_id, const std::string & session_id) const {
    // Hash session_id to short slug
    size_t h = std::hash<std::string>{}(session_id);
    char slug[9];
    snprintf(slug, sizeof(slug), "%08zx", h);
    char slot[8];
    snprintf(slot, sizeof(slot), "%04d", slot_id);
    return std::filesystem::path(base_path_) /
           ("slot_" + std::string(slot) + "_" + slug + ".kvc");
}

inline const KvDiskEntry * KvDiskCache::lookup_session(const std::string & session_id) const {
    std::lock_guard lock(this->mu_);
    auto it = this->sessions_.find(session_id);
    if (it != this->sessions_.end()) {
        return &it->second;
    }
    return nullptr;
}

inline void KvDiskCache::record_write(int slot_id, const std::string & session_id,
                                       const std::string & file, size_t bytes, int64_t written_at) {
    std::lock_guard lock(this->mu_);
    KvDiskEntry entry;
    entry.file       = file;
    entry.session_id = session_id;
    entry.slot_id    = slot_id;
    entry.file_bytes = bytes;
    entry.written_at = written_at;
    entry.n_tokens   = 0;

    this->sessions_[session_id] = std::move(entry);
    this->used_bytes_ += bytes;
    flush_index_locked();
}

inline void KvDiskCache::record_delete(const std::string & session_id) {
    std::lock_guard lock(this->mu_);
    auto it = this->sessions_.find(session_id);
    if (it != this->sessions_.end()) {
        this->used_bytes_ -= it->second.file_bytes;
        std::filesystem::remove(it->second.file);
        this->sessions_.erase(it);
    }
    flush_index_locked();
}

inline void KvDiskCache::record_delete_by_path(const std::string & file_path) {
    std::lock_guard lock(this->mu_);
    for (auto it = this->sessions_.begin(); it != this->sessions_.end(); ++it) {
        if (it->second.file == file_path) {
            this->used_bytes_ -= it->second.file_bytes;
            std::filesystem::remove(file_path);
            this->sessions_.erase(it);
            flush_index_locked();
            return;
        }
    }
    flush_index_locked();
}

inline bool KvDiskCache::would_exceed_budget(size_t needed_bytes) const {
    std::lock_guard lock(this->mu_);
    return this->used_bytes_ + needed_bytes > (size_t)(this->max_gb_ * 1024.0 * 1024.0 * 1024.0);
}

inline std::string KvDiskCache::lru_session() const {
    std::lock_guard lock(this->mu_);
    return lru_session_unlocked();
}

inline std::string KvDiskCache::lru_session_unlocked() const {
    if (this->sessions_.empty()) return "";
    auto it = std::min_element(this->sessions_.begin(), this->sessions_.end(),
        [](const auto & a, const auto & b) {
            return a.second.written_at < b.second.written_at;
        });
    return it->first;
}

inline size_t KvDiskCache::used_bytes()    const { return used_bytes_; }
inline float  KvDiskCache::max_gb()        const { return max_gb_; }
inline size_t KvDiskCache::session_count() const { std::lock_guard lock(this->mu_); return this->sessions_.size(); }

inline void KvDiskCache::flush_index_locked() {
    // mu_ must already be held by caller
    std::string tmp = base_path_ + "/index.json.tmp";
    std::string dst = base_path_ + "/index.json";

    json j;
    j["version"] = 1;
    j["entries"] = json::array();
    for (const auto & kv : sessions_) {
        json e;
        e["session_id"] = kv.second.session_id;
        e["file"]       = kv.second.file;
        e["slot_id"]    = kv.second.slot_id;
        e["file_bytes"] = (int64_t)kv.second.file_bytes;
        e["written_at"] = kv.second.written_at;
        e["n_tokens"]   = (int64_t)kv.second.n_tokens;
        j["entries"].push_back(e);
    }

    std::ofstream f(tmp, std::ios::trunc);
    f << j.dump(2);
    f.flush();

    std::filesystem::rename(tmp, dst);
}
