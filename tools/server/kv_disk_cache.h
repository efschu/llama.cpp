#pragma once

#include <llama.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::ordered_json;

// One entry in the disk KV-cache index.
struct KvDiskEntry {
    std::string file;         // absolute path to .kvc file
    std::string tokens_file;  // absolute path to .tokens file
    int         slot_id;      // owning slot id
    size_t      file_bytes;   // size of .kvc file in bytes
    int64_t     written_at;   // unix epoch seconds
    size_t      n_tokens;     // number of cached tokens
};

// Manages disk-tier KV-cache: budget, LRU, slot index, and index.json.
class KvDiskCache {
public:
    KvDiskCache(const std::string & path, float max_gb);

    // ── Startup ──────────────────────────────────────────────────────────
    void init();

    // Warm-start: restore slot states from disk entries.
    // Must be called AFTER slots are created, BEFORE accepting requests.
    // Warm-start: call the provided callback for each entry so the caller
    // can populate slot cache_tokens and disk_state.
    void restore_slot_states(const std::function<void(int slot_id,
                                                      const std::string & kvc_file,
                                                      const std::string & tokens_file,
                                                      size_t n_tokens)> & callback);
    // Path helpers
    std::string kvc_path(int slot_id)    const;
    std::string tokens_path(int slot_id) const;

    // ── Slot lookup ──────────────────────────────────────────────────────
    const KvDiskEntry * lookup_slot(int slot_id) const;

    // ── Accounting (called under caller's lock) ──────────────────────────
    void record_write(int slot_id,
                      const std::string & file,
                      const std::string & tokens_file,
                      size_t bytes,
                      int64_t written_at,
                      size_t n_tokens);
    void record_delete(int slot_id);
    void record_delete_by_path(const std::string & file_path);

    // ── Budget / LRU ─────────────────────────────────────────────────────
    bool would_exceed_budget(size_t needed_bytes) const;

    // slot_id of the oldest entry; -1 if none.
    int lru_slot() const;

    size_t used_bytes()   const;
    float  max_gb()       const;
    size_t slot_count()   const;

private:
    void flush_index_locked();
    int  lru_slot_unlocked() const;

    std::string  base_path_;
    float        max_gb_;
    size_t       used_bytes_ = 0;

    // slot_id -> entry
    std::unordered_map<int, KvDiskEntry> entries_;
    mutable std::mutex mu_;
};

// ── Inline implementations ───────────────────────────────────────────────

inline KvDiskCache::KvDiskCache(const std::string & path, float max_gb)
    : base_path_(path), max_gb_(max_gb) {}

inline void KvDiskCache::init() {
    std::filesystem::create_directories(base_path_);

    std::string idx_path = base_path_ + "/index.json";
    std::string idx_path_tmp = base_path_ + "/index.json.tmp";

    std::unordered_map<int, KvDiskEntry> loaded;
    size_t loaded_bytes = 0;
    int removed = 0;

    if (std::filesystem::exists(idx_path)) {
        try {
            std::ifstream f(idx_path);
            json j = json::parse(f);
            int version = j.value("version", 0);
            if (version != 2) {
                // Skip old format entries
                SRV_WRN("[disk-kv] index.json version %d != 2 — skipping all entries\n", version);
            } else {
                for (const auto & e : j["entries"]) {
                    KvDiskEntry entry;
                    entry.slot_id    = e["slot_id"].get<int>();
                    entry.file       = e["file"].get<std::string>();
                    entry.tokens_file = e["tokens_file"].get<std::string>();
                    entry.file_bytes = e.value("file_bytes", (size_t)0);
                    entry.written_at = e.value("written_at", (int64_t)0);
                    entry.n_tokens   = e.value("n_tokens", (size_t)0);

                    if (std::filesystem::exists(entry.file) &&
                        std::filesystem::exists(entry.tokens_file)) {
                        loaded[entry.slot_id] = std::move(entry);
                        loaded_bytes += entry.file_bytes;
                    } else {
                        // Remove orphaned files
                        std::filesystem::remove(entry.file);
                        std::filesystem::remove(entry.tokens_file);
                        removed++;
                    }
                }
            }
        } catch (...) {
            loaded.clear();
            loaded_bytes = 0;
        }
    }

    {
        std::lock_guard lock(this->mu_);
        this->entries_ = std::move(loaded);
        this->used_bytes_ = loaded_bytes;
    }

    // Rewrite index atomically
    {
        std::lock_guard lock(this->mu_);
        flush_index_locked();
    }

    double miB = (double)used_bytes_ / (1024.0 * 1024.0);
    SRV_INF("[disk-kv] warm-start: %zu slot(s) restorable from disk (%.1f MiB)\n",
            slot_count(), miB);
    if (removed > 0) {
        SRV_WRN("[disk-kv] warm-start: %d entries had missing files — removed\n", removed);
    }
}

inline void KvDiskCache::restore_slot_states(
    const std::function<void(int, const std::string &, const std::string &, size_t)> & callback) {
    std::lock_guard lock(this->mu_);
    for (auto & kv : entries_) {
        KvDiskEntry & entry = kv.second;
        callback(entry.slot_id, entry.file, entry.tokens_file, entry.n_tokens);
    }
}

inline std::string KvDiskCache::kvc_path(int slot_id) const {
    return std::filesystem::path(base_path_) / ("slot_" + std::to_string(slot_id) + ".kvc");
}

inline std::string KvDiskCache::tokens_path(int slot_id) const {
    return std::filesystem::path(base_path_) / ("slot_" + std::to_string(slot_id) + ".tokens");
}

inline const KvDiskEntry * KvDiskCache::lookup_slot(int slot_id) const {
    std::lock_guard lock(this->mu_);
    auto it = this->entries_.find(slot_id);
    return it != this->entries_.end() ? &it->second : nullptr;
}

inline void KvDiskCache::record_write(int slot_id,
                                       const std::string & file,
                                       const std::string & tokens_file,
                                       size_t bytes,
                                       int64_t written_at,
                                       size_t n_tokens) {
    std::lock_guard lock(this->mu_);
    // Remove old entry for this slot if exists
    auto old = this->entries_.find(slot_id);
    if (old != this->entries_.end()) {
        this->used_bytes_ -= old->second.file_bytes;
        this->entries_.erase(old);
    }

    KvDiskEntry entry;
    entry.file        = file;
    entry.tokens_file = tokens_file;
    entry.slot_id     = slot_id;
    entry.file_bytes  = bytes;
    entry.written_at  = written_at;
    entry.n_tokens    = n_tokens;

    this->entries_[slot_id] = std::move(entry);
    this->used_bytes_ += bytes;
    flush_index_locked();
}

inline void KvDiskCache::record_delete(int slot_id) {
    std::lock_guard lock(this->mu_);
    auto it = this->entries_.find(slot_id);
    if (it != this->entries_.end()) {
        this->used_bytes_ -= it->second.file_bytes;
        std::filesystem::remove(it->second.file);
        std::filesystem::remove(it->second.tokens_file);
        this->entries_.erase(it);
    }
    flush_index_locked();
}

inline void KvDiskCache::record_delete_by_path(const std::string & file_path) {
    std::lock_guard lock(this->mu_);
    for (auto it = this->entries_.begin(); it != this->entries_.end(); ++it) {
        if (it->second.file == file_path) {
            this->used_bytes_ -= it->second.file_bytes;
            std::filesystem::remove(file_path);
            std::filesystem::remove(it->second.tokens_file);
            this->entries_.erase(it);
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

inline int KvDiskCache::lru_slot() const {
    std::lock_guard lock(this->mu_);
    return lru_slot_unlocked();
}

inline int KvDiskCache::lru_slot_unlocked() const {
    if (this->entries_.empty()) return -1;
    auto it = std::min_element(this->entries_.begin(), this->entries_.end(),
        [](const auto & a, const auto & b) {
            return a.second.written_at < b.second.written_at;
        });
    return it->first;
}

inline size_t KvDiskCache::used_bytes() const { return used_bytes_; }
inline float  KvDiskCache::max_gb()     const { return max_gb_; }
inline size_t KvDiskCache::slot_count() const { std::lock_guard lock(this->mu_); return this->entries_.size(); }

inline void KvDiskCache::flush_index_locked() {
    std::string tmp = base_path_ + "/index.json.tmp";
    std::string dst = base_path_ + "/index.json";

    json j;
    j["version"] = 2;
    j["entries"] = json::array();
    for (const auto & kv : entries_) {
        json e;
        e["slot_id"]     = kv.second.slot_id;
        e["file"]        = kv.second.file;
        e["tokens_file"] = kv.second.tokens_file;
        e["file_bytes"]  = (int64_t)kv.second.file_bytes;
        e["written_at"]  = kv.second.written_at;
        e["n_tokens"]    = (int64_t)kv.second.n_tokens;
        j["entries"].push_back(e);
    }

    std::ofstream f(tmp, std::ios::trunc);
    f << j.dump(2);
    f.flush();

    std::filesystem::rename(tmp, dst);
}
