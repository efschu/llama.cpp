#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// Minimal fixed-size thread pool for disk I/O operations.
// Used exclusively for eviction writes and LRU file deletes.
// Restores run directly on the request thread (blocking is correct there).
// No std::future — fire-and-forget only.
class DiskIoPool {
public:
    explicit DiskIoPool(size_t n_threads = 2);
    ~DiskIoPool();

    // Enqueue a fire-and-forget task.
    void enqueue(std::function<void()> task);

private:
    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> queue_;
    std::mutex                        mu_;
    std::condition_variable           cv_;
    bool                              stop_ = false;
};

// ── Inline implementations ───────────────────────────────────────────────

inline DiskIoPool::DiskIoPool(size_t n_threads) {
    for (size_t i = 0; i < n_threads; i++) {
        workers_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock lock(this->mu_);
                    this->cv_.wait(lock, [this]() {
                        return this->stop_ || !this->queue_.empty();
                    });
                    if (this->stop_ && this->queue_.empty()) {
                        return;
                    }
                    task = std::move(this->queue_.front());
                    this->queue_.pop();
                }
                try {
                    task();
                } catch (const std::exception & e) {
                    SRV_ERR("[disk-io] task threw: %s\n", e.what());
                }
            }
        });
    }
}

inline DiskIoPool::~DiskIoPool() {
    {
        std::lock_guard lock(this->mu_);
        this->stop_ = true;
    }
    this->cv_.notify_all();
    for (auto & w : this->workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
}

inline void DiskIoPool::enqueue(std::function<void()> task) {
    {
        std::lock_guard lock(this->mu_);
        this->queue_.push(std::move(task));
    }
    this->cv_.notify_one();
}
