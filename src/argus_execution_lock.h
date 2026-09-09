#pragma once
#include <mutex>
#ifdef ARGUS_TESTING
void argus_test_notify(int event, const void * resource);
#endif

// Production has no observer call or branch. Test builds observe actual contention.
inline std::unique_lock<std::mutex> argus_execution_lock(std::mutex & mutex,
                                                        const void * resource, int kind) {
#ifdef ARGUS_TESTING
    std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
    if (!lock.try_lock()) {
        argus_test_notify(10 + kind, resource);
        lock.lock();
    }
    return lock;
#else
    (void)resource;
    (void)kind;
    return std::unique_lock<std::mutex>(mutex);
#endif
}
