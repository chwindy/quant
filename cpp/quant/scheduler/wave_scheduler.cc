// wave_scheduler.cc — WaveScheduler implementation (coroutine-aware)
#include "wave_scheduler.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include "coroutine.h"
#include "work_stealing_executor.h"
#include "affinity_mutex.h"

namespace quant::scheduler {

WaveScheduler::WaveScheduler(WaveSchedulerConfig config)
    : config_(std::move(config)) {}

WaveScheduler::~WaveScheduler() = default;

// ── Synchronous execution (legacy) ──

WaveExecutionResult WaveScheduler::execute(factor::FactorDAG& graph) {
    auto validation = graph.validate();
    if (!validation.valid) {
        WaveExecutionResult r;
        r.success = false;
        r.error_message = "Graph validation failed: " + validation.message;
        return r;
    }

    auto levels = graph.parallel_levels();
    WaveExecutionResult result;
    result.total_tasks = graph.size();

    auto start = std::chrono::high_resolution_clock::now();

    for (auto& level : levels) {
        std::vector<std::thread> threads;
        size_t concurrency = std::min(config_.max_concurrency, level.size());

        for (size_t i = 0; i < level.size(); i += concurrency) {
            threads.clear();
            size_t end = std::min(i + concurrency, level.size());

            for (size_t j = i; j < end; ++j) {
                factor::FactorId id = level[j];
                threads.emplace_back([&graph, id, &result]() {
                    auto* task = graph.get_task(id);
                    if (!task) return;

                    factor::DagTaskStatus expected = factor::DagTaskStatus::kPending;
                    if (!task->status.compare_exchange_strong(
                            expected, factor::DagTaskStatus::kRunning)) return;

                    auto tstart = std::chrono::high_resolution_clock::now();
                    task->started_at = std::chrono::duration_cast<
                        std::chrono::milliseconds>(tstart.time_since_epoch()).count();

                    try {
                        task->execute_fn();
                        task->status.store(factor::DagTaskStatus::kCompleted);
                    } catch (const std::exception& e) {
                        task->status.store(factor::DagTaskStatus::kFailed);
                        task->error_message = e.what();
                    }

                    auto tend = std::chrono::high_resolution_clock::now();
                    task->completed_at = std::chrono::duration_cast<
                        std::chrono::milliseconds>(tend.time_since_epoch()).count();
                    auto duration = std::chrono::duration_cast<
                        std::chrono::nanoseconds>(tend - tstart).count();

                    if (task->status.load() == factor::DagTaskStatus::kCompleted) {
                        result.completed_tasks++;
                    } else {
                        result.failed_tasks++;
                    }
                    result.task_timings.emplace_back(id, duration);
                });
            }

            for (auto& t : threads) {
                if (t.joinable()) t.join();
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.total_duration_ns = std::chrono::duration_cast<
        std::chrono::nanoseconds>(end - start).count();
    result.success = (result.failed_tasks == 0);

    return result;
}

WaveExecutionResult WaveScheduler::execute_tasks(
    factor::FactorDAG& graph, const std::vector<WaveTaskId>& task_ids) {
    // Traverse to collect all transitive dependencies
    std::unordered_set<factor::FactorId> all_ids(task_ids.begin(), task_ids.end());
    std::vector<factor::FactorId> stack(task_ids.begin(), task_ids.end());
    while (!stack.empty()) {
        factor::FactorId id = stack.back(); stack.pop_back();
        for (auto dep : graph.get_dependencies(id)) {
            if (!all_ids.contains(dep)) {
                all_ids.insert(dep);
                stack.push_back(dep);
            }
        }
    }

    // Build filtered graph
    factor::FactorDAG filtered(nullptr);
    std::unordered_map<factor::FactorId, factor::FactorId> id_map;
    for (auto id : all_ids) {
        auto* task = graph.get_task(id);
        if (!task) continue;
        auto new_id = filtered.add_task(task->name, task->execute_fn);
        id_map[id] = new_id;
    }

    for (auto id : all_ids) {
        for (auto dep : graph.get_dependencies(id)) {
            if (id_map.contains(dep) && id_map.contains(id)) {
                filtered.add_dependency(id_map[id], id_map[dep]);
            }
        }
    }

    return execute(filtered);
}

// ── Coroutine-based async execution ──

quant::infra::CoTask<WaveExecutionResult>
WaveScheduler::execute_async(
    factor::FactorDAG& graph,
    quant::infra::WorkStealingExecutor& executor) {
    auto validation = graph.validate();
    if (!validation.valid) {
        WaveExecutionResult r;
        r.success = false;
        r.error_message = "Graph validation failed: " + validation.message;
        co_return r;
    }

    auto levels = graph.parallel_levels();
    WaveExecutionResult result;
    result.total_tasks = graph.size();

    auto start = std::chrono::high_resolution_clock::now();

    for (auto& level : levels) {
        std::atomic<size_t> remaining{level.size()};
        std::promise<void> level_promise;
        auto level_future = level_promise.get_future();
        infra::AffinityMutex result_mutex;

        for (factor::FactorId id : level) {
            executor.add([&, id]() {
                auto* task = graph.get_task(id);
                if (!task) return;

                factor::DagTaskStatus expected = factor::DagTaskStatus::kPending;
                if (!task->status.compare_exchange_strong(
                        expected, factor::DagTaskStatus::kRunning)) return;

                auto tstart = std::chrono::high_resolution_clock::now();
                task->started_at = std::chrono::duration_cast<
                    std::chrono::milliseconds>(tstart.time_since_epoch()).count();

                try {
                    task->execute_fn();
                    task->status.store(factor::DagTaskStatus::kCompleted);
                } catch (const std::exception& e) {
                    task->status.store(factor::DagTaskStatus::kFailed);
                    task->error_message = e.what();
                }

                auto tend = std::chrono::high_resolution_clock::now();
                task->completed_at = std::chrono::duration_cast<
                    std::chrono::milliseconds>(tend.time_since_epoch()).count();
                auto duration = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(tend - tstart).count();

                {
                    auto lock = infra::blockingWait(result_mutex.co_scoped_lock());
                    if (task->status.load() == factor::DagTaskStatus::kCompleted) {
                        result.completed_tasks++;
                    } else {
                        result.failed_tasks++;
                    }
                    result.task_timings.emplace_back(id, duration);
                }

                if (remaining.fetch_sub(1) == 1) {
                    level_promise.set_value();
                }
            });
        }

        level_future.wait();
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.total_duration_ns = std::chrono::duration_cast<
        std::chrono::nanoseconds>(end - start).count();
    result.success = (result.failed_tasks == 0);

    co_return result;
}

quant::infra::CoTask<WaveExecutionResult>
WaveScheduler::execute_tasks_async(
    factor::FactorDAG& graph,
    const std::vector<WaveTaskId>& task_ids,
    quant::infra::WorkStealingExecutor& executor) {
    // Traverse to collect all transitive dependencies
    std::unordered_set<factor::FactorId> all_ids(task_ids.begin(), task_ids.end());
    std::vector<factor::FactorId> stack(task_ids.begin(), task_ids.end());
    while (!stack.empty()) {
        factor::FactorId id = stack.back(); stack.pop_back();
        for (auto dep : graph.get_dependencies(id)) {
            if (!all_ids.contains(dep)) {
                all_ids.insert(dep);
                stack.push_back(dep);
            }
        }
    }

    // Build filtered graph
    factor::FactorDAG filtered(nullptr);
    std::unordered_map<factor::FactorId, factor::FactorId> id_map;
    for (auto id : all_ids) {
        auto* task = graph.get_task(id);
        if (!task) continue;
        auto new_id = filtered.add_task(task->name, task->execute_fn);
        id_map[id] = new_id;
    }

    for (auto id : all_ids) {
        for (auto dep : graph.get_dependencies(id)) {
            if (id_map.contains(dep) && id_map.contains(id)) {
                filtered.add_dependency(id_map[id], id_map[dep]);
            }
        }
    }

    co_return co_await execute_async(filtered, executor);
}

}  // namespace quant::scheduler
