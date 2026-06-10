// wave_scheduler.h — Wave-based parallel task scheduler (coroutine-aware)
// Executes tasks in parallel levels respecting DAG dependencies
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "cpp/quant/factor/factor_dag.h"
#include "coroutine.h"

namespace quant::infra {
class WorkStealingExecutor;
}

namespace quant::scheduler {

// ── Wave scheduler configuration ──
struct WaveSchedulerConfig {
    size_t max_concurrency = 4;  // max threads for parallel execution
    bool   enable_profiling = false;
};

// ── Schedule execution result ──
using WaveTaskId = uint64_t;

struct WaveExecutionResult {
    bool success{true};
    std::string error_message;
    size_t total_tasks{0};
    size_t completed_tasks{0};
    size_t failed_tasks{0};
    int64_t total_duration_ns{0};
    std::vector<std::pair<WaveTaskId, int64_t>> task_timings;
};

// ── WaveScheduler: executes tasks in parallel waves ──
class WaveScheduler {
public:
    explicit WaveScheduler(WaveSchedulerConfig config = {});
    ~WaveScheduler();

    WaveScheduler(const WaveScheduler&) = delete;
    WaveScheduler& operator=(const WaveScheduler&) = delete;

    // ── Synchronous execution (legacy, uses std::thread) ──
    WaveExecutionResult execute(factor::FactorDAG& graph);
    WaveExecutionResult execute_tasks(factor::FactorDAG& graph,
                                       const std::vector<WaveTaskId>& task_ids);

    // ── Coroutine-based async execution ──
    // Runs each wave's tasks concurrently on the given executor.
    quant::infra::CoTask<WaveExecutionResult>
    execute_async(factor::FactorDAG& graph,
                  quant::infra::WorkStealingExecutor& executor);

    quant::infra::CoTask<WaveExecutionResult>
    execute_tasks_async(factor::FactorDAG& graph,
                         const std::vector<WaveTaskId>& task_ids,
                         quant::infra::WorkStealingExecutor& executor);

    const WaveSchedulerConfig& config() const noexcept { return config_; }

private:
    WaveSchedulerConfig config_;
};

}  // namespace quant::scheduler
