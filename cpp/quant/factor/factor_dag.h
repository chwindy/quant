// factor_dag.h — DAG-based factor dependency management
//
// FactorDAG is the single DAG implementation in the quant system.
// It supports both:
//   1. Factor-level DAG — built from FactorRegistry (factors with compute functions)
//   2. Generic task-level DAG — arbitrary tasks with execute_fn
//
// Usage for factor computation:
//   auto dag = FactorDAG::from_graph(ir_graph, registry)
//   auto levels = dag->parallel_levels()
//   for (auto& level : levels) { /* run each level in parallel */ }
//
// Usage for generic tasks:
//   FactorDAG dag(nullptr);  // no registry needed for generic tasks
//   auto a = dag.add_task("A", []{ ... });
//   auto b = dag.add_task("B", []{ ... });
//   dag.add_dependency(b, a);
//   auto levels = dag.parallel_levels();
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cpp/quant/factor/factor_registry.h"

namespace quant::ir { struct StrategyGraph; }

namespace quant::factor {

// ── Task status ──
enum class DagTaskStatus : uint8_t {
    kPending = 0,
    kRunning = 1,
    kCompleted = 2,
    kFailed = 3,
    kCancelled = 4,
};

// ── Generic DAG task node ──
struct DagNode {
    FactorId id;
    std::string name;
    std::function<void()> execute_fn;
    std::vector<FactorId> dependencies;
    std::vector<FactorId> dependents;
    std::atomic<DagTaskStatus> status{DagTaskStatus::kPending};
    std::string error_message;
    int64_t created_at{0};
    int64_t started_at{0};
    int64_t completed_at{0};

    DagNode(FactorId id, std::string name, std::function<void()> fn)
        : id(id), name(std::move(name)), execute_fn(std::move(fn)) {}
};

// ── DAG validation result ──
struct DAGValidationResult {
    bool valid{true};
    std::string message;
    std::vector<FactorId> cycle_path;  // factors forming a cycle (if any)
};

// ── FactorDAG: builds and validates dependency graph ──
class FactorDAG {
public:
    explicit FactorDAG(const FactorRegistry* registry);
    ~FactorDAG() = default;

    FactorDAG(const FactorDAG&) = delete;
    FactorDAG& operator=(const FactorDAG&) = delete;

    // ── Factor-level interface ──

    // Build/rebuild the DAG from registry dependencies
    void build();

    // ── Generic task-level interface ──

    // Add a generic task node (does not require a FactorRegistry)
    FactorId add_task(std::string name, std::function<void()> execute_fn);

    // Get task node by ID (nullptr if not found or not a generic task)
    DagNode* get_task(FactorId id);
    const DagNode* get_task(FactorId id) const;

    // Number of distinct nodes (both factor-backed and generic tasks)
    size_t size() const noexcept { return tasks_.size(); }

    // ── Common interface ──

    // Validate the DAG (cycle detection)
    DAGValidationResult validate() const;

    // Get topological sort order (computation sequence)
    std::vector<FactorId> topological_sort() const;

    // Group nodes into parallel execution levels
    // Each level can be executed concurrently
    std::vector<std::vector<FactorId>> parallel_levels() const;

    // Get dependencies for a node
    std::vector<FactorId> get_dependencies(FactorId id) const;

    // Get dependents (reverse dependencies) for a node
    std::vector<FactorId> get_dependents(FactorId id) const;

    // Clear the DAG
    void clear();

    // Check if DAG is built
    bool is_built() const noexcept { return built_; }

    // Add a dependency edge (for manual DAG construction)
    void add_dependency(FactorId dependent, FactorId dependency);

    // ── IR-based construction ──
    // Build a FactorDAG from an IR StrategyGraph.
    // Registers nodes via OpRegistry, sets up dependency edges from IR edges.
    static std::unique_ptr<FactorDAG> from_graph(
        const quant::ir::StrategyGraph& graph,
        FactorRegistry& registry);

private:
    // DFS-based topological sort with cycle detection
    bool dfs_topo(FactorId id,
                  std::unordered_set<FactorId>& visited,
                  std::unordered_set<FactorId>& in_stack,
                  std::vector<FactorId>& order,
                  std::vector<FactorId>& cycle_path) const;

    const FactorRegistry* registry_;
    bool built_{false};

    // Adjacency lists (shared by factor-backed and generic-task nodes)
    std::unordered_map<FactorId, std::vector<FactorId>> deps_;     // node → deps
    std::unordered_map<FactorId, std::vector<FactorId>> dependents_; // node → dependents

    // Generic task nodes (factor-backed nodes live in the registry)
    std::unordered_map<FactorId, std::unique_ptr<DagNode>> tasks_;
    FactorId next_task_id_{1};
};

}  // namespace quant::factor
