"""
solver.py
Student implementation file for DeepCore search algorithms.

- Breadth-First Search (BFS) is provided below as a complete reference example
  showing how to interact with `env.get_successors(state)` and `SearchMetrics`.
- Implement the remaining required algorithms. Do not change the public
  function signatures or the environment/state modules.
"""

from collections import deque
import heapq
import math
from typing import Callable, Dict, List, Optional, Set, Tuple

from .environment import DeepCoreEnv
from .state import DeepCoreState, SearchMetrics


def _reconstruct_path(
    parent_map: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]],
    goal_state: DeepCoreState,
) -> List[str]:
    """Helper function to trace actions from the initial state to `goal_state`."""
    actions: List[str] = []
    curr: Optional[DeepCoreState] = goal_state
    while curr is not None:
        prev, act = parent_map.get(curr, (None, None))
        if act is not None:
            actions.append(act)
        curr = prev
    actions.reverse()
    return actions


# =============================================================================
# 1. Breadth-First Search (BFS) — [PROVIDED AS REFERENCE EXAMPLE]
# =============================================================================
def breadth_first_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """
    Reference Graph-Search BFS implementation (FIFO queue).
    Minimizes the number of steps, but ignores terrain/payload energy costs!
    """
    metrics = SearchMetrics(algorithm_name="BFS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    frontier = deque([start])
    frontier_set = {start}
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    cost_so_far: Dict[DeepCoreState, int] = {start: 0}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))
        state = frontier.popleft()
        frontier_set.discard(state)

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            if next_state not in explored and next_state not in frontier_set:
                parent[next_state] = (state, action.value)
                cost_so_far[next_state] = cost_so_far[state] + step_cost
                if env.is_goal(next_state):
                    metrics.solved = True
                    metrics.solution_actions = _reconstruct_path(parent, next_state)
                    metrics.total_cost = cost_so_far[next_state]
                    return metrics
                frontier.append(next_state)
                frontier_set.add(next_state)
                metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 2. Depth-First Search (DFS) & Iterative Deepening Search (IDS) — [STUDENT TODO]
# =============================================================================

def depth_first_search(env: DeepCoreEnv, max_expansions: Optional[int] = None) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="DFS")
    start = env.get_initial_state()

    if env.is_goal(start):
        metrics.solved = True
        return metrics

    frontier = [start]
    frontier_set = {start}
    explored = set()
    parent = {start: (None, None)}
    cost_so_far = {start: 0}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

        state = frontier.pop()
        frontier_set.remove(state)

        if state in explored:
            continue

        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = cost_so_far[state]
            return metrics

        if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
            metrics.status_message = "Expansion limit reached."
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1

            if next_state not in explored and next_state not in frontier_set:
                parent[next_state] = (state, action.value)
                cost_so_far[next_state] = cost_so_far[state] + step_cost
                frontier.append(next_state)
                frontier_set.add(next_state)

        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics


def iterative_deepening_search(
    env: DeepCoreEnv,
    max_depth: Optional[int] = None,
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="IDS")
    start = env.get_initial_state()

    if env.is_goal(start):
        metrics.solved = True
        return metrics

    depth_limit = 0

    while max_depth is None or depth_limit <= max_depth:
        stack = [(start, 0)]
        parent = {start: (None, None)}
        costs = {start: 0}
        best_depth = {start: 0}
        cutoff = False

        while stack:
            metrics.max_frontier_size = max(
                metrics.max_frontier_size, len(stack)
            )
            metrics.max_memory_size = max(
                metrics.max_memory_size,
                len(stack) + len(best_depth)
            )

            state, depth = stack.pop()

            if depth != best_depth.get(state):
                continue

            if env.is_goal(state):
                metrics.solved = True
                metrics.solution_actions = _reconstruct_path(parent, state)
                metrics.total_cost = costs[state]
                metrics.status_message = "Goal reached."
                return metrics

            if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
                metrics.status_message = "Expansion limit reached."
                return metrics

            metrics.nodes_expanded += 1
            metrics.explored_order.append(state.bot_pos)

            successors = env.get_successors(state)

            for action, child, step_cost in reversed(successors):
                metrics.nodes_generated += 1
                new_depth = depth + 1

                if new_depth < best_depth.get(child, float("inf")):
                    if depth >= depth_limit:
                        cutoff = True
                        continue

                    best_depth[child] = new_depth
                    parent[child] = (state, action.value)
                    costs[child] = costs[state] + step_cost
                    stack.append((child, new_depth))

        if not cutoff:
            break

        depth_limit += 1

    metrics.status_message = "No solution found."
    return metrics




# =============================================================================
# 3. Bidirectional Search — [STUDENT TODO]
# =============================================================================

def bidirectional_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="Bidirectional")
    start = env.get_initial_state()

    if env.is_goal(start):
        metrics.solved = True
        return metrics

    goal_states = env.get_goal_states()

    if not goal_states:
        metrics.status_message = "No solution found."
        return metrics

    frontier_forward = deque([start])
    frontier_backward = deque(goal_states)

    visited_forward = {start}
    visited_backward = set(goal_states)

    parent_forward = {start: (None, None)}
    parent_backward = {
        state: (None, None) for state in goal_states
    }

    cost_forward = {start: 0}
    cost_backward = {state: 0 for state in goal_states}

    metrics.max_frontier_size = len(frontier_forward) + len(frontier_backward)
    metrics.max_memory_size = len(visited_forward) + len(visited_backward)

    meeting_state = None

    while frontier_forward and frontier_backward and meeting_state is None:
        forward_size = len(frontier_forward)

        for _ in range(forward_size):
            state = frontier_forward.popleft()

            metrics.nodes_expanded += 1
            metrics.explored_order.append(state.bot_pos)

            for action, next_state, step_cost in env.get_successors(state):
                metrics.nodes_generated += 1

                if next_state not in visited_forward:
                    visited_forward.add(next_state)
                    parent_forward[next_state] = (state, action.value)
                    cost_forward[next_state] = cost_forward[state] + step_cost
                    frontier_forward.append(next_state)

                    metrics.max_memory_size = max(
                        metrics.max_memory_size,
                        len(visited_forward) + len(visited_backward)
                    )

                    if next_state in visited_backward:
                        meeting_state = next_state
                        break

            metrics.max_frontier_size = max(
                metrics.max_frontier_size,
                len(frontier_forward) + len(frontier_backward)
            )

            if meeting_state is not None:
                break

        if meeting_state is not None:
            break

        backward_size = len(frontier_backward)

        for _ in range(backward_size):
            state = frontier_backward.popleft()

            metrics.nodes_expanded += 1
            metrics.explored_order.append(state.bot_pos)

            for action, prev_state, step_cost in env.get_predecessors(state):
                metrics.nodes_generated += 1

                if prev_state not in visited_backward:
                    visited_backward.add(prev_state)
                    parent_backward[prev_state] = (state, action.value)
                    cost_backward[prev_state] = cost_backward[state] + step_cost
                    frontier_backward.append(prev_state)

                    metrics.max_memory_size = max(
                        metrics.max_memory_size,
                        len(visited_forward) + len(visited_backward)
                    )

                    if prev_state in visited_forward:
                        meeting_state = prev_state
                        break

            metrics.max_frontier_size = max(
                metrics.max_frontier_size,
                len(frontier_forward) + len(frontier_backward)
            )

            if meeting_state is not None:
                break

    if meeting_state is not None:
        forward_actions = _reconstruct_path(
            parent_forward, meeting_state
        )

        backward_actions = []
        state = meeting_state

        while parent_backward[state][0] is not None:
            next_state, action = parent_backward[state]
            backward_actions.append(action)
            state = next_state

        metrics.solved = True
        metrics.solution_actions = forward_actions + backward_actions
        metrics.total_cost = (
            cost_forward[meeting_state] + cost_backward[meeting_state]
        )
        metrics.status_message = "Goal reached."
        return metrics

    metrics.status_message = "No solution found."
    return metrics



# =============================================================================
# 4. Uniform Cost Search (UCS) — [STUDENT TODO]
# =============================================================================

def uniform_cost_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="UCS")
    start = env.get_initial_state()

    if env.is_goal(start):
        metrics.solved = True
        metrics.status_message = "Goal reached."
        return metrics

    counter = 0
    frontier = [(0, counter, start)]
    explored = set()
    parent = {start: (None, None)}
    cost_so_far = {start: 0}

    while frontier:
        metrics.max_frontier_size = max(
            metrics.max_frontier_size, len(frontier)
        )
        metrics.max_memory_size = max(
            metrics.max_memory_size, len(frontier) + len(explored)
        )

        cost, _, state = heapq.heappop(frontier)

        if cost != cost_so_far[state]:
            continue

        if state in explored:
            continue

        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = cost
            metrics.status_message = "Goal reached."
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            new_cost = cost + step_cost

            if next_state not in explored and new_cost < cost_so_far.get(next_state, math.inf):
                cost_so_far[next_state] = new_cost
                parent[next_state] = (state, action.value)
                counter += 1
                heapq.heappush(
                    frontier, (new_cost, counter, next_state)
                )

        metrics.max_memory_size = max(
            metrics.max_memory_size, len(frontier) + len(explored)
        )

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 5. Greedy Best-First Search (GBFS) — [STUDENT TODO]
# =============================================================================
def greedy_best_first_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """
    Greedy Best-First Graph Search.

    The frontier is a priority queue ordered strictly by h(n); the actual path
    cost g(n) is only tracked for reporting, never for ordering. Goal test is
    performed when a node is popped, like other priority-queue searches.
    """
    metrics = SearchMetrics(algorithm_name="GBFS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        metrics.status_message = "Goal reached."
        return metrics

    counter = 0
    frontier: List[Tuple[float, int, DeepCoreState]] = [(heuristic(start, env), counter, start)]
    frontier_set = {start}
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    cost_so_far: Dict[DeepCoreState, int] = {start: 0}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

        _, _, state = heapq.heappop(frontier)
        frontier_set.discard(state)

        if state in explored:
            continue
        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = cost_so_far[state]
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            if next_state not in explored and next_state not in frontier_set:
                h_value = heuristic(next_state, env)
                if math.isinf(h_value):
                    continue
                parent[next_state] = (state, action.value)
                cost_so_far[next_state] = cost_so_far[state] + step_cost
                counter += 1
                heapq.heappush(frontier, (h_value, counter, next_state))
                frontier_set.add(next_state)
                metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 6. A* Search — [STUDENT TODO]
# =============================================================================
def a_star_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """
    A* Graph Search ordering the priority queue by f(n) = g(n) + h(n).

    The goal test happens when a node is popped, so with an admissible heuristic
    the returned path is optimal. A cheaper path to an already expanded state
    reopens it instead of being discarded, which keeps the search correct even
    for inconsistent heuristics.
    """
    metrics = SearchMetrics(algorithm_name="A*")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        metrics.status_message = "Goal reached."
        return metrics

    counter = 0
    frontier: List[Tuple[float, int, int, DeepCoreState]] = [
        (heuristic(start, env), counter, 0, start)
    ]
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    cost_so_far: Dict[DeepCoreState, int] = {start: 0}
    metrics.max_frontier_size = 1
    metrics.max_memory_size = 1

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

        _, _, popped_g, state = heapq.heappop(frontier)

        if popped_g != cost_so_far.get(state):
            continue
        if state in explored:
            continue
        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = cost_so_far[state]
            metrics.status_message = "Goal reached."
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            tentative_g = cost_so_far[state] + step_cost
            if tentative_g >= cost_so_far.get(next_state, math.inf):
                continue

            h_value = heuristic(next_state, env)
            if math.isinf(h_value):
                continue

            cost_so_far[next_state] = tentative_g
            parent[next_state] = (state, action.value)
            explored.discard(next_state)
            counter += 1
            f_value = tentative_g + h_value
            heapq.heappush(frontier, (f_value, counter, tentative_g, next_state))
            metrics.max_memory_size = max(
                metrics.max_memory_size, len(frontier) + len(explored)
            )

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# BONUS 1: Weighted A* Search — [STUDENT BONUS TODO]
# =============================================================================
def weighted_a_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    weight: float = 1.5,
) -> Optional[SearchMetrics]:
    """
    Weighted A* Graph Search ordering the priority queue by f(n) = g(n) + W * h(n).

    With W > 1 the heuristic is inflated, trading optimality for fewer
    expansions. A cheaper path to an already expanded state reopens it, but
    because the ordering itself is weighted, the returned path is only
    guaranteed to be W-suboptimal.
    """
    metrics = SearchMetrics(algorithm_name="Weighted A*")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        metrics.status_message = "Goal reached."
        return metrics

    counter = 0
    frontier: List[Tuple[float, int, int, DeepCoreState]] = [
        (weight * heuristic(start, env), counter, 0, start)
    ]
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    cost_so_far: Dict[DeepCoreState, int] = {start: 0}
    metrics.max_frontier_size = 1
    metrics.max_memory_size = 1

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

        _, _, popped_g, state = heapq.heappop(frontier)

        if popped_g != cost_so_far.get(state):
            continue
        if state in explored:
            continue
        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = cost_so_far[state]
            metrics.status_message = "Goal reached."
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            tentative_g = cost_so_far[state] + step_cost
            if tentative_g >= cost_so_far.get(next_state, math.inf):
                continue

            h_value = heuristic(next_state, env)
            if math.isinf(h_value):
                continue

            cost_so_far[next_state] = tentative_g
            parent[next_state] = (state, action.value)
            explored.discard(next_state)
            counter += 1
            f_value = tentative_g + weight * h_value
            heapq.heappush(frontier, (f_value, counter, tentative_g, next_state))
            metrics.max_memory_size = max(
                metrics.max_memory_size, len(frontier) + len(explored)
            )

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# BONUS 2: Iterative Deepening A* (IDA*) — [STUDENT BONUS TODO]
# =============================================================================

def ida_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:

    metrics = SearchMetrics(algorithm_name="IDA*")
    start = env.get_initial_state()

    if env.is_goal(start):
        metrics.solved = True
        metrics.status_message = "Goal reached."
        return metrics

    h_cache = {}

    def get_h(state):
        if state not in h_cache:
            h_cache[state] = heuristic(state, env)
        return h_cache[state]

    threshold = get_h(start)

    while not math.isinf(threshold):
        stack = [[start, 0, None, 0]]
        path = {start}
        best_g = {start: 0}
        actions = []
        next_threshold = math.inf

        while stack:
            metrics.max_frontier_size = max(
                metrics.max_frontier_size, len(stack)
            )
            metrics.max_memory_size = max(
                metrics.max_memory_size,
                len(stack) + len(best_g)
            )

            state, cost, successors, index = stack[-1]
            f_value = cost + get_h(state)

            if f_value > threshold:
                next_threshold = min(next_threshold, f_value)
                stack.pop()
                path.remove(state)

                if actions:
                    actions.pop()

                continue

            if env.is_goal(state):
                metrics.solved = True
                metrics.solution_actions = actions.copy()
                metrics.total_cost = cost
                metrics.status_message = "Goal reached."
                return metrics

            if successors is None:
                if (
                    max_expansions is not None
                    and metrics.nodes_expanded >= max_expansions
                ):
                    metrics.status_message = "Expansion limit reached."
                    return metrics

                metrics.nodes_expanded += 1
                metrics.explored_order.append(state.bot_pos)
                successors = []

                for action, child, step_cost in env.get_successors(state):
                    metrics.nodes_generated += 1

                    if child in path:
                        continue

                    new_cost = cost + step_cost
                    h_value = get_h(child)

                    if math.isinf(h_value):
                        continue

                    new_f = new_cost + h_value

                    if new_f > threshold:
                        next_threshold = min(next_threshold, new_f)
                        continue

                    if new_cost < best_g.get(child, math.inf):
                        best_g[child] = new_cost
                        successors.append(
                            (new_f, action, child, step_cost)
                        )

                successors.sort(key=lambda item: item[0])
                stack[-1][2] = successors

            if index >= len(successors):
                stack.pop()
                path.remove(state)

                if actions:
                    actions.pop()

                continue

            _, action, child, step_cost = successors[index]
            stack[-1][3] += 1

            path.add(child)
            actions.append(action.value)

            stack.append([
                child,
                cost + step_cost,
                None,
                0
            ])

        if next_threshold == math.inf:
            break

        threshold = next_threshold

    metrics.status_message = "No solution found."
    return metrics




# =============================================================================
# BONUS 3: Simplified Memory-Bounded A* (SMA*) — [STUDENT BONUS TODO]
# =============================================================================
def sma_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    memory_limit: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Simplified Memory-Bounded A* (SMA*)."""
    raise NotImplementedError
