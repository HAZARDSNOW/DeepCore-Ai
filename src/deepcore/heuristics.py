"""
heuristics.py
Student heuristic starter file for DeepCore:
  - h_naive_manhattan : provided baseline
  - h1_basic          : required student implementation
  - h2_advanced       : required student implementation
"""

import heapq
import math
from typing import Dict, List, Tuple
from collections import deque
from .environment import DeepCoreEnv, DIRECTION_DELTAS
from .state import DeepCoreState, Pos

# Cache wall-aware shortest path distances per environment instance
_WALL_DIST_CACHE: Dict[int, Dict[Tuple[Pos, Pos], int]] = {}

# Cache wall-aware *movement cost* distances (sludge priced at its real cost)
_STATIC_COST_CACHE: Dict[int, Dict[Tuple[Pos, Pos], int]] = {}


def _get_wall_aware_distances(env: DeepCoreEnv) -> Dict[Tuple[Pos, Pos], int]:
    """Precomputes all-pairs shortest path steps ignoring dynamic obstacles (only walls # block)."""
    env_id = id(env)
    if env_id in _WALL_DIST_CACHE:
        return _WALL_DIST_CACHE[env_id]

    walkable = [
        (r, c)
        for r in range(env.rows)
        for c in range(env.cols)
        if (r, c) not in env.walls
    ]
    dist_map: Dict[Tuple[Pos, Pos], int] = {}

    for src in walkable:
        q = deque([(src, 0)])
        visited = {src}
        while q:
            curr, d = q.popleft()
            dist_map[(src, curr)] = d
            cr, cc = curr
            for dr, dc in DIRECTION_DELTAS.values():
                nxt = (cr + dr, cc + dc)
                if env.in_bounds(nxt) and nxt not in env.walls and nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, d + 1))

    _WALL_DIST_CACHE[env_id] = dist_map
    return dist_map


def _prim_mst_cost(nodes: List[Pos], dist_fn) -> float:
    """Computes Minimum Spanning Tree weight over `nodes` using `dist_fn(u, v)`."""
    if len(nodes) <= 1:
        return 0.0
    in_mst = {nodes[0]}
    total_weight = 0.0

    while len(in_mst) < len(nodes):
        best_edge = float("inf")
        best_node = None
        for u in in_mst:
            for v in nodes:
                if v not in in_mst:
                    w = dist_fn(u, v)
                    if w < best_edge:
                        best_edge = w
                        best_node = v
        if best_node is None:
            break
        in_mst.add(best_node)
        total_weight += best_edge

    return total_weight


def h_naive_manhattan(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """
    Baseline Naive Heuristic: Manhattan distance to nearest waiting miner (or Elevator if none).
    """
    if env.is_goal(state):
        return 0.0
    r, c = state.bot_pos
    if state.miners:
        return float(min(abs(r - mr) + abs(c - mc) for mr, mc in state.miners))
    er, ec = env.elevator_pos
    return float(abs(r - er) + abs(c - ec))


def h1_basic(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """
    Basic Heuristic: Minimum Spanning Tree over bot, miners, and elevator, plus payload cost.
    """
    if env.is_goal(state):
        return 0.0

    distances = _get_wall_aware_distances(env)

    def step_dist(u: Pos, v: Pos) -> float:
        d = distances.get((u, v))
        if d is None:
            return float(abs(u[0] - v[0]) + abs(u[1] - v[1]))
        return float(d)

    sites: List[Pos] = [state.bot_pos, *sorted(state.miners), env.elevator_pos]
    mst_cost = _prim_mst_cost(sites, step_dist)

    total_miners = state.carried + len(state.miners)
    if state.miners:
        # whichever miner is collected last, the pod then carries every miner
        last_pickup = min(
            state.miners, key=lambda p: step_dist(p, env.elevator_pos)
        )
        payload_steps = step_dist(last_pickup, env.elevator_pos)
    else:
        payload_steps = step_dist(state.bot_pos, env.elevator_pos)

    payload_cost = env.PAYLOAD_SURCHARGE * total_miners * payload_steps
    return float(mst_cost + payload_cost)


def _is_corner_deadlock(pos: Pos, env: DeepCoreEnv) -> bool:
    """Returns True if a boulder at `pos` is wedged against perpendicular rock walls (#)."""
    r, c = pos
    n_wall = (r - 1, c) in env.walls
    s_wall = (r + 1, c) in env.walls
    e_wall = (r, c + 1) in env.walls
    w_wall = (r, c - 1) in env.walls
    return (n_wall or s_wall) and (e_wall or w_wall)


def _get_static_move_costs(env: DeepCoreEnv) -> Dict[Tuple[Pos, Pos], int]:

    env_id = id(env)
    cached = _STATIC_COST_CACHE.get(env_id)
    if cached is not None:
        return cached

    walkable = [
        (r, c)
        for r in range(env.rows)
        for c in range(env.cols)
        if (r, c) not in env.walls
    ]
    dist_map: Dict[Tuple[Pos, Pos], int] = {}

    for src in walkable:
        best: Dict[Pos, int] = {src: 0}
        heap: List[Tuple[int, Pos]] = [(0, src)]
        while heap:
            d, curr = heapq.heappop(heap)
            if d != best.get(curr):
                continue
            cr, cc = curr
            for dr, dc in DIRECTION_DELTAS.values():
                nxt = (cr + dr, cc + dc)
                if not env.in_bounds(nxt) or nxt in env.walls:
                    continue
                step = env.COST_SLUDGE if nxt in env.sludge else env.COST_CLEAR_FLOOR
                nd = d + step
                if nd < best.get(nxt, math.inf):
                    best[nxt] = nd
                    heapq.heappush(heap, (nd, nxt))
        for cell, d in best.items():
            dist_map[(src, cell)] = d

    _STATIC_COST_CACHE[env_id] = dist_map
    return dist_map


def _entry_move_cost(env: DeepCoreEnv, cell: Pos) -> int:
    return env.COST_SLUDGE if cell in env.sludge else env.COST_CLEAR_FLOOR


def _push_extra_per_chasm(env: DeepCoreEnv, chasm: Pos) -> float:
   
    r, c = chasm
    entry: int | None = None
    for dr, dc in DIRECTION_DELTAS.values():
        neighbour = (r + dr, c + dc)
        if not env.in_bounds(neighbour) or neighbour in env.walls:
            continue
        cost = _entry_move_cost(env, neighbour)
        entry = cost if entry is None else max(entry, cost)
    if entry is None:
        entry = env.COST_CLEAR_FLOOR
    return float(env.COST_PUSH_BASE - entry)


def _min_barrier_crossings(
    env: DeepCoreEnv,
    start: Pos,
    targets: List[Pos],
    barriers: frozenset,
    step_distances: Dict[Tuple[Pos, Pos], int] | None = None,
) -> float:
    
    if not targets:
        return 0.0

    if not barriers:
        if step_distances is None:
            step_distances = _get_wall_aware_distances(env)
        for target in targets:
            if (start, target) not in step_distances:
                return math.inf
        return 0.0

    dist: Dict[Pos, int] = {start: 0}
    queue: deque = deque([start])
    while queue:
        curr = queue.popleft()
        base = dist[curr]
        r, c = curr
        for dr, dc in DIRECTION_DELTAS.values():
            nxt = (r + dr, c + dc)
            if not env.in_bounds(nxt) or nxt in env.walls:
                continue
            weight = 1 if nxt in barriers else 0
            nd = base + weight
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                if weight:
                    queue.append(nxt)
                else:
                    queue.appendleft(nxt)

    worst = 0
    for target in targets:
        d = dist.get(target)
        if d is None:
            return math.inf
        worst = max(worst, d)
    return float(worst)


def _min_weighted_barrier_cost(
    env: DeepCoreEnv, start: Pos, targets: List[Pos], weights: Dict[Pos, float]
) -> float:
    if not targets:
        return 0.0

    dist: Dict[Pos, float] = {start: 0.0}
    heap: List[Tuple[float, Pos]] = [(0.0, start)]
    while heap:
        d, curr = heapq.heappop(heap)
        if d != dist.get(curr):
            continue
        r, c = curr
        for dr, dc in DIRECTION_DELTAS.values():
            nxt = (r + dr, c + dc)
            if not env.in_bounds(nxt) or nxt in env.walls:
                continue
            nd = d + weights.get(nxt, 0.0)
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))

    worst = 0.0
    for target in targets:
        d = dist.get(target)
        if d is None:
            return math.inf
        worst = max(worst, d)
    return worst


def h2_advanced(state: DeepCoreState, env: DeepCoreEnv) -> float:
    if env.is_goal(state):
        return 0.0

    static_costs = _get_static_move_costs(env)
    step_distances = _get_wall_aware_distances(env)

    def move_cost(u: Pos, v: Pos) -> float:
        d = static_costs.get((u, v))
        if d is None:
            return float(abs(u[0] - v[0]) + abs(u[1] - v[1]))
        return float(d)

    def step_cost(u: Pos, v: Pos) -> float:
        d = step_distances.get((u, v))
        if d is None:
            return float(abs(u[0] - v[0]) + abs(u[1] - v[1]))
        return float(d)

    sites: List[Pos] = [state.bot_pos, *sorted(state.miners), env.elevator_pos]
    mst_cost = _prim_mst_cost(sites, move_cost)

    elevator = env.elevator_pos
    total_miners = state.carried + len(state.miners)
  
    payload_bounds = [

        env.PAYLOAD_SURCHARGE
        * (
            state.carried * step_cost(state.bot_pos, elevator)
            + sum(step_cost(miner, elevator) for miner in state.miners)
        ),

        env.PAYLOAD_SURCHARGE
        * total_miners
        * min(
            (
                step_cost(miner, elevator)
                for miner in state.miners
            ),
            default=step_cost(state.bot_pos, elevator),
        ),
    ]
    payload_cost = max(payload_bounds)

    targets: List[Pos] = [elevator, *sorted(state.miners)]

    push_lb = _min_barrier_crossings(
        env, state.bot_pos, targets, state.chasms, step_distances
    )
    if math.isinf(push_lb):
        return math.inf


    movable_boulders = sum(
        1 for boulder in state.boulders if not _is_corner_deadlock(boulder, env)
    )
    if push_lb > movable_boulders:
        return math.inf

    drill_lb = _min_barrier_crossings(
        env, state.bot_pos, targets, state.rubble, step_distances
    )
    if math.isinf(drill_lb):
        return math.inf

    if state.chasms:
        push_weights = {
            chasm: _push_extra_per_chasm(env, chasm) for chasm in state.chasms
        }
        uniform = set(push_weights.values())
        if len(uniform) == 1:
            push_extra = uniform.pop() * push_lb
        else:
            push_extra = _min_weighted_barrier_cost(
                env, state.bot_pos, targets, push_weights
            )
            if math.isinf(push_extra):
                return math.inf
    else:
        push_extra = 0.0

    drill_extra = env.COST_DRILL * drill_lb
    return float(mst_cost + payload_cost + push_extra + drill_extra)
