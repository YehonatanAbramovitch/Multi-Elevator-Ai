"""
AI Statement of Use:
I used Claude Code as an AI-assisted tool during development.
The tool helped with debugging and code suggestions.
The design, decisions, and implementation were done by me.
"""

import heapq
import time

import ext_elev

id = ["<your-id>"]


class Controller:
    """Reinforcement-learning multi-elevator controller.

    The transition probabilities and delivery-reward distributions are
    hidden. This controller keeps running success/failure counts for every
    elevator and person, and running reward statistics for every person,
    updated from the observable consequences of each action (the state
    before/after and get_last_gained_reward()). It then plans against
    these live estimates using the same expected-cost A* idea as
    Assignment 2, sampling optimistic probability estimates (Thompson
    sampling) and an optimistic reward prior so that under-explored
    elevators/persons/deliveries keep looking attractive until enough
    data accumulates to settle on their true value.
    """

    REPLAN_PERIOD = 8
    ASTAR_TIME_BUDGET = 1.0

    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self.reachable = game.get_reachable()
        self.capacities = game.get_capacities()

        self.initial_state = game.get_initial_state()
        elevators0, persons0, _ = self.initial_state
        self.all_eids = list(self.reachable)
        self.all_pids = [p for p, _ in persons0]
        self.initial_ef = {e: f for e, f, _ in elevators0}
        self.initial_el = {e: 0 for e in self.all_eids}
        self.initial_floor = {p: loc[1] for p, loc in persons0}

        self.pw = {p: game.get_person_weight(p) for p in self.all_pids}
        self.pg = {p: game.get_person_goal(p) for p in self.all_pids}
        self.goal_r = game.get_goal_reward()
        self.horizon = game.get_max_steps()

        # Elevator connectivity: floors reachable by hopping between
        # elevators that share at least one floor (same as Assignment 2).
        adj = {e: set() for e in self.reachable}
        for e1, r1 in self.reachable.items():
            for e2, r2 in self.reachable.items():
                if e1 != e2 and r1 & r2:
                    adj[e1].add(e2)
        self.ecr = {}
        for eid_ in self.reachable:
            vis, q = {eid_}, [eid_]
            floors = set(self.reachable[eid_])
            while q:
                c = q.pop()
                floors |= self.reachable[c]
                for n in adj[c]:
                    if n not in vis:
                        vis.add(n)
                        q.append(n)
            self.ecr[eid_] = floors

        # ---- learned (online) estimates ----
        self.e_succ = {e: 0 for e in self.all_eids}
        self.e_att = {e: 0 for e in self.all_eids}
        self.p_succ = {p: 0 for p in self.all_pids}
        self.p_att = {p: 0 for p in self.all_pids}
        self.p_rew_sum = {p: 0.0 for p in self.all_pids}
        self.p_rew_n = {p: 0 for p in self.all_pids}

        # Optimistic prior for unseen delivery rewards, scaled to the one
        # reward value we do know (goal_reward); acts as one fictitious
        # high-value sample that fades out as real samples arrive.
        self._reward_prior = max(self.goal_r, 1.0)

        self.mc = {e: 1.0 for e in self.all_eids}
        self.pc = {p: 1.0 for p in self.all_pids}

        # ---- bookkeeping to credit the outcome of the last action ----
        self.prev_action = None
        self.prev_ef = None
        self.prev_total_remaining = None

        # ---- cached plan ----
        self._strategy = 'full'
        self._loop_pid = None
        self._plan_actions, self._plan_states, self._plan_idx = [], [], 0
        self._calls_since_replan = 0
        self._astar_cooldown = 0  # steps to skip A* after a failed search

    # ------------------------------------------------------------------ #
    # Learning: credit the outcome of the previous action                #
    # ------------------------------------------------------------------ #
    def _learn_from_last_action(self, state):
        if self.prev_action is None:
            return

        elevators_t, persons_t, _ = state
        act = self.prev_action

        if act == "RESET":
            return

        if act.startswith("MOVE"):
            e, tgt = self._parse_pair(act)
            new_f = next(f for (eid_, f, _) in elevators_t if eid_ == e)
            origin = self.prev_ef[e]
            if tgt != origin:
                self.e_att[e] += 1
                if new_f == tgt:
                    self.e_succ[e] += 1

        elif act.startswith("ENTER"):
            p, e = self._parse_pair(act)
            loc = next(l for (pid_, l) in persons_t if pid_ == p)
            self.p_att[p] += 1
            if loc[0] == 'in' and loc[1] == e:
                self.p_succ[p] += 1

        elif act.startswith("EXIT"):
            p, _e = self._parse_pair(act)
            found = next((l for (pid_, l) in persons_t if pid_ == p), None)
            self.p_att[p] += 1
            last_r = self.game.get_last_gained_reward()
            if last_r > 0:
                # delivered (delivery rewards are always positive). Detected via
                # the reward, not the state: when this delivery completes the
                # goal the engine resets to the initial state, so the person
                # reappears on their starting floor instead of disappearing.
                self.p_succ[p] += 1
                if self.prev_total_remaining == 1:
                    # this delivery also completed the goal; strip goal_reward
                    last_r -= self.goal_r
                self.p_rew_sum[p] += last_r
                self.p_rew_n[p] += 1
            elif found is not None and found[0] == 'floor':
                # successfully exited onto a non-destination floor
                self.p_succ[p] += 1
            # else still ('in', e): the exit attempt failed, nothing to credit

    @staticmethod
    def _parse_pair(act):
        inner = act[act.index('{') + 1: act.index('}')]
        a, b = inner.split(',')
        return int(a), int(b)

    # ------------------------------------------------------------------ #
    # Estimating the hidden model                                        #
    # ------------------------------------------------------------------ #
    def _sample_probs(self):
        """Stable point-estimate (posterior mean of Beta(succ+1, fail+1))
        for each success probability. Used for every tactical replan,
        including the very frequent forced replans caused by an unreliable
        elevator/person failing -- using a *deterministic* estimate here
        (rather than a fresh random draw each time) keeps the chosen route
        consistent across those replans instead of thrashing between
        wildly different cost guesses. Exploration is instead driven by
        the optimistic reward prior in _reward_estimate, which fades in a
        smooth, deterministic way as real samples accumulate."""
        for e in self.all_eids:
            s, n = self.e_succ[e], self.e_att[e]
            p = (s + 1) / (n + 2)
            self.mc[e] = 1.0 / max(p, 0.05)
        for p_id in self.all_pids:
            s, n = self.p_succ[p_id], self.p_att[p_id]
            p = (s + 1) / (n + 2)
            self.pc[p_id] = 1.0 / max(p, 0.05)

    def _reward_estimate(self, p):
        n = self.p_rew_n[p]
        if n == 0:
            return self._reward_prior
        return (self.p_rew_sum[p] + self._reward_prior) / (n + 1)

    # ------------------------------------------------------------------ #
    # Main entry point                                                    #
    # ------------------------------------------------------------------ #
    def choose_next_action(self, state):
        self._learn_from_last_action(state)

        elevators_t, persons_t, total_remaining = state
        if not persons_t:
            self.prev_action = "RESET"
            self.prev_ef = {e: f for e, f, _ in elevators_t}
            self.prev_total_remaining = total_remaining
            return "RESET"

        ef = {e: f for e, f, _ in elevators_t}
        el = {e: w for e, _, w in elevators_t}
        pl = {}
        for pid, loc in persons_t:
            pl[pid] = loc[1] if loc[0] == 'floor' else ('e', loc[1])

        # Strategy refresh (full-vs-loop, which person to loop on) is
        # expensive only in the sense that it scores every candidate, but
        # it uses the cheap A* heuristic as a proxy instead of a real
        # search, so it stays fast even with several persons. It only runs
        # periodically.
        need_strategy_refresh = self._calls_since_replan >= self.REPLAN_PERIOD
        self._calls_since_replan += 1

        need_reroute = need_strategy_refresh or self._plan_idx >= len(self._plan_actions)
        if not need_reroute:
            idx = self._plan_idx
            if self._strategy == 'loop':
                pid = self._loop_pid
                cur = (ef, el, {pid: pl[pid]}) if pid in pl else None
                need_reroute = cur is None or not (
                    idx < len(self._plan_states) and self._plan_states[idx] == cur)
            else:
                cur_full = (ef, el, pl)
                need_reroute = not (idx < len(self._plan_states) and self._plan_states[idx] == cur_full)

        if need_strategy_refresh or need_reroute:
            self._sample_probs()
        if need_strategy_refresh:
            self._decide_strategy(ef, el, pl)
            self._calls_since_replan = 0
            need_reroute = True
        if need_reroute:
            if self._astar_cooldown > 0:
                self._astar_cooldown -= 1
                # A* recently returned empty; use greedy this step instead of
                # retrying immediately, so a failing search doesn't consume the
                # full ASTAR_TIME_BUDGET on every single step.
            else:
                self._reroute(ef, el, pl)

        if self._strategy == 'loop' and self._plan_idx == 0 and state != self.initial_state:
            # we're about to start a fresh loop plan, which assumes a clean
            # start (empty elevators, everyone on their starting floor) --
            # reset to get there first. Once the plan is under way
            # (plan_idx > 0) we keep following/replanning it normally;
            # by then the real state naturally differs from the initial
            # one (the loop target has moved), which is expected, not a
            # signal to reset again.
            action = "RESET"
        else:
            action = self._take_from_plan()
            if action is None:
                action = "RESET" if self._strategy == 'loop' else self._greedy(ef, el, pl)

        self.prev_action = action
        self.prev_ef = ef
        self.prev_total_remaining = total_remaining
        return action

    def _take_from_plan(self):
        if self._plan_idx < len(self._plan_actions):
            action = self._plan_actions[self._plan_idx]
            self._plan_idx += 1
            return action
        return None

    # ------------------------------------------------------------------ #
    # Planning (Assignment-2-style expected-cost A*, against estimates)  #
    # ------------------------------------------------------------------ #
    def _decide_strategy(self, ef, el, pl):
        """Choose between delivering everyone ('full') and repeatedly
        resetting to chase a single lucrative person ('loop'). Uses the
        cheap admissible heuristic as a cost proxy instead of running a
        real search per candidate -- with several persons, running a full
        A* for every candidate on every refresh is what made replanning
        blow up in cost when an unreliable elevator forces frequent
        refreshes. The heuristic is only a rough proxy, but it is refreshed
        periodically and the real route is always computed for execution
        by _reroute, so a slightly-off strategic estimate just gets
        corrected on the next refresh."""
        apr = {p: self._reward_estimate(p) for p in self.all_pids}

        full_cost = self._h(ef, pl)
        full_rew = sum(apr[p] for p in pl) + self.goal_r
        full_vps = full_rew / full_cost if full_cost > 0 else float('inf')

        # Loop candidates are scored from a clean reset (empty elevators,
        # everyone on their starting floor) since that's what executing the
        # loop actually requires.
        best_vps, best_pid = -1.0, None
        for pid in self.all_pids:
            cost = self._h(self.initial_ef, {pid: self.initial_floor[pid]}) + 1.0
            vps = apr[pid] / cost
            if vps > best_vps:
                best_vps, best_pid = vps, pid

        if best_pid is not None and best_vps > full_vps:
            self._strategy, self._loop_pid = 'loop', best_pid
        else:
            self._strategy, self._loop_pid = 'full', None

    def _reroute(self, ef, el, pl):
        """Compute the actual route (a single A* call) for whichever
        strategy/target is currently active."""
        if self._strategy == 'loop':
            pid = self._loop_pid
            plan, states = self._astar(self.initial_ef, self.initial_el,
                                        {pid: self.initial_floor[pid]})
        else:
            plan, states = self._astar(ef, el, pl)
        self._plan_actions, self._plan_states = plan, states
        self._plan_idx = 0
        if not plan:
            # Back off: don't retry A* for REPLAN_PERIOD steps so a failing
            # search (state space too large) doesn't burn the full time budget.
            self._astar_cooldown = self.REPLAN_PERIOD

    def _astar(self, ef0, el0, pl0):
        def is_goal(pl):
            return all(isinstance(v, int) and v == self.pg[p] for p, v in pl.items())

        def sk(ef, pl):
            return (tuple(sorted(ef.items())), tuple(sorted(pl.items())))

        start = (ef0, el0, pl0)
        if is_goal(pl0):
            return [], [start]

        W = 3.0
        cnt = 0
        heap = [(W * self._h(ef0, pl0), cnt, sk(ef0, pl0), ef0, el0, pl0, 0.0, [], [start])]
        best = {}
        expansions = 0
        deadline = time.perf_counter() + self.ASTAR_TIME_BUDGET

        while heap:
            _, __, sk_, ef, el, pl, g, acts, states = heapq.heappop(heap)
            if sk_ in best and best[sk_] <= g:
                continue
            best[sk_] = g
            if is_goal(pl):
                return acts, states
            expansions += 1
            if expansions > 20000 or (expansions % 256 == 0 and time.perf_counter() > deadline):
                return [], [start]
            for act, nef, nel, npl, cost in self._succs(ef, el, pl):
                nsk = sk(nef, npl)
                ng = g + cost
                if nsk not in best or best[nsk] > ng:
                    cnt += 1
                    heapq.heappush(heap,
                        (ng + W * self._h(nef, npl), cnt, nsk, nef, nel, npl, ng,
                         acts + [act], states + [(nef, nel, npl)]))
        return [], [start]

    def _succs(self, ef, el, pl):
        res = []
        fp = {}
        ep_ = {e: [] for e in ef}
        for pid, loc in pl.items():
            if isinstance(loc, int):
                fp.setdefault(loc, []).append(pid)
            else:
                ep_[loc[1]].append(pid)

        for eid, floor in ef.items():
            for tgt in self.reachable[eid]:
                if tgt == floor:
                    continue
                useful = (
                    any(el[eid] + self.pw[p] <= self.capacities[eid]
                        and self.pg[p] in self.ecr[eid]
                        for p in fp.get(tgt, []))
                    or any(self.pg[p] == tgt for p in ep_[eid])
                    or any(
                        self.pg[p] not in self.reachable[eid]
                        and any(tgt in self.reachable[e2]
                                and self.pg[p] in self.ecr[e2]
                                for e2 in ef if e2 != eid)
                        for p in ep_[eid]
                    )
                )
                if useful:
                    res.append((f"MOVE{{{eid},{tgt}}}",
                                {**ef, eid: tgt}, el, pl, self.mc[eid]))

        for eid, floor in ef.items():
            for pid in fp.get(floor, []):
                if (el[eid] + self.pw[pid] <= self.capacities[eid]
                        and self.pg[pid] in self.ecr[eid]):
                    res.append((f"ENTER{{{pid},{eid}}}",
                                ef,
                                {**el, eid: el[eid] + self.pw[pid]},
                                {**pl, pid: ('e', eid)},
                                self.pc[pid]))

        for pid, loc in pl.items():
            if not isinstance(loc, tuple):
                continue
            eid, floor, goal = loc[1], ef[loc[1]], self.pg[pid]
            should_exit = (
                floor == goal
                or (goal not in self.reachable[eid]
                    and any(floor in self.reachable[e2] and goal in self.ecr[e2]
                            for e2 in ef if e2 != eid))
            )
            if should_exit:
                res.append((f"EXIT{{{pid},{eid}}}",
                            ef,
                            {**el, eid: el[eid] - self.pw[pid]},
                            {**pl, pid: floor},
                            self.pc[pid]))
        return res

    def _h(self, ef, pl):
        h, max_e = 0.0, 0.0
        for pid, loc in pl.items():
            goal = self.pg[pid]
            if isinstance(loc, int):
                if loc == goal:
                    continue
                f = loc
                h += 2.0 * self.pc[pid]
                best = None
                for e in ef:
                    if f in self.reachable[e] and goal in self.reachable[e]:
                        c = (self.mc[e] if ef[e] != f else 0.0) + self.mc[e]
                        if best is None or c < best:
                            best = c
                if best is not None:
                    max_e = max(max_e, best)
            else:
                eid = loc[1]
                floor = ef[eid]
                h += self.pc[pid]
                if floor != goal:
                    max_e = max(max_e, self.mc[eid])
        return h + max_e

    def _greedy(self, ef, el, pl):
        for pid, loc in pl.items():
            if isinstance(loc, tuple):
                eid = loc[1]
                if ef[eid] == self.pg[pid]:
                    return f"EXIT{{{pid},{eid}}}"

        for pid, loc in pl.items():
            if isinstance(loc, int):
                for eid, efloor in ef.items():
                    if efloor == loc and el[eid] + self.pw[pid] <= self.capacities[eid]:
                        return f"ENTER{{{pid},{eid}}}"

        for eid, floor in ef.items():
            for f in sorted(self.reachable[eid]):
                if f != floor:
                    return f"MOVE{{{eid},{f}}}"

        return "RESET"
