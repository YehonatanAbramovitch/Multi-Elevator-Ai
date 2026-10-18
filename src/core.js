// JavaScript port of Ex3/ex3_solution.py (Controller) plus a copy of the
// Ex3 engine rules (ext_elev.py). Person locations are encoded as numbers:
// a floor f >= 0, or -(e + 1) when inside elevator e.
// Objects keyed by integer ids iterate in ascending id order, which matches
// the Python dict order (built from id-sorted state tuples).

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// Iteration order of CPython's frozenset(<tuple of small non-negative ints>):
// open addressing with linear probing (LINEAR_PROBES=9) and perturbation,
// table starts at 8 slots and grows x4 when fill*5 >= mask*3.
function pySetOrder(items) {
  let mask = 7, table = new Array(8).fill(null), fill = 0;
  const insert = (tab, m, key) => {
    let perturb = key, i = key & m;
    for (;;) {
      if (tab[i] === null) { tab[i] = key; return true; }
      if (tab[i] === key) return false;
      if (i + 9 <= m) {
        for (let j = 1; j <= 9; j++) {
          if (tab[i + j] === null) { tab[i + j] = key; return true; }
          if (tab[i + j] === key) return false;
        }
      }
      perturb = Math.floor(perturb / 32);
      i = (i * 5 + 1 + perturb) & m;
    }
  };
  for (const k of items) {
    if (insert(table, mask, k)) {
      fill++;
      if (fill * 5 >= mask * 3) {
        let ns = 8; while (ns <= fill * 4) ns <<= 1;
        const old = table; table = new Array(ns).fill(null); mask = ns - 1;
        for (const v of old) if (v !== null) insert(table, mask, v);
      }
    }
  }
  return table.filter(v => v !== null);
}

// ------------------------------------------------------------------ //
// Engine (mirrors ext_elev.Game)                                     //
// ------------------------------------------------------------------ //
class Engine {
  constructor(problem, seed) {
    this.problem = problem;
    this.rng = mulberry32(seed >>> 0);
    this.maxSteps = problem.horizon;
    this.goalReward = problem.goal_reward;
    this.eids = Object.keys(problem.Elevators).map(Number).sort((a, b) => a - b);
    this.pids = Object.keys(problem.Persons).map(Number).sort((a, b) => a - b);
    this.reach = {}; this.cap = {}; this.pw = {}; this.pg = {};
    for (const e of this.eids) {
      const [, F, c] = problem.Elevators[e];
      this.reach[e] = pySetOrder(F); this.cap[e] = c;  // frozenset iteration order, as in ext_elev
    }
    for (const p of this.pids) {
      const [, w, g] = problem.Persons[p];
      this.pw[p] = w; this.pg[p] = g;
    }
    this.initial = {
      e: this.eids.map(e => [e, problem.Elevators[e][0], 0]),
      p: this.pids.map(p => [p, 'floor', problem.Persons[p][0]]),
      rem: this.pids.length,
    };
    this.state = this.initial;
    this.steps = 0; this.reward = 0; this.lastGained = 0; this.done = false;
    this.lastOutcome = null;
  }

  submit(action) {
    if (this.done) return;
    const before = this.reward;
    let out;
    if (action === 'RESET') { this.state = this.initial; out = { kind: 'reset' }; }
    else {
      const m = action.match(/^(MOVE|ENTER|EXIT)\{(-?\d+),(-?\d+)\}$/);
      if (!m) throw new Error('Unknown action ' + action);
      const a = +m[2], b = +m[3];
      if (m[1] === 'MOVE') out = this._move(a, b);
      else if (m[1] === 'ENTER') out = this._enter(a, b);
      else out = this._exit(a, b);
    }
    this.steps++;
    this.lastGained = this.reward - before;
    if (this.steps >= this.maxSteps) this.done = true;
    this.lastOutcome = out;
    return out;
  }

  _elev(e) { return this.state.e.find(x => x[0] === e); }
  _person(p) { return this.state.p.find(x => x[0] === p); }

  _move(e, f) {
    if (!this.reach[e] || !this.reach[e].includes(f)) throw new Error(`Illegal MOVE{${e},${f}}`);
    const [, cur, w] = this._elev(e);
    let nf, ok;
    if (this.rng() < this.problem.elevator_chosen_action_prob[e]) { nf = f; ok = true; }
    else {
      const opts = Array.from(new Set([cur].concat(this.reach[e].filter(x => x !== f)))).sort((a, b) => a - b);
      nf = opts[Math.floor(this.rng() * opts.length)]; ok = false;
    }
    this.state = { e: this.state.e.map(x => x[0] === e ? [e, nf, w] : x), p: this.state.p, rem: this.state.rem };
    return { kind: 'move', ok, from: cur, to: f, landed: nf };
  }

  _enter(p, e) {
    const pr = this._person(p), [, cur, w] = this._elev(e);
    if (!pr || pr[1] !== 'floor') throw new Error(`Illegal ENTER{${p},${e}}: not on a floor`);
    if (pr[2] !== cur) throw new Error(`Illegal ENTER{${p},${e}}: different floor`);
    if (w + this.pw[p] > this.cap[e]) throw new Error(`Illegal ENTER{${p},${e}}: capacity`);
    if (this.rng() < this.problem.person_chosen_action_prob[p]) {
      this.state = {
        e: this.state.e.map(x => x[0] === e ? [e, cur, w + this.pw[p]] : x),
        p: this.state.p.map(x => x[0] === p ? [p, 'in', e] : x),
        rem: this.state.rem,
      };
      return { kind: 'enter', ok: true };
    }
    return { kind: 'enter', ok: false };
  }

  _exit(p, e) {
    const pr = this._person(p), [, cur, w] = this._elev(e);
    if (!pr || pr[1] !== 'in' || pr[2] !== e) throw new Error(`Illegal EXIT{${p},${e}}`);
    if (this.rng() < this.problem.person_chosen_action_prob[p]) {
      const elevs = this.state.e.map(x => x[0] === e ? [e, cur, w - this.pw[p]] : x);
      if (cur === this.pg[p]) {
        const list = this.problem.persons_reward[p];
        const r = list[Math.floor(this.rng() * list.length)];
        this.reward += r;
        const rem = this.state.rem - 1;
        if (rem === 0) {
          this.reward += this.goalReward;
          this.state = this.initial;
          return { kind: 'exit', ok: true, delivered: true, r, goal: true, floor: cur };
        }
        this.state = { e: elevs, p: this.state.p.filter(x => x[0] !== p), rem };
        return { kind: 'exit', ok: true, delivered: true, r, goal: false, floor: cur };
      }
      this.state = { e: elevs, p: this.state.p.map(x => x[0] === p ? [p, 'floor', cur] : x), rem: this.state.rem };
      return { kind: 'exit', ok: true, delivered: false, floor: cur };
    }
    return { kind: 'exit', ok: false };
  }

  // GameAPI surface used by the controller
  get_reachable() { return this.reach; }
  get_capacities() { return this.cap; }
  get_initial_state() { return this.initial; }
  get_person_weight(p) { return this.pw[p]; }
  get_person_goal(p) { return this.pg[p]; }
  get_goal_reward() { return this.goalReward; }
  get_max_steps() { return this.maxSteps; }
  get_last_gained_reward() { return this.lastGained; }
}

// ------------------------------------------------------------------ //
// Binary heap ordered by (f, cnt), like heapq on (f, cnt, ...)        //
// ------------------------------------------------------------------ //
class Heap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  less(x, y) { return x[0] < y[0] || (x[0] === y[0] && x[1] < y[1]); }
  push(v) {
    const a = this.a; a.push(v); let i = a.length - 1;
    while (i > 0) { const j = (i - 1) >> 1; if (this.less(a[i], a[j])) { [a[i], a[j]] = [a[j], a[i]]; i = j; } else break; }
  }
  pop() {
    const a = this.a, top = a[0], last = a.pop();
    if (a.length) {
      a[0] = last; let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < a.length && this.less(a[l], a[m])) m = l;
        if (r < a.length && this.less(a[r], a[m])) m = r;
        if (m === i) break; [a[i], a[m]] = [a[m], a[i]]; i = m;
      }
    }
    return top;
  }
}

const isFloor = v => v >= 0;
const inElev = v => -v - 1;
const encIn = e => -(e + 1);
const objEq = (a, b) => {
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (const k of ka) if (a[k] !== b[k]) return false;
  return true;
};
const stEq = (x, y) => objEq(x[0], y[0]) && objEq(x[1], y[1]) && objEq(x[2], y[2]);

// ------------------------------------------------------------------ //
// Controller (port of ex3_solution.Controller)                        //
// ------------------------------------------------------------------ //
class Controller {
  constructor(game, opts = {}) {
    this.REPLAN_PERIOD = 8;
    this.ASTAR_TIME_BUDGET = opts.timeBudget != null ? opts.timeBudget : 1.0;
    // Lab-only option: when the run ends after a single round, the RESET-loop
    // strategy cannot pay off, so the strategy choice always picks 'full'.
    // With singleRound false (the default) behaviour matches controller.py.
    this.singleRound = !!opts.singleRound;
    this.now = opts.now || (() => (typeof performance !== 'undefined' ? performance.now() : Date.now()) / 1000);

    this.game = game;
    const reach = game.get_reachable();
    this.capacities = game.get_capacities();
    this.initial_state = game.get_initial_state();
    const [elevators0, persons0] = [this.initial_state.e, this.initial_state.p];
    this.all_eids = elevators0.map(x => x[0]);
    this.all_pids = persons0.map(x => x[0]);
    // reachOrder: iteration order for MOVE generation; reachSet for membership
    this.reachOrder = {}; this.reachSet = {};
    for (const e of this.all_eids) { this.reachOrder[e] = reach[e].slice(); this.reachSet[e] = new Set(reach[e]); }
    this.initial_ef = {}; this.initial_el = {}; this.initial_floor = {};
    for (const [e, f] of elevators0) { this.initial_ef[e] = f; this.initial_el[e] = 0; }
    for (const [p, , f] of persons0) this.initial_floor[p] = f;

    this.pw = {}; this.pg = {};
    for (const p of this.all_pids) { this.pw[p] = game.get_person_weight(p); this.pg[p] = game.get_person_goal(p); }
    this.goal_r = game.get_goal_reward();
    this.horizon = game.get_max_steps();

    // connectivity (floors reachable through transfers)
    const adj = {};
    for (const e1 of this.all_eids) {
      adj[e1] = new Set();
      for (const e2 of this.all_eids) if (e1 !== e2 && [...this.reachSet[e1]].some(f => this.reachSet[e2].has(f))) adj[e1].add(e2);
    }
    this.ecr = {};
    for (const e of this.all_eids) {
      const vis = new Set([e]), q = [e], floors = new Set(this.reachSet[e]);
      while (q.length) {
        const c = q.pop();
        for (const f of this.reachSet[c]) floors.add(f);
        for (const n of adj[c]) if (!vis.has(n)) { vis.add(n); q.push(n); }
      }
      this.ecr[e] = floors;
    }

    this.e_succ = {}; this.e_att = {}; this.p_succ = {}; this.p_att = {}; this.p_rew_sum = {}; this.p_rew_n = {};
    for (const e of this.all_eids) { this.e_succ[e] = 0; this.e_att[e] = 0; }
    for (const p of this.all_pids) { this.p_succ[p] = 0; this.p_att[p] = 0; this.p_rew_sum[p] = 0; this.p_rew_n[p] = 0; }
    this._reward_prior = Math.max(this.goal_r, 1.0);
    this.mc = {}; this.pc = {};
    for (const e of this.all_eids) this.mc[e] = 1.0;
    for (const p of this.all_pids) this.pc[p] = 1.0;

    this.prev_action = null; this.prev_ef = null; this.prev_total_remaining = null;
    this._strategy = 'full'; this._loop_pid = null;
    this._plan_actions = []; this._plan_states = []; this._plan_idx = 0;
    this._calls_since_replan = 0; this._astar_cooldown = 0;
  }

  static _parse_pair(act) {
    const m = act.match(/\{(-?\d+),(-?\d+)\}/);
    return [+m[1], +m[2]];
  }

  _learn_from_last_action(state) {
    if (this.prev_action === null) return;
    const act = this.prev_action;
    if (act === 'RESET') return;
    if (act.startsWith('MOVE')) {
      const [e, tgt] = Controller._parse_pair(act);
      const new_f = state.e.find(x => x[0] === e)[1];
      const origin = this.prev_ef[e];
      if (tgt !== origin) { this.e_att[e] += 1; if (new_f === tgt) this.e_succ[e] += 1; }
    } else if (act.startsWith('ENTER')) {
      const [p, e] = Controller._parse_pair(act);
      const loc = state.p.find(x => x[0] === p);
      this.p_att[p] += 1;
      if (loc[1] === 'in' && loc[2] === e) this.p_succ[p] += 1;
    } else if (act.startsWith('EXIT')) {
      const [p] = Controller._parse_pair(act);
      const found = state.p.find(x => x[0] === p) || null;
      this.p_att[p] += 1;
      let last_r = this.game.get_last_gained_reward();
      if (last_r > 0) {
        this.p_succ[p] += 1;
        if (this.prev_total_remaining === 1) last_r -= this.goal_r;
        this.p_rew_sum[p] += last_r;
        this.p_rew_n[p] += 1;
      } else if (found !== null && found[1] === 'floor') {
        this.p_succ[p] += 1;
      }
    }
  }

  _sample_probs() {
    for (const e of this.all_eids) {
      const p = (this.e_succ[e] + 1) / (this.e_att[e] + 2);
      this.mc[e] = 1.0 / Math.max(p, 0.05);
    }
    for (const q of this.all_pids) {
      const p = (this.p_succ[q] + 1) / (this.p_att[q] + 2);
      this.pc[q] = 1.0 / Math.max(p, 0.05);
    }
  }

  _reward_estimate(p) {
    const n = this.p_rew_n[p];
    if (n === 0) return this._reward_prior;
    return (this.p_rew_sum[p] + this._reward_prior) / (n + 1);
  }

  choose_next_action(state) {
    this._learn_from_last_action(state);
    const total_remaining = state.rem;
    if (!state.p.length) {
      this.prev_action = 'RESET';
      this.prev_ef = {}; for (const [e, f] of state.e) this.prev_ef[e] = f;
      this.prev_total_remaining = total_remaining;
      return 'RESET';
    }
    const ef = {}, el = {}, pl = {};
    for (const [e, f, w] of state.e) { ef[e] = f; el[e] = w; }
    for (const [p, k, v] of state.p) pl[p] = k === 'floor' ? v : encIn(v);

    const need_strategy_refresh = this._calls_since_replan >= this.REPLAN_PERIOD;
    this._calls_since_replan += 1;

    let need_reroute = need_strategy_refresh || this._plan_idx >= this._plan_actions.length;
    if (!need_reroute) {
      const idx = this._plan_idx;
      if (this._strategy === 'loop') {
        const pid = this._loop_pid;
        const cur = (pid in pl) ? [ef, el, { [pid]: pl[pid] }] : null;
        need_reroute = cur === null || !(idx < this._plan_states.length && stEq(this._plan_states[idx], cur));
      } else {
        need_reroute = !(idx < this._plan_states.length && stEq(this._plan_states[idx], [ef, el, pl]));
      }
    }

    if (need_strategy_refresh || need_reroute) this._sample_probs();
    if (need_strategy_refresh) { this._decide_strategy(ef, el, pl); this._calls_since_replan = 0; need_reroute = true; }
    this.lastReplanned = false;
    if (need_reroute) {
      if (this._astar_cooldown > 0) this._astar_cooldown -= 1;
      else { this._reroute(ef, el, pl); this.lastReplanned = true; }
    }

    let action;
    if (this._strategy === 'loop' && this._plan_idx === 0 && !this._isInitial(state)) action = 'RESET';
    else {
      action = this._take_from_plan();
      if (action === null) action = this._strategy === 'loop' ? 'RESET' : this._greedy(ef, el, pl);
    }
    this.prev_action = action; this.prev_ef = ef; this.prev_total_remaining = total_remaining;
    return action;
  }

  _isInitial(state) {
    const I = this.initial_state;
    if (state.rem !== I.rem || state.e.length !== I.e.length || state.p.length !== I.p.length) return false;
    for (let i = 0; i < I.e.length; i++) for (let j = 0; j < 3; j++) if (state.e[i][j] !== I.e[i][j]) return false;
    for (let i = 0; i < I.p.length; i++) for (let j = 0; j < 3; j++) if (state.p[i][j] !== I.p[i][j]) return false;
    return true;
  }

  _take_from_plan() {
    if (this._plan_idx < this._plan_actions.length) return this._plan_actions[this._plan_idx++];
    return null;
  }

  _decide_strategy(ef, el, pl) {
    if (this.singleRound) { this._strategy = 'full'; this._loop_pid = null; return; }
    const apr = {};
    for (const p of this.all_pids) apr[p] = this._reward_estimate(p);
    const full_cost = this._h(ef, pl);
    let full_rew = 0; for (const p in pl) full_rew += apr[p]; full_rew += this.goal_r;
    const full_vps = full_cost > 0 ? full_rew / full_cost : Infinity;
    let best_vps = -1.0, best_pid = null;
    for (const pid of this.all_pids) {
      const cost = this._h(this.initial_ef, { [pid]: this.initial_floor[pid] }) + 1.0;
      const vps = apr[pid] / cost;
      if (vps > best_vps) { best_vps = vps; best_pid = pid; }
    }
    if (best_pid !== null && best_vps > full_vps) { this._strategy = 'loop'; this._loop_pid = best_pid; }
    else { this._strategy = 'full'; this._loop_pid = null; }
  }

  _reroute(ef, el, pl) {
    let res;
    if (this._strategy === 'loop') {
      const pid = this._loop_pid;
      res = this._astar(this.initial_ef, this.initial_el, { [pid]: this.initial_floor[pid] });
    } else res = this._astar(ef, el, pl);
    this._plan_actions = res[0]; this._plan_states = res[1]; this._plan_idx = 0;
    if (!res[0].length) this._astar_cooldown = this.REPLAN_PERIOD;
  }

  _isGoal(pl) { for (const p in pl) { const v = pl[p]; if (!(isFloor(v) && v === this.pg[p])) return false; } return true; }
  _sk(ef, pl) { let s = ''; for (const e in ef) s += ef[e] + ','; s += '|'; for (const p in pl) s += p + ':' + pl[p] + ','; return s; }

  _astar(ef0, el0, pl0) {
    const start = [ef0, el0, pl0];
    if (this._isGoal(pl0)) return [[], [start]];
    const W = 3.0;
    let cnt = 0;
    const heap = new Heap();
    heap.push([W * this._h(ef0, pl0), cnt, this._sk(ef0, pl0), ef0, el0, pl0, 0.0, null]);
    const best = new Map();
    let expansions = 0;
    const deadline = this.now() + this.ASTAR_TIME_BUDGET;
    while (heap.size) {
      const node = heap.pop();
      const [, , sk, ef, el, pl, g] = node;
      const b = best.get(sk);
      if (b !== undefined && b <= g) continue;
      best.set(sk, g);
      if (this._isGoal(pl)) {
        const acts = [], states = [];
        for (let n = node; n; n = n[7]) { states.push([n[3], n[4], n[5]]); if (n[8] !== undefined) acts.push(n[8]); }
        acts.reverse(); states.reverse();
        return [acts, states];
      }
      expansions += 1;
      if (expansions > 20000 || (expansions % 256 === 0 && this.now() > deadline)) return [[], [start]];
      for (const [act, nef, nel, npl, cost] of this._succs(ef, el, pl)) {
        const nsk = this._sk(nef, npl), ng = g + cost;
        const bb = best.get(nsk);
        if (bb === undefined || bb > ng) {
          cnt += 1;
          heap.push([ng + W * this._h(nef, npl), cnt, nsk, nef, nel, npl, ng, node, act]);
        }
      }
    }
    return [[], [start]];
  }

  _succs(ef, el, pl) {
    const res = [], fp = {}, ep_ = {};
    for (const e in ef) ep_[e] = [];
    for (const p in pl) {
      const v = pl[p];
      if (isFloor(v)) (fp[v] || (fp[v] = [])).push(+p);
      else ep_[inElev(v)].push(+p);
    }
    const eids = Object.keys(ef).map(Number);
    for (const eid of eids) {
      const floor = ef[eid];
      for (const tgt of this.reachOrder[eid]) {
        if (tgt === floor) continue;
        const useful =
          (fp[tgt] || []).some(p => el[eid] + this.pw[p] <= this.capacities[eid] && this.ecr[eid].has(this.pg[p])) ||
          ep_[eid].some(p => this.pg[p] === tgt) ||
          ep_[eid].some(p => !this.reachSet[eid].has(this.pg[p]) &&
            eids.some(e2 => e2 !== eid && this.reachSet[e2].has(tgt) && this.ecr[e2].has(this.pg[p])));
        if (useful) res.push([`MOVE{${eid},${tgt}}`, { ...ef, [eid]: tgt }, el, pl, this.mc[eid]]);
      }
    }
    for (const eid of eids) {
      const floor = ef[eid];
      for (const pid of (fp[floor] || [])) {
        if (el[eid] + this.pw[pid] <= this.capacities[eid] && this.ecr[eid].has(this.pg[pid])) {
          res.push([`ENTER{${pid},${eid}}`, ef, { ...el, [eid]: el[eid] + this.pw[pid] }, { ...pl, [pid]: encIn(eid) }, this.pc[pid]]);
        }
      }
    }
    for (const ps in pl) {
      const v = pl[ps]; if (isFloor(v)) continue;
      const pid = +ps, eid = inElev(v), floor = ef[eid], goal = this.pg[pid];
      const should_exit = floor === goal ||
        (!this.reachSet[eid].has(goal) && eids.some(e2 => e2 !== eid && this.reachSet[e2].has(floor) && this.ecr[e2].has(goal)));
      if (should_exit) res.push([`EXIT{${pid},${eid}}`, ef, { ...el, [eid]: el[eid] - this.pw[pid] }, { ...pl, [pid]: floor }, this.pc[pid]]);
    }
    return res;
  }

  _h(ef, pl) {
    let h = 0.0, max_e = 0.0;
    for (const ps in pl) {
      const v = pl[ps], goal = this.pg[ps];
      if (isFloor(v)) {
        if (v === goal) continue;
        const f = v;
        h += 2.0 * this.pc[ps];
        let best = null;
        for (const e in ef) {
          if (this.reachSet[e].has(f) && this.reachSet[e].has(goal)) {
            const c = (ef[e] !== f ? this.mc[e] : 0.0) + this.mc[e];
            if (best === null || c < best) best = c;
          }
        }
        if (best !== null) max_e = Math.max(max_e, best);
      } else {
        const eid = inElev(v);
        h += this.pc[ps];
        if (ef[eid] !== goal) max_e = Math.max(max_e, this.mc[eid]);
      }
    }
    return h + max_e;
  }

  _greedy(ef, el, pl) {
    for (const ps in pl) { const v = pl[ps]; if (!isFloor(v)) { const eid = inElev(v); if (ef[eid] === this.pg[ps]) return `EXIT{${ps},${eid}}`; } }
    for (const ps in pl) {
      const v = pl[ps];
      if (isFloor(v)) for (const e in ef) if (ef[e] === v && el[e] + this.pw[ps] <= this.capacities[e]) return `ENTER{${ps},${e}}`;
    }
    for (const e in ef) { const floor = ef[e]; for (const f of [...this.reachSet[e]].sort((a, b) => a - b)) if (f !== floor) return `MOVE{${e},${f}}`; }
    return 'RESET';
  }

  // snapshot of what the controller currently believes (for the UI)
  beliefs() {
    const ep = {}, en = {}, pp = {}, pn = {}, rw = {}, rn = {};
    for (const e of this.all_eids) { ep[e] = (this.e_succ[e] + 1) / (this.e_att[e] + 2); en[e] = this.e_att[e]; }
    for (const p of this.all_pids) { pp[p] = (this.p_succ[p] + 1) / (this.p_att[p] + 2); pn[p] = this.p_att[p]; rw[p] = this._reward_estimate(p); rn[p] = this.p_rew_n[p]; }
    return { ep, en, pp, pn, rw, rn };
  }
  upcoming(n) { return this._plan_actions.slice(this._plan_idx, this._plan_idx + n); }
}

if (typeof module !== 'undefined') module.exports = { Engine, Controller, mulberry32, pySetOrder };
