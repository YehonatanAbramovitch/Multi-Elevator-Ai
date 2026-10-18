# Multi-Elevator AI

**Live demo: https://yehonatanabramovitch.github.io/Multi-Elevator-Ai/**

A browser lab where you build a building, hide how its world behaves, and watch a reinforcement-learning controller figure it out while it delivers people to their floors. The interface is in Hebrew.

## What you can do

1. **Describe the building.** Choose the top floor, the elevators (start floor, the floors each one reaches, weight capacity) and the people (start floor, destination, weight). The controller sees all of this. Or start from one of 36 ready-made problems.
2. **Hide the world.** Set each elevator's chance of a successful move, each person's chance of boarding or exiting, and the list of rewards each delivery can pay. The controller never sees these numbers.
3. **Run it.** The controller starts knowing nothing about the hidden world. You watch it move elevators and board people, and the side panel shows:
   - the action it chose and whether it worked
   - its current strategy and the next steps of its A* plan
   - its estimate of every hidden number next to the true value, converging as it gathers data
   - cumulative reward against a random policy on the same world and seed
4. **Compare.** One click averages 10 runs of the controller against the random policy.

### Two ways a run can end

- **When everyone has arrived** (the default). The run stops as soon as every person reaches their floor, and the lab reports how many steps it took. In this mode the controller always uses the `full` strategy and never resets, since a single round leaves no room for the `loop` strategy to pay off. Everything else is unchanged. On all 36 problems × 10 seeds it finished every run, in at most 50 steps ([`verify/single_round_check.js`](verify/single_round_check.js)). The random policy usually doesn't finish at all.
- **As in the assignment.** After everyone arrives the building resets and the controller keeps collecting reward until the step budget runs out, `loop` strategy included. This is the algorithm exactly as written.

## The domain

Elevators move between the floors they can reach and have a weight limit. Some people can only get to their destination by switching elevators on a shared floor. Actions are `MOVE{e,f}`, `ENTER{p,e}`, `EXIT{p,e}` and `RESET`, and every action costs one step out of a fixed budget.

Actions can fail. A failed move lands the elevator on a random other floor it can reach, and a failed board or exit does nothing. Delivering a person pays a random reward from that person's list. Delivering everyone pays a fixed bonus and resets the building, and the clock keeps running.

## How the controller works

The algorithm is in [`algorithm/controller.py`](algorithm/controller.py). It combines online learning with planning:

- **Learning from experience.** After every action it checks what actually happened. It keeps success/attempt counts for every elevator and person and estimates each success rate as the Beta-posterior mean `(s+1)/(n+2)`. It records every delivery reward it observes, and subtracts the known bonus when a delivery completes the round.
- **Optimism for exploration.** A person it hasn't delivered yet is assumed to be worth `max(bonus, 1)`, so unexplored deliveries look attractive until real samples replace the guess.
- **Planning on what it has learned.** A weighted A* search (W=3) finds the plan with the lowest expected cost, where an action that succeeds with probability *p* costs `1/p` steps. The heuristic counts each person's remaining board/exit costs plus the single largest elevator move still needed. Successors are pruned to moves that pick someone up, drop someone at their destination, or drop them at a floor where another elevator can carry them on.
- **Replanning.** It follows the current plan while reality matches what the plan expected. It replans when something fails, when the plan runs out, and every 8 steps so newer estimates get used. After a search that gives up, it waits a few steps before searching again.
- **Choosing a strategy** (assignment mode). It compares the reward per step of delivering everyone (`full`) with repeatedly delivering only the single most profitable person and resetting (`loop`), and switches between them as its estimates change.

### A bug fix that matters

When the last person in a round is delivered, the building resets immediately, so that person reappears on their starting floor instead of disappearing. The original version only recognized a delivery when the person disappeared, so it never learned the reward of whoever was delivered last. Their estimate stayed stuck at the optimistic guess and sometimes led it into a bad `loop`. The fix recognizes a delivery from the reward itself (delivery rewards are always positive).

Across 36 problems × 30 seeds: 24 problems improved, none got worse, and total reward rose by 5.6%. The largest gain was +70% on `e5_hard` (509 → 867).

## Running Python in the browser: the port

The lab runs a JavaScript port of the Python controller, [`src/core.js`](src/core.js), together with a copy of the environment's rules. The port is checked to behave identically:

- **Same actions, step for step.** [`verify/py_trace.py`](verify/py_trace.py) records the Python controller's observations and actions, and [`verify/compare.js`](verify/compare.js) feeds the same observations to the JavaScript controller. On all 36 problems × 2 seeds, **all 27,000 actions match**.
- **Same tie-breaking.** The Python code iterates over `frozenset`s of floors, whose order depends on CPython's hash table (for example `{2,4,6,8}` iterates as `8,2,4,6`). `pySetOrder` in `core.js` reproduces that order. It matches CPython on every set from the 36 problems and on 3,000 random sets ([`verify/pyset_check.js`](verify/pyset_check.js)).

The simulation runs in a Web Worker, so the animation stays smooth during long A* searches.

## Running locally

Open `index.html` in a browser. It is a single self-contained file.

To rebuild it after editing `src/`:

```bash
python src/build.py
```

To rerun the verification, you need the original course environment (`ext_elev.py` and `ex3_check.py`), which isn't included here:

```bash
python verify/py_trace.py <course-ex3-folder> algorithm/controller.py py_trace.json 2
node verify/compare.js py_trace.json
```

## Project layout

```
index.html            the lab (built, self-contained)
algorithm/controller.py   the reinforcement-learning controller (Python)
src/core.js           JavaScript port of the controller + environment rules
src/lab_template.html the lab's UI
src/build.py          builds index.html
data/                 the 36 problems and the Python benchmark averages
verify/               scripts that check the port against Python
```

## Background

The controller is my solution to the reinforcement-learning assignment in Bar-Ilan University's Introduction to Artificial Intelligence course (CS 89-570). The environment's rules and the 36 problems come from the course. I used Claude Code as a development aid for design discussion, debugging, the JavaScript port and the lab interface.

If you are taking this course, do not submit this code, or anything derived from it, as your own work.
