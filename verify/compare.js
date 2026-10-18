// Feed the JavaScript controller the exact observations the Python controller
// saw (recorded by py_trace.py) and check that it picks the same action at
// every step. Usage: node verify/compare.js py_trace.json
const fs = require('fs');
const path = require('path');
const { Controller } = require(path.join(__dirname, '..', 'src', 'core.js'));

const runs = JSON.parse(fs.readFileSync(process.argv[2] || 'py_trace.json', 'utf8'));
let total = 0, bad = 0;
for (const r of runs) {
  let lastR = 0;
  const game = {
    get_reachable: () => r.reach,   // Python's frozenset iteration order
    get_capacities: () => r.cap,
    get_initial_state: () => r.init,
    get_person_weight: p => r.pw[p],
    get_person_goal: p => r.pg[p],
    get_goal_reward: () => r.goal,
    get_max_steps: () => r.horizon,
    get_last_gained_reward: () => lastR,
  };
  const c = new Controller(game, { timeBudget: 1e9 });
  for (let i = 0; i < r.steps.length; i++) {
    const [state, lastGained, pyAction] = r.steps[i];
    lastR = lastGained;
    const jsAction = c.choose_next_action(state);
    total++;
    if (jsAction !== pyAction) {
      bad++;
      console.log(`MISMATCH ${r.name} seed ${r.seed} step ${i}: python=${pyAction} js=${jsAction}`);
      break;
    }
  }
}
console.log(`runs=${runs.length} steps=${total} mismatched_runs=${bad}`);
process.exit(bad ? 1 : 0);
