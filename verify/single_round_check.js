const fs = require('fs');
const { Engine, Controller } = require(require('path').join(__dirname, '..', 'src', 'core.js'));
const probs = JSON.parse(fs.readFileSync(require('path').join(__dirname, '..', 'data', 'problems.json'), 'utf8'));
const toProblem = P => ({ height: P.height, horizon: P.horizon, goal_reward: P.goal, Elevators: P.E, Persons: P.P,
  elevator_chosen_action_prob: P.ep, person_chosen_action_prob: P.pp, persons_reward: P.rw });
let unfinished = 0, worst = 0; const out = [];
for (const P of probs) {
  let fin = 0, sum = 0, mx = 0, resets = 0;
  for (let s = 0; s < 10; s++) {
    const g = new Engine(toProblem(P), s + 1), c = new Controller(g, { singleRound: true });
    let done = false;
    for (let i = 0; i < g.maxSteps; i++) {
      const a = c.choose_next_action(g.state); if (a === 'RESET') resets++;
      const o = g.submit(a);
      if (o.goal) { fin++; sum += i + 1; mx = Math.max(mx, i + 1); done = true; break; }
    }
    if (!done) unfinished++;
  }
  worst = Math.max(worst, mx);
  out.push(`${P.name.padEnd(8)} finished ${fin}/10  avg ${(sum / Math.max(1, fin)).toFixed(0).padStart(3)} steps  max ${mx}  resets ${resets}`);
}
console.log(out.join('\n'));
console.log(`unfinished runs: ${unfinished} / ${probs.length * 10}, slowest finish: ${worst} steps`);
