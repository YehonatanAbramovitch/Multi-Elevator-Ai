# Record (state, last_gained_reward, action) for the Python controller with
# the A* wall-clock budget disabled, so runs are deterministic.
import sys, json, importlib.util
ex3_dir, ctrl_file, out, seeds = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, ex3_dir)
import ext_elev
spec = importlib.util.spec_from_file_location("ctrl", ctrl_file)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.Controller.ASTAR_TIME_BUDGET = 1e9
sys.modules["ex3"] = m
import ex3_check
def enc(s):
    el, ps, rem = s
    return {"e": [[e, f, w] for e, f, w in el], "p": [[p, loc[0], loc[1]] for p, loc in ps], "rem": rem}
res = []
for name, prob in ex3_check.PROBLEMS:
    for seed in range(seeds):
        prob["seed"] = seed
        api = ext_elev.create_elevators_game(prob, debug=False)
        c = m.Controller(api)
        steps = []
        for t in range(api.get_max_steps()):
            st = api.get_current_state(); lr = api.get_last_gained_reward()
            a = c.choose_next_action(st)
            steps.append([enc(st), lr, a])
            api.submit_next_action(a)
        res.append({"name": name, "seed": seed,
                    "reach": {str(e): list(fs) for e, fs in api.get_reachable().items()},
                    "cap": {str(e): v for e, v in api.get_capacities().items()},
                    "pw": {str(p): prob["Persons"][p][1] for p in prob["Persons"]},
                    "pg": {str(p): prob["Persons"][p][2] for p in prob["Persons"]},
                    "goal": prob["goal_reward"], "horizon": prob["horizon"],
                    "init": enc(api.get_initial_state()), "steps": steps})
        print(name, seed, flush=True)
json.dump(res, open(out, "w"), separators=(",", ":"))
