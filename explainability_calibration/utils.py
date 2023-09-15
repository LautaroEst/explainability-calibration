from itertools import product

def create_hyperparams_grid(hyperparams, n=10, random_state=None):
    keys = sorted(hyperparams.keys())
    grid = [{k: values[i] for i, k in enumerate(keys)} for values in product(*[hyperparams[name] for name in keys])]
    rnd_idx = random_state.permutation(len(grid))[:n]
    return [grid[idx] for idx in rnd_idx]
