from _policy import make_agent

CONFIG = {'setup_active': {722: 100, 721: 80}, 'setup_bench': {722: 100, 721: 80}, 'evolve': {723: 7000}, 'play': {1145: 4500, 1227: 3500, 1235: 4300, 1205: 4000, 1158: 2000, 1262: 2200}, 'search': {723: 100, 722: 90, 721: 70, 3: 80}, 'energy': [3], 'attach_targets': [723, 722], 'attack_damage': {'Frost Barrier': 200}}
agent = make_agent(CONFIG)
