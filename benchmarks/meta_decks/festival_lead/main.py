from _policy import make_agent

CONFIG = {'setup_active': {89: 100, 149: 90, 100: 80, 93: 70}, 'setup_bench': {89: 100, 149: 90, 100: 80, 93: 70}, 'evolve': {90: 6000, 93: 6000, 240: 5000}, 'play': {1086: 4200, 1094: 4200, 1152: 4000, 1227: 3500, 1245: 4000, 1097: 2000, 1174: 1500, 1175: 1500, 1182: 3500, 1210: 3000, 1080: 2000, 1184: 1500, 1191: 1500, 1211: 1500}, 'search': {90: 100, 93: 90, 89: 80, 149: 80, 100: 70, 240: 70, 1: 80}, 'abilities': {90: 5000}, 'energy': [1], 'attach_targets': [93, 240, 89], 'attack_damage': {'Do the Wave': 100, 'Rapid Draw': 60, 'Whirlpool': 30}}
agent = make_agent(CONFIG)
