from _policy import make_agent

CONFIG = {'setup_active': {319: 100, 675: 60, 676: 60, 81: 50}, 'setup_bench': {319: 100, 675: 80, 676: 80, 81: 60}, 'evolve': {320: 8000}, 'play': {1142: 4200, 1152: 4000, 1121: 4000, 1185: 4000, 1192: 3500, 1182: 3500, 1430: 3000, 1256: 2500, 1128: 2500, 1123: 1500}, 'search': {320: 100, 319: 90, 675: 70, 676: 70, 6: 80, 2: 80, 11: 60}, 'abilities': {675: 5000}, 'energy': [6, 2, 11], 'attach_targets': [320, 675, 676, 81], 'attack_damage': {'Abyssal Flames': 200, 'Raging Amethyst': 280}}
agent = make_agent(CONFIG)
