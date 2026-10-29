"""The processes and settings of the de Bruijn section (section 10)."""
F = lambda n: {"F": [1.0 / n] * n}
CONFIGS = {
    "renewal-5": dict(process="renewal", params=F(5)),
    "renewal-10": dict(process="renewal", params=F(10)),
    "renewal-15": dict(process="renewal", params=F(15)),
    "renewal-30": dict(process="renewal", params=F(30), history_len=60, rollout_len=30, n_states=64, context=29),
    "renewal-60": dict(process="renewal", params=F(60), history_len=120, rollout_len=60, n_states=128, context=59),
    "flower(3,5)": dict(process="flower", params={"n": 3, "m": 5, "dice_seed": 0}),
    "flower(3,5)-backward": dict(process="flower", params={"n": 3, "m": 5, "dice_seed": 0, "direction": "backward"}),
    "flower(8,14)": dict(process="flower", params={"n": 8, "m": 14, "dice_seed": 0}),
    "flower(8,14)-backward": dict(process="flower", params={"n": 8, "m": 14, "dice_seed": 0, "direction": "backward"})}
MODELS = ["de_bruijn", "de_bruijn_bic"]
