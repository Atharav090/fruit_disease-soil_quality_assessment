import torch
from pathlib import Path
p = Path("results/vit_phase1/best_model.pt")
print("Path exists:", p.exists())
ckpt = torch.load(p, map_location="cpu")
print("Type:", type(ckpt))
if isinstance(ckpt, dict):
    print("Top-level keys:", list(ckpt.keys())[:50])
    for candidate in ("model_state_dict","state_dict","state_dict_ema"):
        if candidate in ckpt:
            sd = ckpt[candidate]
            print(f"Found {candidate}: {len(sd)} keys")
            for k,v in list(sd.items())[:20]:
                print(k, getattr(v,"shape", type(v)))
            break
    else:
        for k,v in list(ckpt.items())[:20]:
            print(k, getattr(v,"shape", type(v)))
else:
    print(repr(ckpt)[:1000])
