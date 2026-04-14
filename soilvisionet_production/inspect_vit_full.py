import torch
from pathlib import Path
p = Path("results/vit_phase1/best_model.pt")
print("Resolved path:", p.resolve())
print("Path exists:", p.exists())
if not p.exists():
    p = Path("../results/vit_phase1/best_model.pt")
    print("Alt path:", p.resolve(), "exists:", p.exists())

ckpt = torch.load(p, map_location="cpu")
print("Loaded checkpoint type:", type(ckpt))
if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print("Top-level keys count:", len(keys), "sample:", keys[:50])
    for candidate in ("model_state_dict","state_dict","state_dict_ema","state_dict_raw"):
        if candidate in ckpt:
            sd = ckpt[candidate]
            print(f"Found {candidate} with {len(sd)} keys")
            for k,v in list(sd.items())[:40]:
                print(k, getattr(v,"shape", type(v)))
            break
    else:
        for k,v in list(ckpt.items())[:40]:
            print(k, getattr(v,"shape", type(v)))
else:
    print("Checkpoint is not a dict; repr:", repr(ckpt)[:1000])
