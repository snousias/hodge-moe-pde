# ========================================================================================
# TRAINER, vef   —   fp32, the released recipe
# ========================================================================================
"""fp32 trainer for `config.toml` / `VARIANT=vef`.

This is a delegate. The loop itself is `source/training/loop.py`, unchanged: it is the code
that produced `checkpoints/hodgemoe_vef_best_ep400.pth`, and the repository's invariant is that
it stays identical to the code behind the released checkpoint. Nothing is wrapped, nothing is
overridden; `main` below IS `training.loop.main`.

The companion case is `cases/v`, which runs the same loop under bf16 autocast.

    python train.py --variant vef --out runs/train_vef/
    VARIANT=vef bash scripts/run_train.sh
"""
from training.loop import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
