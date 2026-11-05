# ========================================================================================
# EVALUATORS, vef   —   fp32, the released code
# ========================================================================================
"""fp32 evaluation for `config.toml` / `VARIANT=vef`.

Delegates. `source/evaluation/fields.py` and `source/evaluation/lift.py` are the released
evaluators, and they produced every number in `results/eval_release/`; nothing is wrapped or
overridden here. The companion case is `cases/v`, which runs the same evaluators with the
forward pass under bf16 autocast, because that is the precision its published metrics were
measured in.

    python eval_fields.py --variant vef --out runs/eval_vef/fields
    VARIANT=vef bash scripts/run_eval.sh
"""
__all__ = ["fields_main", "lift_main"]


def fields_main():
    from evaluation.fields import main
    return main()


def lift_main():
    from evaluation.lift import main
    return main()
