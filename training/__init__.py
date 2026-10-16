"""TRAINING — losses, metrics, schedules and the epoch loops.

    ENTRY POINT:  training.loop.main()      (invoked by ../train.py)

    loop.py       train_epoch / validate_epoch / main: data init, optimiser, schedule, checkpoints
    losses.py     helpers — compute_field_loss, band_loss, cl_loss, and the target-column splitters
    metrics.py    helpers — regression_metrics, append_metrics (per-epoch metrics.jsonl/csv)
    schedules.py  helper  — make_warmup_flat_cosine
"""
