"""Learning-rate scheduler selection for the I2V overfit trainer."""


def create_lr_scheduler(
    schedule_name,
    optimizer,
    warmup_steps: int,
    max_train_steps: int,
    *,
    cosine_factory,
    constant_factory,
):
    """Create a cosine or constant-after-warmup learning-rate scheduler."""
    if schedule_name == "cosine":
        return cosine_factory(optimizer, warmup_steps, max_train_steps)
    if schedule_name == "constant":
        return constant_factory(optimizer, warmup_steps)
    raise ValueError(f"Unsupported learning-rate scheduler: {schedule_name!r}")
