import torch

from training.schedules import create_lr_scheduler


def test_pc_constant_scheduler_uses_constant_warmup_factory():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))])
    calls = []

    scheduler = create_lr_scheduler(
        "constant",
        optimizer,
        warmup_steps=100,
        max_train_steps=60000,
        cosine_factory=lambda *args: calls.append(("cosine", args)),
        constant_factory=lambda *args: calls.append(("constant", args)) or "constant",
    )

    assert scheduler == "constant"
    assert calls == [("constant", (optimizer, 100))]
