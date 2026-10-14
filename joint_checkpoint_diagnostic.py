"""Reproduce the checkpointed 3-slot history/Utonia backward pass on one GPU."""

import torch

from wan.modules.joint_wan_physctrl import JointWanPhysCtrlModel
from wan.modules.model import WanModel
from wan.modules.pc_trajectory import PCTrajectoryModel


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("joint checkpoint diagnostic requires CUDA")

    torch.manual_seed(0)
    device = torch.device("cuda")
    wan = WanModel(
        model_type="ti2v",
        patch_size=(1, 2, 2),
        text_len=2,
        in_dim=1,
        dim=64,
        ffn_dim=128,
        freq_dim=16,
        text_dim=4,
        out_dim=1,
        num_heads=1,
        num_layers=8,
    )
    model = JointWanPhysCtrlModel(
        wan,
        PCTrajectoryModel(
            n_points=2,
            n_future_frames=45,
            latent_dim=64,
            n_layers=8,
            num_heads=1,
            objective_type="ddpm",
            conditioning="history",
            history_frames=4,
            utonia_feature_dim=3,
        ),
    ).to(device)
    model.train()
    model.wan_model.gradient_checkpointing = True

    video, trajectories = model(
        video_x=[torch.randn(1, 1, 2, 2, device=device)],
        video_t=torch.tensor([[1.0]], device=device),
        context=[torch.randn(2, 4, device=device)],
        seq_len=1,
        noisy_future_state=torch.randn(1, 3, 45, 1, 2, 3, device=device),
        frame_times=torch.cat(
            (
                torch.zeros(1, 3, 4, device=device),
                torch.ones(1, 3, 45, device=device),
            ),
            dim=2,
        ),
        init_pc=torch.randn(1, 3, 4, 1, 2, 3, device=device),
        utonia_features=torch.randn(1, 3, 2, 3, device=device),
    )
    (video[0].square().mean() + trajectories.square().mean()).backward()

    if model.wan_model.blocks[0].self_attn.q.weight.grad is None:
        raise AssertionError("Wan attention did not receive a gradient")
    if model.pc_model.blocks[0].norm1.linear.weight.grad is None:
        raise AssertionError("PhysCtrl block did not receive a gradient")
    print("PASS: paired checkpoint backward completed", flush=True)


if __name__ == "__main__":
    main()
