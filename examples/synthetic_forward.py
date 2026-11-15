"""Run a deterministic CPU-only forward pass with synthetic graph inputs."""

from __future__ import annotations

import torch

from gnn_for_futures import (
    ShockFollowGraphMixerV2,
    hierarchical_joint_scores,
    hierarchical_predictions,
)


def main() -> int:
    torch.manual_seed(20261007)
    device = torch.device("cpu")
    batch_size, node_count, sequence_length, feature_count = 1, 4, 6, 3

    model = ShockFollowGraphMixerV2(
        node_feature_dim=feature_count,
        temporal_hidden_dim=5,
        attention_history_length=sequence_length,
        expected_num_futures=node_count,
        role_dim=7,
        neighbor_capacity=2,
        head_hidden_dim=11,
    ).to(device)
    model.eval()

    node_history = torch.randn(
        batch_size,
        node_count,
        sequence_length,
        feature_count,
        device=device,
    )
    diagonal = torch.eye(node_count, dtype=torch.bool, device=device)
    off_diagonal = ~diagonal
    economic_prior = off_diagonal.to(torch.float32)
    regression_weight = torch.rand(node_count, node_count, device=device)
    regression_weight.masked_fill_(diagonal, 0.0)
    prior_available = off_diagonal
    regression_available = off_diagonal.unsqueeze(0)

    with torch.inference_mode():
        output = model(
            node_history,
            economic_prior,
            regression_weight,
            prior_available=prior_available,
            regression_available=regression_available,
        )

    expected_shapes = {
        "follow_logits": (batch_size, node_count, node_count, 2),
        "direction_logits": (batch_size, node_count, node_count, 2),
        "joint_scores": (batch_size, node_count, node_count, 3),
    }
    for name, expected in expected_shapes.items():
        tensor = getattr(output, name)
        if tuple(tensor.shape) != expected:
            raise AssertionError(f"{name}: expected {expected}, got {tuple(tensor.shape)}")
        if tensor.device.type != "cpu" or not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} must be finite and CPU-resident")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise AssertionError("all model parameters must remain on CPU")

    recomposed = hierarchical_joint_scores(
        output.follow_logits,
        output.direction_logits,
    )
    if not torch.allclose(output.joint_scores, recomposed, atol=1e-7, rtol=1e-7):
        raise AssertionError("joint scores disagree with the public composition helper")
    if not torch.allclose(
        output.joint_scores.sum(dim=-1),
        torch.ones_like(output.joint_scores[..., 0]),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError("joint scores do not sum to one")
    follow_scores = torch.softmax(output.follow_logits, dim=-1)
    if not torch.allclose(
        output.joint_scores[..., 1:].sum(dim=-1),
        follow_scores[..., 1],
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError("direction mass does not equal the follow probability")

    prediction = hierarchical_predictions(
        output.follow_logits,
        output.direction_logits,
    )
    if tuple(prediction.shape) != (batch_size, node_count, node_count):
        raise AssertionError("hierarchical prediction has the wrong shape")
    tied_follow = torch.zeros_like(output.follow_logits)
    tied_prediction = hierarchical_predictions(tied_follow, output.direction_logits)
    if torch.count_nonzero(tied_prediction).item() != 0:
        raise AssertionError("an exact follow-logit tie must map to no_follow")

    print("Synthetic CPU forward passed")
    for name in ("follow_logits", "direction_logits", "joint_scores"):
        print(f"{name}: {tuple(getattr(output, name).shape)}")
    print(f"prediction: {tuple(prediction.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
