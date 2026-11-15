"""Source-event-balanced training loop used by the seven-year ablation."""
from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from gnn_for_futures.shock_follow.models.mixer_v2 import ShockFollowGraphMixerV2
from gnn_for_futures.shock_follow.models.head_v2 import hierarchical_source_event_loss
from gnn_for_futures.shock_follow.evaluation.metrics_v2 import V2MetricsAccumulator

LAMBDA_FOLLOW = 1.0


LAMBDA_DIRECTION = 1.0


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: (
            value.to(device, non_blocking=device.type == "cuda")
            if isinstance(value, Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _forward(model: ShockFollowGraphMixerV2, batch: dict[str, object]):
    return model(
        batch["node_history"],
        batch["economic_prior"],
        batch["regression_weight"],
        prior_available=batch["prior_available"],
        regression_available=batch["regression_available"],
    )


def _event_sums(
    cell_losses: Tensor,
    mask: Tensor,
) -> tuple[float, int]:
    counts = mask.sum(dim=-1)
    events = counts > 0
    event_count = int(events.sum().item())
    if not event_count:
        return 0.0, 0
    per_cell = torch.zeros_like(mask, dtype=cell_losses.dtype)
    per_cell[mask] = cell_losses
    per_source = per_cell.sum(dim=-1) / counts.clamp_min(1)
    return float(per_source[events].sum().item()), event_count


def run_epoch(
    model: ShockFollowGraphMixerV2,
    loader: DataLoader,
    device: torch.device,
    optimizer: Adam | None,
    follow_class_weights: Tensor,
) -> dict[str, object]:
    """Run one V2 epoch and aggregate losses by their defining events/cells."""
    training = optimizer is not None
    model.train(training)
    weights = follow_class_weights.to(device=device)
    weighted_follow_event_sum = 0.0
    unweighted_follow_event_sum = 0.0
    direction_event_sum = 0.0
    follow_events = 0
    direction_events = 0
    weighted_follow_cell_sum = 0.0
    unweighted_follow_cell_sum = 0.0
    direction_cell_sum = 0.0
    valid_pairs = 0
    direction_pairs = 0
    observed_batches = 0
    nonempty_batches = 0
    follow_empty_batches = 0
    direction_empty_batches = 0
    optimizer_steps = 0
    sample_count = 0
    edges = 0
    possible_edges = 0
    isolated_nodes = 0
    padded_out_slots = 0
    out_slots = 0
    saturation_sum = 0.0
    probability_sum_error = 0.0
    joint_follow_error = 0.0
    task_metrics = V2MetricsAccumulator(
        follow_class_weights,
        lambda_follow=LAMBDA_FOLLOW,
        lambda_direction=LAMBDA_DIRECTION,
    )

    for raw_batch in loader:
        observed_batches += 1
        batch = _move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = _forward(model, batch)
            raw_mask = batch["loss_mask"]
            if raw_mask.dtype != torch.bool:
                raise TypeError("loss_mask must be boolean")
            diagonal = torch.eye(
                raw_mask.shape[1], dtype=torch.bool, device=raw_mask.device
            ).unsqueeze(0)
            mask = raw_mask & ~diagonal
            outcome = hierarchical_source_event_loss(
                output.follow_logits,
                output.direction_logits,
                batch["target_class"],
                mask,
                weights,
                lambda_follow=LAMBDA_FOLLOW,
                lambda_direction=LAMBDA_DIRECTION,
            )
            task_metrics.update(
                output.follow_logits,
                output.direction_logits,
                batch["target_class"],
                mask,
                joint_scores=output.joint_scores,
            )
            adjacency = output.fused_graph.adjacency
            current_batch_size, node_count, _ = adjacency.shape
            sample_count += current_batch_size
            edges += int(adjacency.sum().item())
            possible_edges += current_batch_size * node_count * (node_count - 1)
            isolated_nodes += int((~adjacency.any(dim=-1)).sum().item())
            out_slots += output.directed_views.out_valid.numel()
            padded_out_slots += int((~output.directed_views.out_valid).sum().item())
            saturation_sum += float(output.attention.saturation_rate.sum().item())
            if not torch.isfinite(output.follow_logits).all() or not torch.isfinite(
                output.direction_logits
            ).all() or not torch.isfinite(output.joint_scores).all():
                raise FloatingPointError("encountered non-finite V2 model output")
            probability_sum_error = max(
                probability_sum_error,
                float((output.joint_scores.sum(dim=-1) - 1.0).abs().max().item()),
            )
            p_follow = torch.softmax(output.follow_logits, dim=-1)[..., 1]
            joint_follow_error = max(
                joint_follow_error,
                float((output.joint_scores[..., 1:].sum(dim=-1) - p_follow).abs().max().item()),
            )
            if outcome.is_empty:
                follow_empty_batches += 1
                direction_empty_batches += 1
                continue
            nonempty_batches += 1
            if outcome.direction_is_empty:
                direction_empty_batches += 1
            if outcome.total_loss is None or not torch.isfinite(outcome.total_loss):
                raise FloatingPointError("encountered non-finite V2 training loss")
            if training:
                outcome.total_loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("encountered non-finite V2 gradient")
                optimizer.step()
                optimizer_steps += 1

        targets = batch["target_class"]
        valid_targets = targets[mask]
        follow_targets = valid_targets.ne(0).long()
        weighted_cells = torch.nn.functional.cross_entropy(
            output.follow_logits[mask],
            follow_targets,
            weight=weights.to(dtype=output.follow_logits.dtype),
            reduction="none",
        )
        unweighted_cells = torch.nn.functional.cross_entropy(
            output.follow_logits[mask],
            follow_targets,
            reduction="none",
        )
        weighted_event_sum, event_count = _event_sums(weighted_cells, mask)
        unweighted_event_sum, unweighted_event_count = _event_sums(
            unweighted_cells, mask
        )
        if event_count != outcome.follow_event_count or event_count != unweighted_event_count:
            raise AssertionError("follow event aggregation count mismatch")
        weighted_follow_event_sum += weighted_event_sum
        unweighted_follow_event_sum += unweighted_event_sum
        follow_events += event_count
        weighted_follow_cell_sum += float(weighted_cells.sum().item())
        unweighted_follow_cell_sum += float(unweighted_cells.sum().item())
        valid_pairs += int(mask.sum().item())

        direction_mask = mask & targets.ne(0)
        if direction_mask.any():
            direction_targets = targets[direction_mask] - 1
            direction_cells = torch.nn.functional.cross_entropy(
                output.direction_logits[direction_mask],
                direction_targets.long(),
                reduction="none",
            )
            event_sum, event_count = _event_sums(direction_cells, direction_mask)
            if event_count != outcome.direction_event_count:
                raise AssertionError("direction event aggregation count mismatch")
            direction_event_sum += event_sum
            direction_events += event_count
            direction_cell_sum += float(direction_cells.sum().item())
            direction_pairs += int(direction_mask.sum().item())

    if follow_events:
        weighted_follow_loss = weighted_follow_event_sum / follow_events
        unweighted_follow_loss = unweighted_follow_event_sum / follow_events
        direction_loss = (
            direction_event_sum / direction_events if direction_events else None
        )
        total_loss = weighted_follow_loss + (direction_loss or 0.0)
    else:
        weighted_follow_loss = None
        unweighted_follow_loss = None
        direction_loss = None
        total_loss = None
    finite_values = [
        value
        for value in (
            weighted_follow_loss,
            unweighted_follow_loss,
            direction_loss,
            total_loss,
        )
        if value is not None
    ]
    if not all(np.isfinite(value) for value in finite_values):
        raise FloatingPointError("epoch produced non-finite aggregate V2 losses")
    classification_metrics = task_metrics.finalize()
    accumulator_loss = classification_metrics["total_loss"]
    if total_loss is None:
        if accumulator_loss is not None:
            raise AssertionError("empty epoch loss disagrees with V2 metrics accumulator")
    elif not isinstance(accumulator_loss, float) or not np.isclose(
        total_loss,
        accumulator_loss,
        rtol=1e-7,
        atol=1e-8,
    ):
        raise AssertionError("epoch loss disagrees with V2 metrics accumulator")
    return {
        "total_loss": total_loss,
        "classification": classification_metrics,
        "weighted_follow_source_event_loss": weighted_follow_loss,
        "unweighted_follow_source_event_loss": unweighted_follow_loss,
        "weighted_follow_cell_loss": (
            weighted_follow_cell_sum / valid_pairs if valid_pairs else None
        ),
        "unweighted_follow_cell_loss": (
            unweighted_follow_cell_sum / valid_pairs if valid_pairs else None
        ),
        "direction_source_event_loss": direction_loss,
        "direction_cell_loss": (
            direction_cell_sum / direction_pairs if direction_pairs else None
        ),
        "follow_source_event_count": follow_events,
        "direction_source_event_count": direction_events,
        "valid_pair_count": valid_pairs,
        "direction_pair_count": direction_pairs,
        "observed_batch_count": observed_batches,
        "nonempty_batch_count": nonempty_batches,
        "follow_empty_batch_count": follow_empty_batches,
        "direction_empty_batch_count": direction_empty_batches,
        "optimizer_step_count": optimizer_steps,
        "mean_edges_per_sample": edges / sample_count if sample_count else 0.0,
        "edge_density": edges / possible_edges if possible_edges else 0.0,
        "mean_out_isolated_nodes_per_sample": (
            isolated_nodes / sample_count if sample_count else 0.0
        ),
        "out_slot_padding_fraction": (
            padded_out_slots / out_slots if out_slots else 0.0
        ),
        "mean_attention_saturation_rate": (
            saturation_sum / sample_count if sample_count else 0.0
        ),
        "joint_score_sum_max_abs_error": probability_sum_error,
        "joint_follow_identity_max_abs_error": joint_follow_error,
    }
