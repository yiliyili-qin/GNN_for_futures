"""Frozen Full + standardized 119-D linear direction baseline, without History."""

import json
from pathlib import Path

import numpy as np
import torch

from tools import alpha39_direction_concat as heads
from tools.ten_year_train import sha, write_json

MODEL = "Full+Dir119Linear-v1"
SEED = 20261002
FEATURE_NAMES = (
    "source_sign", "source_scaled_return", "target_scaled_return", "source_sign_x_target_return",
    *[f"source_momentum_{n}" for n in (2, 4, 8)],
    *[f"target_momentum_{n}" for n in (2, 4, 8)],
    *[f"source_sign_x_target_momentum_{n}" for n in (2, 4, 8)],
    *[f"source_available_{n}" for n in (2, 4, 8)],
    *[f"target_available_{n}" for n in (2, 4, 8)],
)
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reference(path):
    path = Path(path).resolve()
    return dict(path=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
                sha256=sha(path))


def checked_reference(record):
    path = ROOT / record["path"]
    if sha(path) != record["sha256"]:
        raise ValueError("Frozen baseline hash mismatch: " + str(path))
    return path


def feature_contract():
    return dict(tail="input_to_Full.prediction_head.follow_output", tail_dimensions=100,
        raw_dimensions=19, raw_order=list(FEATURE_NAMES), total_dimensions=119,
        raw_formula="tools.run_direction_bypass.features", windows=[2, 4, 8],
        momentum="compound_same_trading_date_contiguous_15m_through_cutoff",
        missing="zero_with_window_availability", scaling="train_true_OS_only_population_std_floor1e-8_to1",
        output="sigmoid(Linear(119,1));120_parameters", q="native_Full_softmax_follow_class1",
        d="P(same_given_follow)", decision="N_if_q<=0.5_else_S_if_d>0.5_else_O",
        routing="q>0.5_offdiagonal_only;otherwise_d=0.5", history_backbone_required=False)


def full_forward(model, inputs):
    captured = []
    handle = model.prediction_head.follow_output.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach()))
    try:
        with torch.inference_mode():
            output = model(**inputs)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape[-1] != 100:
        raise ValueError("Expected exactly one 100-D Full pair representation")
    q = output.follow_logits.softmax(-1)[..., 1].cpu().numpy()
    d = output.direction_logits.softmax(-1)[..., 1].cpu().numpy()
    tail = captured[0].cpu().numpy()
    if not all(np.isfinite(value).all() for value in (q, d, tail)):
        raise ValueError("Nonfinite frozen Full output")
    return q, tail, d


def routed_direction(fitted, q, tail, raw, enabled):
    if tail.shape != q.shape + (100,) or raw.shape != q.shape + (19,) or enabled.shape != q.shape:
        raise ValueError("Misaligned baseline pair features")
    gate = (q > .5) & ~np.eye(q.shape[-1], dtype=bool)
    if not enabled[gate].all():
        raise ValueError("Direction features unavailable for an alerted pair")
    result = np.full(q.shape, .5, dtype=np.float32)
    result[gate] = heads.predict(fitted, np.concatenate((tail[gate], raw[gate]), -1))
    return result


def freeze(path, full_path, head_directory, nodes, domain, protocol_path, *, origin=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    info = read(Path(head_directory) / "training_completed.json")
    if (info["kind"], info["seed"], info["features"], info["parameters"]) != ("linear", SEED, 119, 120):
        raise ValueError("Unexpected baseline direction checkpoint")
    head = reference(Path(head_directory) / "best.pt")
    if head["sha256"] != info["checkpoint_sha256"]:
        raise ValueError("Direction training/checkpoint hash mismatch")
    record = dict(schema="full_dir119_linear_frozen_baseline", version="1.0.0", model=MODEL,
        domain=domain, node_order=list(nodes), backbone_seed=SEED, head_seed=SEED,
        seed_role="single_primary_designated_2026-11-03_not_best_test_seed", seed_replicates=1,
        components=dict(full=reference(full_path), direction=head,
                        scaler=reference(Path(head_directory) / "scaler.npz")),
        feature_contract=feature_contract(), protocol=reference(protocol_path), origin=origin,
        implementation={p.relative_to(ROOT).as_posix(): sha(p) for p in
            (Path(__file__).resolve(), ROOT / "tools/alpha39_direction_concat.py", ROOT / "tools/run_direction_bypass.py",
             *sorted((ROOT / "src/gnn_for_futures/shock_follow/models").glob("*.py")),
             ROOT / "src/gnn_for_futures/shock_follow/graph.py",
             *sorted((ROOT / "src/gnn_for_futures/graph").glob("*.py")))},
        selection=dict(direction=info["selection"], direction_epoch=info["best_epoch"]),
        strict_pit=False, fresh_oos=False, previously_inspected_test=True,
        scope="frozen_inference_artifact_not_full_parent_suite_admission")
    write_json(path, record)
    return record


def load(manifest_path):
    from gnn_for_futures.shock_follow.models.mixer_v2 import ShockFollowGraphMixerV2
    manifest = read(manifest_path)
    if (manifest.get("schema") != "full_dir119_linear_frozen_baseline"
        or manifest.get("model") != MODEL or manifest["feature_contract"] != feature_contract()):
        raise ValueError("Unsupported frozen baseline contract")
    for name, digest in manifest["implementation"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen baseline implementation changed: " + name)
    checked_reference(manifest["protocol"])
    paths = {name: checked_reference(record) for name, record in manifest["components"].items()}
    full = ShockFollowGraphMixerV2(9, expected_num_futures=len(manifest["node_order"]), learnable_fusion=False)
    full.load_state_dict(torch.load(paths["full"], map_location="cpu", weights_only=True))
    full.eval().requires_grad_(False)
    linear = heads.network(119, "linear")
    linear.load_state_dict(torch.load(paths["direction"], map_location="cpu", weights_only=True))
    linear.eval().requires_grad_(False)
    with np.load(paths["scaler"], allow_pickle=False) as archive:
        mean, scale = archive["mean"].copy(), archive["scale"].copy()
    if (mean.shape != (119,) or scale.shape != (119,) or not np.isfinite(mean).all()
        or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("Invalid frozen baseline scaler")
    return full, dict(model=linear, mean=mean, scale=scale, kind="linear"), manifest


def predict(manifest_path, inputs, panel, thresholds, cutoffs):
    full, fitted, manifest = load(manifest_path)
    if list(panel.node_order) != manifest["node_order"]:
        raise ValueError("Frozen baseline node order mismatch")
    q, tail, _ = full_forward(full, inputs)
    raw, enabled = heads.features(panel.signed_returns, panel.return_valid, panel.timestamps,
                                  panel.trading_dates, thresholds, np.asarray(cutoffs))
    # Non-shock sources are not actionable, even if the dense Full output exceeds .5.
    actionable = enabled & (np.abs(panel.signed_returns[cutoffs]) >= thresholds[cutoffs])[:, :, None]
    gated_q = np.where(actionable, q, 0.)
    d = routed_direction(fitted, gated_q, tail, raw, enabled)
    return dict(q=q, d=d, predicted_class=np.where(q > .5, np.where(d > .5, 2, 1), 0),
                actionable=actionable, joint=np.stack((1-q, q*(1-d), q*d), -1))
