"""Public API for the first packaged gnn-for-futures release."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__version__ = "0.1.0"

__all__ = [
    "ShockFollowGraphMixerV2",
    "ShockFollowModelV2Output",
    "hierarchical_joint_scores",
    "hierarchical_predictions",
]

if TYPE_CHECKING:
    from gnn_for_futures.shock_follow.models.head_v2 import (
        hierarchical_joint_scores,
        hierarchical_predictions,
    )
    from gnn_for_futures.shock_follow.models.mixer_v2 import (
        ShockFollowGraphMixerV2,
        ShockFollowModelV2Output,
    )


def __getattr__(name: str) -> Any:
    """Load Torch-backed public symbols only when they are requested."""
    if name in {"hierarchical_joint_scores", "hierarchical_predictions"}:
        from gnn_for_futures.shock_follow.models import head_v2

        value = getattr(head_v2, name)
    elif name in {"ShockFollowGraphMixerV2", "ShockFollowModelV2Output"}:
        from gnn_for_futures.shock_follow.models import mixer_v2

        value = getattr(mixer_v2, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
