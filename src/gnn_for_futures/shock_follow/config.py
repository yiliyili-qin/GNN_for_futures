"""Shared configuration types and defaults for shock-follow GraphMixer V2."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_NODE_FEATURE_DIM = 9
DEFAULT_SEQUENCE_LENGTH = 40
DEFAULT_TEMPORAL_HIDDEN_DIM = 13
DEFAULT_EXPECTED_NUM_FUTURES = 44
DEFAULT_ROLE_DIM = 32
DEFAULT_NEIGHBOR_CAPACITY = 5
DEFAULT_HEAD_HIDDEN_DIM = 100
DEFAULT_FUSION_WEIGHTS = (0.2, 0.4, 0.4)
DEFAULT_TOPOLOGY_THRESHOLD = 0.1
DEFAULT_ATTENTION_ROBUST_QUANTILE = 0.75

LAMBDA_FOLLOW = 1.0
LAMBDA_DIRECTION = 1.0


@dataclass(frozen=True)
class ModelConfig:
    """Serializable constructor arguments for the V2 model."""

    node_feature_dim: int = DEFAULT_NODE_FEATURE_DIM
    temporal_hidden_dim: int = DEFAULT_TEMPORAL_HIDDEN_DIM
    attention_history_length: int = DEFAULT_SEQUENCE_LENGTH
    expected_num_futures: int = DEFAULT_EXPECTED_NUM_FUTURES
    role_dim: int = DEFAULT_ROLE_DIM
    neighbor_capacity: int = DEFAULT_NEIGHBOR_CAPACITY
    head_hidden_dim: int = DEFAULT_HEAD_HIDDEN_DIM
    fusion_weights: tuple[float, float, float] = DEFAULT_FUSION_WEIGHTS
    topology_threshold: float = DEFAULT_TOPOLOGY_THRESHOLD
    attention_robust_quantile: float = DEFAULT_ATTENTION_ROBUST_QUANTILE


@dataclass(frozen=True)
class TrainingConfig:
    """Serializable training parameters persisted in V2 checkpoints."""

    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    seed: int
    sequence_length: int
    train_end: str
    development_start: str
    development_end: str
    confirmation_start: str
    confirmation_end: str
    prohibited_test_start: str
    prohibited_test_end: str
    lambda_follow: float
    lambda_direction: float
