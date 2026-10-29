"""Stage 3 Post-Training: SFT Warm-Start and GRPO/DAPO Verifiable RL."""

from .grpo_dapo_trainer import (
    GRPODAPOMetrics,
    StateJumpGRPODAPOTrainer,
    compute_dapo_loss_from_logprobs,
    compute_group_advantages,
)
from .sft_trainer import SocraticSFTTrainer

__all__ = [
    "SocraticSFTTrainer",
    "StateJumpGRPODAPOTrainer",
    "GRPODAPOMetrics",
    "compute_group_advantages",
    "compute_dapo_loss_from_logprobs",
]
