"""Meta world proposals are validated; cells learn without model control."""
from .world_designer import design_world, evaluate_world, validate_proposal, DesignError

__all__ = ["design_world", "evaluate_world", "validate_proposal", "DesignError"]
