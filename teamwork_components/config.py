"""
Teamwork Configuration Management

Centralized configuration for modular teamwork components.
Each component can be independently enabled/disabled for ablation studies.

Usage:
    config = TeamworkConfig(
        smm=True,
        leadership=True,
        team_orientation=True,
        trust=True,
        mutual_monitoring=True,
        n_turns=2
    )

    if config.smm:
        # Use Shared Mental Model

    if config.is_enabled('leadership'):
        # Use Leadership component
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class TeamworkConfig:
    """
    Configuration for modular teamwork components.

    All components default to OFF for backwards compatibility.
    Enable selectively for ablation studies or full system operation.
    """

    # Component flags
    smm: bool = False                      # Shared Mental Model
    leadership: bool = False               # Leadership coordination
    team_orientation: bool = False         # Specialized roles + hierarchical weights
    trust: bool = False                    # Trust network scoring
    mutual_monitoring: bool = False        # Inter-round validation

    # Discussion parameters
    n_turns: int = 2                       # Number of R3 discussion turns (2 or 3)

    # Component-specific settings
    hierarchical_weights: list = field(default_factory=lambda: [0.5, 0.3, 0.2])
    trust_default: float = 0.8             # Default trust score when Trust disabled
    trust_range: tuple = (0.4, 1.0)        # Trust score bounds

    def __post_init__(self):
        """Validate configuration."""
        if self.n_turns not in [2, 3]:
            logging.warning(f"n_turns={self.n_turns} is unusual, expected 2 or 3")

        # Validate dependencies
        if self.team_orientation and not self.leadership:
            logging.warning("Team Orientation works best with Leadership enabled")

        if self.mutual_monitoring and not self.leadership:
            logging.warning("Mutual Monitoring requires Leadership to function properly")
            self.mutual_monitoring = False

        if self.mutual_monitoring and not self.trust:
            logging.info("Mutual Monitoring enabled without Trust - trust scores won't be updated")

    def is_enabled(self, component: str) -> bool:
        """Check if a component is enabled."""
        return getattr(self, component, False)

    def get_active_components(self) -> list:
        """Return list of enabled component names."""
        components = []
        if self.smm:
            components.append('SMM')
        if self.leadership:
            components.append('Leadership')
        if self.team_orientation:
            components.append('TeamOrientation')
        if self.trust:
            components.append('Trust')
        if self.mutual_monitoring:
            components.append('MutualMonitoring')
        return components

    def get_expected_api_calls(self, n_agents: int) -> int:
        """
        Calculate expected API calls based on enabled components.

        Reference from algorithm:
        - Base: 2N+3 (2 turns) or 3N+3 (3 turns)
        - +Leadership: +2 per turn
        - +MM: +3 per inter-turn

        Args:
            n_agents: Number of agents

        Returns:
            Expected total API calls
        """
        # R1: 2 calls (recruit + initialize)
        calls = 2

        # R2: N + 1 (N parallel predictions + 1 post-processing)
        calls += n_agents + 1

        # R3: Base calls per turn
        r3_calls_per_turn = n_agents  # Round-robin discourse

        if self.leadership:
            r3_calls_per_turn += 1  # Mediation

        calls += r3_calls_per_turn * self.n_turns

        # MM: Between turns only (n_turns - 1 times)
        if self.mutual_monitoring:
            mm_calls = 3  # concern + response + update
            calls += mm_calls * (self.n_turns - 1)

        return calls

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'smm': self.smm,
            'leadership': self.leadership,
            'team_orientation': self.team_orientation,
            'trust': self.trust,
            'mutual_monitoring': self.mutual_monitoring,
            'n_turns': self.n_turns,
            'hierarchical_weights': self.hierarchical_weights,
            'trust_default': self.trust_default,
            'trust_range': self.trust_range,
            'active_components': self.get_active_components()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TeamworkConfig':
        """Create from dictionary."""
        return cls(
            smm=data.get('smm', False),
            leadership=data.get('leadership', False),
            team_orientation=data.get('team_orientation', False),
            trust=data.get('trust', False),
            mutual_monitoring=data.get('mutual_monitoring', False),
            n_turns=data.get('n_turns', 2),
            hierarchical_weights=data.get('hierarchical_weights', [0.5, 0.3, 0.2]),
            trust_default=data.get('trust_default', 0.8),
            trust_range=tuple(data.get('trust_range', (0.4, 1.0)))
        )

    @classmethod
    def base_system(cls) -> 'TeamworkConfig':
        """Create base system config (all components OFF)."""
        return cls()

    @classmethod
    def all_enabled(cls, n_turns: int = 2) -> 'TeamworkConfig':
        """Create full system config (all components ON)."""
        return cls(
            smm=True,
            leadership=True,
            team_orientation=True,
            trust=True,
            mutual_monitoring=True,
            n_turns=n_turns
        )

    def __repr__(self) -> str:
        """String representation."""
        components = self.get_active_components()
        if not components:
            return "TeamworkConfig(Base System - All OFF)"
        return f"TeamworkConfig({', '.join(components)}, n_turns={self.n_turns})"


__all__ = ['TeamworkConfig']
