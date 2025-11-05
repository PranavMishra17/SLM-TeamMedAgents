"""
Teamwork Components Module

Modular teamwork mechanisms for multi-agent medical reasoning systems.

Components:
1. Shared Mental Model (SMM) - Passive knowledge repository
2. Leadership - Active orchestration and coordination
3. Team Orientation - Specialized roles with hierarchical weights
4. Trust Network - Dynamic trust scoring and weighted voting
5. Mutual Monitoring - Inter-round validation and challenges

Each component is independently toggleable via TeamworkConfig for ablation studies.

Usage:
    from teamwork_components import (
        TeamworkConfig,
        SharedMentalModel,
        LeadershipCoordinator,
        TeamOrientationManager,
        TrustNetwork,
        MutualMonitoringCoordinator
    )

    # Configure components
    config = TeamworkConfig(
        smm=True,
        leadership=True,
        team_orientation=True,
        trust=True,
        mutual_monitoring=True,
        n_turns=2
    )

    # Initialize components
    smm = SharedMentalModel() if config.smm else None
    trust = TrustNetwork(config) if config.trust else None
    # ... etc

For full algorithm and integration details, see documentation.
"""

# Configuration
from .config import TeamworkConfig

# Core components
from .shared_mental_model import (
    SharedMentalModel,
    extract_facts_intersection,
    detect_question_tricks
)

from .leadership import LeadershipCoordinator

from .team_orientation import (
    TeamOrientationManager,
    AgentRole,
    enhance_prompt_with_role
)

from .trust_network import (
    TrustNetwork,
    AgentTrustProfile
)

from .mutual_monitoring import (
    MutualMonitoringCoordinator,
    MutualMonitoringResult
)


__version__ = '1.0.0'

__all__ = [
    # Configuration
    'TeamworkConfig',

    # Shared Mental Model
    'SharedMentalModel',
    'extract_facts_intersection',
    'detect_question_tricks',

    # Leadership
    'LeadershipCoordinator',

    # Team Orientation
    'TeamOrientationManager',
    'AgentRole',
    'enhance_prompt_with_role',

    # Trust Network
    'TrustNetwork',
    'AgentTrustProfile',

    # Mutual Monitoring
    'MutualMonitoringCoordinator',
    'MutualMonitoringResult',
]
