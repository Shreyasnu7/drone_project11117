from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class SessionState:
    """
    Authoritative per-drone session state.
    """

    drone_id: str

    human_control_active: bool = True
    ai_control_active: bool = False

    active_intent_id: Optional[str] = None
    execution_phase: Optional[str] = None  # planning / executing / paused

    last_known_position: Optional[Dict[str, float]] = None
    last_health_status: Optional[Dict[str, float]] = None

    metadata: Dict[str, str] = field(default_factory=dict)