"""Independent teaching-network fixtures; expected answers live in tests, not the app."""

import numpy as np
import pytest

from loadflow_app.models import BranchData, BusData, NetworkCase, SolverConfig


@pytest.fixture
def three_bus():
    return NetworkCase(
        buses=[
            BusData(id="1", type="Slack", v_set=1.0, angle_set=0.0),
            BusData(id="2", type="PQ", p=-2.0, q=-0.5, v_guess=1.0, angle_guess=0.0),
            BusData(id="3", type="PV", p=1.0, v_set=1.0, angle_guess=0.0,
                    q_min=-5.0, q_max=5.0, q_load=0.0),
        ],
        branches=[
            BranchData(id="L12", from_bus="1", to_bus="2", r=0.0, x=0.1),
            BranchData(id="L13", from_bus="1", to_bus="3", r=0.0, x=0.4),
            BranchData(id="L23", from_bus="2", to_bus="3", r=0.0, x=0.2),
        ],
        config=SolverConfig(tolerance=1e-10, max_iterations=500),
    )


@pytest.fixture
def expected_ybus():
    return 1j * np.array([[-12.5, 10, 2.5], [10, -15, 5], [2.5, 5, -7.5]])
