"""Keep the original NumPy-random regression tests reproducible and isolated."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def seeded_legacy_random_states():
    """Restore global random state after each historical random-input test."""
    previous = np.random.get_state()
    np.random.seed(314159)
    yield
    np.random.set_state(previous)
