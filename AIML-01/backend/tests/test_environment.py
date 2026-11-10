"""Basic environment and dependency checks."""

import numpy as np
import pandas as pd


def test_pandas_import():
    assert pd.__version__

def test_numpy_import():
    assert np.__version__
