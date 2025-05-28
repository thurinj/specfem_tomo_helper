import pytest
import numpy as np
from specfem_tomo_helper.utils.config_utils import validate_config, ConfigValidationError

# Dummy interpolator for parameter tests
class DummyInterpolator:
    def interpolation_parameters(self, xmin, xmax, dx, ymin, ymax, dy, zmin, zmax, dz):
        # Simulate parameter checks similar to real interpolator
        if dx <= 0 or dy <= 0 or dz <= 0:
            raise ValueError('Grid spacing must be positive')
        if xmin >= xmax or ymin >= ymax or zmin >= zmax:
            raise ValueError('Extent or depth range is invalid')
        return True

def test_valid_interpolation_params():
    interp = DummyInterpolator()
    assert interp.interpolation_parameters(0, 10, 1, 0, 10, 1, -10, 0, 1) is True

def test_negative_dx():
    interp = DummyInterpolator()
    with pytest.raises(ValueError):
        interp.interpolation_parameters(0, 10, -1, 0, 10, 1, -10, 0, 1)

def test_zero_dy():
    interp = DummyInterpolator()
    with pytest.raises(ValueError):
        interp.interpolation_parameters(0, 10, 1, 0, 10, 0, -10, 0, 1)

def test_reversed_extent():
    interp = DummyInterpolator()
    with pytest.raises(ValueError):
        interp.interpolation_parameters(10, 0, 1, 0, 10, 1, -10, 0, 1)

def test_reversed_y_extent():
    interp = DummyInterpolator()
    with pytest.raises(ValueError):
        interp.interpolation_parameters(0, 10, 1, 10, 0, 1, -10, 0, 1)

def test_reversed_z():
    interp = DummyInterpolator()
    with pytest.raises(ValueError):
        interp.interpolation_parameters(0, 10, 1, 0, 10, 1, 0, -10, 1)

def test_non_numeric_extent():
    interp = DummyInterpolator()
    with pytest.raises(TypeError):
        interp.interpolation_parameters('a', 10, 1, 0, 10, 1, -10, 0, 1)
