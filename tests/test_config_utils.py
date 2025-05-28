import pytest
import os
from specfem_tomo_helper.utils.config_utils import validate_config, ConfigValidationError

# Use our generated test data file
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VALID_DATA_PATH = os.path.join(TEST_DATA_DIR, "valid_model.nc")

# Minimal valid config for testing
VALID_CONFIG = {
    'data_path': VALID_DATA_PATH,
    'dx': 5000,
    'dy': 5000,
    'dz': 5000,
    'z_min': -250,
    'z_max': 0,
    'variable': 'vsv',
    'utm_zone': 32,
    'utm_hemisphere': 'N',
}

def test_valid_config():
    # Test with our generated valid NetCDF file
    config = VALID_CONFIG.copy()
    assert validate_config(config) is True

def test_missing_required():
    config = VALID_CONFIG.copy()
    del config['dx']
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_wrong_type():
    config = VALID_CONFIG.copy()
    config['dx'] = 'not_a_number'
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_invalid_range():
    config = VALID_CONFIG.copy()
    config['dx'] = -1
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_invalid_utm_zone():
    config = VALID_CONFIG.copy()
    config['utm_zone'] = 0
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_mesh_options():
    config = VALID_CONFIG.copy()
    config['generate_mesh'] = True
    config['mesh_output_dir'] = './mesh_test'
    config['max_depth'] = 250.0
    config['dx_target_km'] = 5.0
    config['dz_target_km'] = 5.0
    config['max_cpu'] = 64
    config['doubling_layers'] = [-31000, -80000]
    assert validate_config(config) is True

def test_topography_options():
    config = VALID_CONFIG.copy()
    config['generate_topography'] = True
    config['topography_output_dir'] = './topo_test'
    config['slope_thresholds'] = [10, 15, 20]
    config['smoothing_sigma'] = 'auto'
    assert validate_config(config) is True

def test_invalid_mesh_option_type():
    config = VALID_CONFIG.copy()
    config['generate_mesh'] = True
    config['mesh_output_dir'] = './mesh_test'
    config['max_depth'] = 250.0
    config['dx_target_km'] = 5.0
    config['dz_target_km'] = 5.0
    config['max_cpu'] = 64
    config['doubling_layers'] = 'not_a_list'
    with pytest.raises(ConfigValidationError):
        validate_config(config)
