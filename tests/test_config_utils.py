import pytest
import os
from specfem_tomo_helper.utils.config_utils import validate_config, ConfigValidationError, auto_detect_utm_from_extent

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
    config['doubling_layers'] = [31, 80]  # in km (positive-down convention)
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

def test_utm_optional_without_extent():
    """Test that UTM zone and hemisphere are optional when no extent is provided AND GUI is enabled"""
    config = VALID_CONFIG.copy()
    # Remove UTM fields and extent
    del config['utm_zone']
    del config['utm_hemisphere']
    config['use_gui'] = True  # GUI must be enabled for missing UTM
    # Should pass validation without UTM fields when no extent provided AND GUI enabled
    assert validate_config(config) is True

def test_utm_required_without_extent_and_no_gui():
    """Test that UTM zone and hemisphere are required when no extent is provided and GUI is disabled"""
    config = VALID_CONFIG.copy()
    # Remove UTM fields and extent
    del config['utm_zone']
    del config['utm_hemisphere']
    config['use_gui'] = False  # GUI disabled
    # Should fail validation - UTM is required when no extent and no GUI
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_utm_optional_with_null_extent():
    """Test that UTM zone and hemisphere are optional when extent is null AND GUI is enabled"""
    config = VALID_CONFIG.copy()
    del config['utm_zone']
    del config['utm_hemisphere']
    config['extent'] = None
    config['use_gui'] = True  # GUI must be enabled for null UTM when extent is null
    # Should pass validation without UTM fields when extent is null AND GUI enabled
    assert validate_config(config) is True

def test_utm_validation_when_present():
    """Test that UTM validation still works when fields are present"""
    config = VALID_CONFIG.copy()
    config['utm_zone'] = 0  # Invalid zone
    with pytest.raises(ConfigValidationError):
        validate_config(config)
    
    config['utm_zone'] = 32
    config['utm_hemisphere'] = 'X'  # Invalid hemisphere
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_utm_null_values_without_extent_requires_gui():
    """Test that UTM zone and hemisphere can be null when no extent is provided AND GUI is enabled"""
    config = VALID_CONFIG.copy()
    config['utm_zone'] = None
    config['utm_hemisphere'] = None
    config['use_gui'] = True  # GUI must be enabled
    # Should pass validation with null UTM fields when no extent provided AND GUI enabled
    assert validate_config(config) is True

def test_utm_null_values_with_null_extent():
    """Test that UTM zone and hemisphere can be null when extent is null AND GUI is enabled"""
    config = VALID_CONFIG.copy()
    config['utm_zone'] = None
    config['utm_hemisphere'] = None
    config['extent'] = None
    config['use_gui'] = True  # GUI must be enabled
    # Should pass validation with null UTM fields when extent is null AND GUI enabled
    assert validate_config(config) is True

def test_utm_null_values_with_extent_should_fail():
    """Test that null UTM values are NOT OK when extent is provided in geographic coordinates (should fail)"""
    config = VALID_CONFIG.copy()
    config['utm_zone'] = None
    config['utm_hemisphere'] = None
    config['extent'] = [2.0, 8.0, 50.0, 54.0]  # Geographic coordinates
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_utm_auto_detect_from_extent_should_fail():
    """Test that UTM cannot be auto-detected when extent is provided in geographic coordinates (should fail)"""
    config = VALID_CONFIG.copy()
    config['utm_zone'] = None
    config['utm_hemisphere'] = None
    config['extent'] = [-150.0, -140.0, 60.0, 65.0]  # Alaska region
    with pytest.raises(ConfigValidationError):
        validate_config(config)
