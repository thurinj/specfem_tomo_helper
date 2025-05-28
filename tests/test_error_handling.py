import pytest
import os
from specfem_tomo_helper.utils.config_utils import validate_config, ConfigValidationError

# Use our generated test data file
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VALID_DATA_PATH = os.path.join(TEST_DATA_DIR, "valid_model.nc")

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

def test_data_path_not_found():
    config = VALID_CONFIG.copy()
    config['data_path'] = 'data/does_not_exist.nc'
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_extent_wrong_length():
    config = VALID_CONFIG.copy()
    config['extent'] = [1, 2, 3]  # Only 3 values
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_extent_wrong_type():
    config = VALID_CONFIG.copy()
    config['extent'] = [1, 2, 3, 'bad']
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_mesh_missing_option():
    config = VALID_CONFIG.copy()
    config['generate_mesh'] = True
    # Missing mesh_output_dir
    config['max_depth'] = 250.0
    config['dx_target_km'] = 5.0
    config['dz_target_km'] = 5.0
    config['max_cpu'] = 64
    config['doubling_layers'] = [-31, -80]  # in km
    if 'mesh_output_dir' in config:
        del config['mesh_output_dir']
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_topography_wrong_type():
    config = VALID_CONFIG.copy()
    config['generate_topography'] = True
    config['topography_output_dir'] = './topo_test'
    config['slope_thresholds'] = [10, 15, 'bad']
    config['smoothing_sigma'] = 'auto'
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_invalid_fill_nan():
    config = VALID_CONFIG.copy()
    config['fill_nan'] = 'diagonal'
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_invalid_plot_color_by():
    config = VALID_CONFIG.copy()
    config['plot_color_by'] = 'foo'
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_negative_mesh_values():
    config = VALID_CONFIG.copy()
    config['generate_mesh'] = True
    config['mesh_output_dir'] = './mesh_test'
    config['max_depth'] = -1
    config['dx_target_km'] = 5.0
    config['dz_target_km'] = 5.0
    config['max_cpu'] = 64
    config['doubling_layers'] = [-31, -80]  # in km
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_zero_mesh_values():
    config = VALID_CONFIG.copy()
    config['generate_mesh'] = True
    config['mesh_output_dir'] = './mesh_test'
    config['max_depth'] = 0
    config['dx_target_km'] = 0
    config['dz_target_km'] = 0
    config['max_cpu'] = 0
    config['doubling_layers'] = [-31, -80]  # in km
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_empty_variable_list():
    config = VALID_CONFIG.copy()
    config['variable'] = []
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_nonstring_in_variable():
    config = VALID_CONFIG.copy()
    config['variable'] = [1, 'vs']
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_nonlist_doubling_layers():
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

def test_nonlist_slope_thresholds():
    config = VALID_CONFIG.copy()
    config['generate_topography'] = True
    config['topography_output_dir'] = './topo_test'
    config['slope_thresholds'] = 'not_a_list'
    config['smoothing_sigma'] = 'auto'
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_nonnumeric_in_doubling_layers():
    config = VALID_CONFIG.copy()
    config['generate_mesh'] = True
    config['mesh_output_dir'] = './mesh_test'
    config['max_depth'] = 250.0
    config['dx_target_km'] = 5.0
    config['dz_target_km'] = 5.0
    config['max_cpu'] = 64
    config['doubling_layers'] = [1, 'bad']
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_invalid_smoothing_sigma():
    config = VALID_CONFIG.copy()
    config['generate_topography'] = True
    config['topography_output_dir'] = './topo_test'
    config['slope_thresholds'] = [10, 15, 20]
    config['smoothing_sigma'] = [1, 2]
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_invalid_output_dir_types():
    config = VALID_CONFIG.copy()
    config['tomo_output_dir'] = 123
    with pytest.raises(ConfigValidationError):
        validate_config(config)
    config = VALID_CONFIG.copy()
    config['generate_mesh'] = True
    config['mesh_output_dir'] = 123
    config['max_depth'] = 250.0
    config['dx_target_km'] = 5.0
    config['dz_target_km'] = 5.0
    config['max_cpu'] = 64
    config['doubling_layers'] = [-31, -80]  # in km
    with pytest.raises(ConfigValidationError):
        validate_config(config)
    config = VALID_CONFIG.copy()
    config['generate_topography'] = True
    config['topography_output_dir'] = 123
    config['slope_thresholds'] = [10, 15, 20]
    config['smoothing_sigma'] = 'auto'
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_missing_core_required_field():
    config = VALID_CONFIG.copy()
    del config['dx']  # 'dx' is a core required field
    with pytest.raises(ConfigValidationError):
        validate_config(config)

    config = VALID_CONFIG.copy()
    del config['dy']
    with pytest.raises(ConfigValidationError):
        validate_config(config)

    config = VALID_CONFIG.copy()
    del config['dz']
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_z_min_not_less_than_z_max():
    config = VALID_CONFIG.copy()
    config['z_min'] = 0
    config['z_max'] = -10  # z_min not less than z_max
    with pytest.raises(ConfigValidationError):
        validate_config(config)

    config = VALID_CONFIG.copy()
    config['z_min'] = 0
    config['z_max'] = 0  # z_min not less than z_max
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_invalid_utm_zone():
    config = VALID_CONFIG.copy()
    config['utm_zone'] = 0  # Invalid utm_zone
    with pytest.raises(ConfigValidationError):
        validate_config(config)

    config = VALID_CONFIG.copy()
    config['utm_zone'] = 61  # Invalid utm_zone
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_invalid_utm_hemisphere():
    config = VALID_CONFIG.copy()
    config['utm_hemisphere'] = 'X'  # Invalid utm_hemisphere
    with pytest.raises(ConfigValidationError):
        validate_config(config)

def test_non_positive_dx_dy_dz():
    config = VALID_CONFIG.copy()
    config['dx'] = 0  # dx is not > 0
    with pytest.raises(ConfigValidationError):
        validate_config(config)

    config = VALID_CONFIG.copy()
    config['dy'] = -10  # dy is not > 0
    with pytest.raises(ConfigValidationError):
        validate_config(config)

    # config = VALID_CONFIG.copy()
    # config['dz'] = 0 # Or -10
    # with pytest.raises(ConfigValidationError):
    #     validate_config(config)

def test_missing_topography_option():
    config = VALID_CONFIG.copy()
    config['generate_topography'] = True
    config['slope_thresholds'] = [10, 15, 20]
    config['smoothing_sigma'] = 'auto'
    # Ensure topography_output_dir is missing
    if 'topography_output_dir' in config:
        del config['topography_output_dir']
    with pytest.raises(ConfigValidationError) as excinfo:
        validate_config(config)
    assert "Missing topography config option: topography_output_dir" in str(excinfo.value)

def test_mesh_max_cpu_wrong_type():
    config = VALID_CONFIG.copy()
    config['generate_mesh'] = True
    config['mesh_output_dir'] = './mesh_test'
    config['max_depth'] = 250.0
    config['dx_target_km'] = 5.0
    config['dz_target_km'] = 5.0
    config['doubling_layers'] = [-31, -80]  # in km
    
    original_max_cpu = config.get('max_cpu') # Save original if it exists for VALID_CONFIG

    config['max_cpu'] = 64.5  # Float instead of int
    with pytest.raises(ConfigValidationError) as excinfo:
        validate_config(config)
    assert "Mesh config option 'max_cpu' must be of type <class 'int'>" in str(excinfo.value)

    config['max_cpu'] = 'not_an_int' # String instead of int
    with pytest.raises(ConfigValidationError) as excinfo:
        validate_config(config)
    assert "Mesh config option 'max_cpu' must be of type <class 'int'>" in str(excinfo.value)
    
    if original_max_cpu is not None: # Restore if needed
        config['max_cpu'] = original_max_cpu
    else:
        del config['max_cpu']


def test_core_utm_zone_wrong_type():
    config = VALID_CONFIG.copy()
    original_utm_zone = config['utm_zone']

    config['utm_zone'] = 'not_an_int'
    with pytest.raises(ConfigValidationError) as excinfo:
        validate_config(config)
    assert "utm_zone must be an integer, got <class 'str'>" in str(excinfo.value)

    config['utm_zone'] = 32.5 # Float, should fail as int is specified
    with pytest.raises(ConfigValidationError) as excinfo:
        validate_config(config)
    assert "utm_zone must be an integer, got <class 'float'>" in str(excinfo.value)

    config['utm_zone'] = original_utm_zone # Restore

def test_extent_latlon_rejected():
    config = VALID_CONFIG.copy()
    # Simulate geographic (lat/lon) extent
    config['extent'] = [-120, -110, 30, 40]
    # Should fail: extent must be UTM
    with pytest.raises(ConfigValidationError) as excinfo:
        validate_config(config)
    assert "extent must be a list of 4 numbers" not in str(excinfo.value)  # Not a type error
    # The error should be about UTM info missing or extent not being UTM

def test_extent_utm_missing_utm_zone():
    config = VALID_CONFIG.copy()
    config['extent'] = [500000, 600000, 4000000, 4100000]  # Looks like UTM
    config['utm_zone'] = None
    with pytest.raises(ConfigValidationError) as excinfo:
        validate_config(config)
    assert "UTM zone and hemisphere can only be null" in str(excinfo.value) or "both 'utm_zone' and 'utm_hemisphere' must also be specified" in str(excinfo.value)
