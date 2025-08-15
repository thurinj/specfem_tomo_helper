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

class TestMissingValidationCoverage:
    """Test cases for configuration validation gaps identified in the analysis."""
    
    def test_extent_ordering_validation(self):
        """Test that extent coordinates are properly ordered (min < max)."""
        config = VALID_CONFIG.copy()
        
        # min_x >= max_x should fail
        config['extent'] = [100, 50, 200, 300]  # min_x > max_x
        with pytest.raises(ConfigValidationError, match="extent min_x must be less than max_x"):
            validate_config(config)
        
        # min_y >= max_y should fail
        config['extent'] = [50, 100, 300, 200]  # min_y > max_y
        with pytest.raises(ConfigValidationError, match="extent min_y must be less than max_y"):
            validate_config(config)
        
        # Equal values should also fail
        config['extent'] = [100, 100, 200, 300]  # min_x = max_x
        with pytest.raises(ConfigValidationError, match="extent min_x must be less than max_x"):
            validate_config(config)

    def test_boolean_options_validation(self):
        """Test that boolean configuration options are properly validated."""
        config = VALID_CONFIG.copy()
        
        boolean_options = ['use_gui', 'generate_mesh', 'generate_topography', 
                          'plot_outer_shell', 'show_plot', 'filter_topography']
        
        for option in boolean_options:
            # String instead of boolean
            config[option] = 'true'
            with pytest.raises(ConfigValidationError, match=f"'{option}' must be a boolean"):
                validate_config(config)
            
            # Integer instead of boolean
            config[option] = 1
            with pytest.raises(ConfigValidationError, match=f"'{option}' must be a boolean"):
                validate_config(config)
            
            # Reset for next iteration
            if option in config:
                del config[option]

    def test_float_format_validation(self):
        """Test float_format field validation."""
        config = VALID_CONFIG.copy()
        
        # Non-string float_format should fail
        config['float_format'] = 8
        with pytest.raises(ConfigValidationError, match="float_format must be a string"):
            validate_config(config)
        
        # Invalid format string should fail
        config['float_format'] = 'not_a_format'
        with pytest.raises(ConfigValidationError, match="float_format must be a valid Python format string"):
            validate_config(config)
        
        # Missing % should fail
        config['float_format'] = '.8f'
        with pytest.raises(ConfigValidationError, match="float_format must be a valid Python format string"):
            validate_config(config)
        
        # Valid format should pass
        config['float_format'] = '%.8f'
        assert validate_config(config) is True

    def test_fill_nan_options(self):
        """Test that valid fill_nan options are accepted."""
        config = VALID_CONFIG.copy()
        
        # 'vertical' should be valid
        config['fill_nan'] = 'vertical'
        assert validate_config(config) is True
        
        # null should be valid
        config['fill_nan'] = None
        assert validate_config(config) is True
        
        # Invalid option should fail
        config['fill_nan'] = 'lateral'
        with pytest.raises(ConfigValidationError, match="fill_nan must be 'vertical' or null"):
            validate_config(config)

    def test_slope_thresholds_range_validation(self):
        """Test that slope thresholds are within valid range (0-90 degrees)."""
        config = VALID_CONFIG.copy()
        config['generate_topography'] = True
        config['topography_output_dir'] = './topo_test'
        config['smoothing_sigma'] = 'auto'
        
        # Negative slope threshold should fail
        config['slope_thresholds'] = [-5, 15, 20]
        with pytest.raises(ConfigValidationError, match="slope_thresholds must be between 0 and 90 degrees"):
            validate_config(config)
        
        # Slope threshold > 90 should fail
        config['slope_thresholds'] = [10, 95, 20]
        with pytest.raises(ConfigValidationError, match="slope_thresholds must be between 0 and 90 degrees"):
            validate_config(config)
        
        # Valid range should pass
        config['slope_thresholds'] = [0, 45, 90]
        assert validate_config(config) is True

    def test_smoothing_sigma_negative_validation(self):
        """Test that numeric smoothing_sigma values must be non-negative."""
        config = VALID_CONFIG.copy()
        config['generate_topography'] = True
        config['topography_output_dir'] = './topo_test'
        config['slope_thresholds'] = [10, 15, 20]
        
        # Negative smoothing_sigma should fail
        config['smoothing_sigma'] = -1.0
        with pytest.raises(ConfigValidationError, match="smoothing_sigma must be >= 0 when numeric"):
            validate_config(config)
        
        # Zero should pass
        config['smoothing_sigma'] = 0.0
        assert validate_config(config) is True
        
        # Positive should pass
        config['smoothing_sigma'] = 1.5
        assert validate_config(config) is True

    def test_plot_color_by_string_validation(self):
        """Test that plot_color_by must be a string."""
        config = VALID_CONFIG.copy()
        
        # Non-string plot_color_by should fail
        config['plot_color_by'] = 123
        with pytest.raises(ConfigValidationError, match="plot_color_by must be a string"):
            validate_config(config)
        
        # List should fail
        config['plot_color_by'] = ['vp']
        with pytest.raises(ConfigValidationError, match="plot_color_by must be a string"):
            validate_config(config)

    def test_anisotropic_mixed_case_variables(self):
        """Test anisotropic validation with mixed case variable names."""
        config = VALID_CONFIG.copy()
        config['basis'] = None
        # Test with mixed case - should still be detected as anisotropic
        config['variable'] = ['C11', 'c12', 'C13', 'c14', 'c15', 'c16',
                             'c22', 'c23', 'c24', 'c25', 'c26',
                             'c33', 'c34', 'c35', 'c36',
                             'c44', 'c45', 'c46',
                             'c55', 'c56',
                             'c66', 'RHO']
        assert validate_config(config) is True

    def test_anisotropic_partial_components_detection(self):
        """Test that partial anisotropic components trigger validation."""
        config = VALID_CONFIG.copy()
        config['basis'] = None
        # Just a few anisotropic components should trigger anisotropic detection
        # but fail validation due to missing components
        config['variable'] = ['c11', 'c22', 'c33']
        with pytest.raises(ConfigValidationError, match="Anisotropic model detected but missing required components"):
            validate_config(config)

    def test_anisotropic_input_basis_definition(self):
        """Test anisotropic input basis definition."""
        config = VALID_CONFIG.copy()
        config['variable'] = ['c11', 'c12', 'c13', 'c14', 'c15', 'c16',
                              'c22', 'c23', 'c24', 'c25', 'c26',
                              'c33', 'c34', 'c35', 'c36',
                              'c44', 'c45', 'c46',
                              'c55', 'c56',
                              'c66', 'rho']
        config['basis'] = None
        assert validate_config(config) is True
        config['basis'] = 123
        with pytest.raises(
            ConfigValidationError,
            match=f"Expected string or None, got int"
        ): validate_config(config)
        config['basis'] = "xyz"
        with pytest.raises(
            ConfigValidationError,
            match=f"'xyz' must consist of exactly three directions separated "
                  f"by underscores. Allowed directions are: \['east', 'west', "
                  f"'north', 'south', 'up', 'down'\]"
        ): validate_config(config)
        config['basis'] = "x_y_z"
        with pytest.raises(
            ConfigValidationError,
            match=f"Invalid direction 'x' in 'x_y_z'. Allowed directions are: "
                  f"\['east', 'west', 'north', 'south', 'up', 'down'\]"
        ): validate_config(config)
        config['basis'] = "east_down_up"
        assert validate_config(config) is True

    def test_mesh_doubling_layers_edge_cases(self):
        """Test edge cases for doubling layers validation."""
        config = VALID_CONFIG.copy()
        config['generate_mesh'] = True
        config['mesh_output_dir'] = './mesh_test'
        config['max_depth'] = 100.0
        config['dx_target_km'] = 5.0
        config['dz_target_km'] = 5.0
        config['max_cpu'] = 64
        
        # Doubling layer exactly at max_depth should pass
        config['doubling_layers'] = [100.0]
        assert validate_config(config) is True
        
        # Doubling layer beyond max_depth should fail
        config['doubling_layers'] = [150.0]
        with pytest.raises(ConfigValidationError, match="All doubling_layers.*must be within the max_depth range"):
            validate_config(config)
        
        # Mixed valid/invalid layers should fail
        config['doubling_layers'] = [50.0, 150.0]
        with pytest.raises(ConfigValidationError, match="All doubling_layers.*must be within the max_depth range"):
            validate_config(config)

    def test_empty_slope_thresholds(self):
        """Test validation with empty slope_thresholds list."""
        config = VALID_CONFIG.copy()
        config['generate_topography'] = True
        config['topography_output_dir'] = './topo_test'
        config['smoothing_sigma'] = 'auto'
        
        # Empty list should pass validation but might cause runtime issues
        config['slope_thresholds'] = []
        assert validate_config(config) is True

    def test_complex_anisotropic_plot_validation(self):
        """Test plot_color_by validation for anisotropic models."""
        config = VALID_CONFIG.copy()
        config['basis'] = None
        config['variable'] = ['c11', 'c12', 'c13', 'c14', 'c15', 'c16',
                             'c22', 'c23', 'c24', 'c25', 'c26',
                             'c33', 'c34', 'c35', 'c36',
                             'c44', 'c45', 'c46',
                             'c55', 'c56',
                             'c66', 'rho']
        
        # Valid anisotropic variable should pass
        config['plot_color_by'] = 'c33'
        assert validate_config(config) is True
        
        # Case insensitive should work
        config['plot_color_by'] = 'C33'
        assert validate_config(config) is True
        
        # Invalid variable for anisotropic model should fail
        config['plot_color_by'] = 'vp'
        with pytest.raises(ConfigValidationError, match="plot_color_by 'vp' must be one of the variables"):
            validate_config(config)

    def test_utm_hemisphere_case_sensitivity(self):
        """Test UTM hemisphere validation is case sensitive."""
        config = VALID_CONFIG.copy()
        
        # Lowercase should fail
        config['utm_hemisphere'] = 'n'
        with pytest.raises(ConfigValidationError, match="utm_hemisphere must be 'N' or 'S'"):
            validate_config(config)
        
        # Mixed case should fail
        config['utm_hemisphere'] = 'North'
        with pytest.raises(ConfigValidationError, match="utm_hemisphere must be 'N' or 'S'"):
            validate_config(config)

    def test_boundary_utm_zones(self):
        """Test boundary UTM zone values."""
        config = VALID_CONFIG.copy()
        
        # Zone 1 should pass
        config['utm_zone'] = 1
        assert validate_config(config) is True
        
        # Zone 60 should pass
        config['utm_zone'] = 60
        assert validate_config(config) is True
        
        # Zone 0 should fail
        config['utm_zone'] = 0
        with pytest.raises(ConfigValidationError, match="utm_zone must be between 1 and 60"):
            validate_config(config)
        
        # Zone 61 should fail
        config['utm_zone'] = 61
        with pytest.raises(ConfigValidationError, match="utm_zone must be between 1 and 60"):
            validate_config(config)

    def test_variable_list_with_non_string_elements(self):
        """Test that variable lists with non-string elements are caught."""
        config = VALID_CONFIG.copy()
        
        # Mixed types in variable list should fail
        config['variable'] = ['vp', 123, 'vs']
        with pytest.raises(ConfigValidationError, match="All elements of variable list must be strings"):
            validate_config(config)
        
        # All non-strings should fail
        config['variable'] = [1, 2, 3]
        with pytest.raises(ConfigValidationError, match="All elements of variable list must be strings"):
            validate_config(config)

    def test_extreme_coordinate_values(self):
        """Test validation with extreme but valid coordinate values."""
        config = VALID_CONFIG.copy()
        
        # Very large UTM coordinates (but still reasonable)
        config['extent'] = [500000, 600000, 4000000, 4100000]
        assert validate_config(config) is True
        
        # Very small but positive spacing
        config['dx'] = 0.001
        config['dy'] = 0.001
        config['dz'] = 0.001
        assert validate_config(config) is True
