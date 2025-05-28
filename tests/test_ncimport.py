import pytest
import os
import netCDF4
import numpy as np
from specfem_tomo_helper.fileimport.ncimport import Nc_model, Model_array

# Define the path to the test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VALID_MODEL_NC = os.path.join(TEST_DATA_DIR, "valid_model.nc")

@pytest.fixture(scope="module")
def valid_model_path():
    """Fixture to provide the path to the valid_model.nc file."""
    # We assume generate_test_netcdf_data.py has been run and the file exists
    if not os.path.exists(VALID_MODEL_NC):
        pytest.fail(f"Test data file not found: {VALID_MODEL_NC}. Run generate_test_netcdf_data.py first.")
    return VALID_MODEL_NC

@pytest.fixture(scope="module")
def missing_latitude_model_path():
    """Fixture to provide the path to the missing_latitude_model.nc file."""
    path = os.path.join(TEST_DATA_DIR, "missing_latitude_model.nc")
    if not os.path.exists(path):
        pytest.fail(f"Test data file not found: {path}. Run generate_test_netcdf_data.py first.")
    return path

@pytest.fixture(scope="module")
def depth_positive_up_model_path():
    """Fixture to provide the path to the depth_positive_up_model.nc file."""
    path = os.path.join(TEST_DATA_DIR, "depth_positive_up_model.nc")
    if not os.path.exists(path):
        pytest.fail(f"Test data file not found: {path}. Run generate_test_netcdf_data.py first.")
    return path

def test_nc_model_init_load_valid_model(valid_model_path):
    """Test Nc_model initialization and load_ncfile with a valid NetCDF file."""
    model = Nc_model(valid_model_path)
    assert model.path == valid_model_path
    assert model.dataset is not None
    assert isinstance(model.dataset, netCDF4.Dataset)
    # Check if essential coordinate variables were identified (they are loaded later)
    assert 'latitude' in model.dataset.variables
    assert 'longitude' in model.dataset.variables
    assert 'depth' in model.dataset.variables
    model.dataset.close() # Good practice to close the file

def test_nc_model_init_file_not_found():
    """Test Nc_model initialization with a non-existent file."""
    non_existent_path = os.path.join(TEST_DATA_DIR, "non_existent_model.nc")
    with pytest.raises(FileNotFoundError): # Or the specific error netCDF4 might raise
        Nc_model(non_existent_path)

def test_nc_model_init_missing_latitude(missing_latitude_model_path):
    """Test Nc_model initialization with a file missing the latitude coordinate."""
    with pytest.raises(AssertionError, match="Missing latitude key in netCDF variables"):
        Nc_model(missing_latitude_model_path)

def test_load_coordinates_basic(valid_model_path):
    """Test loading coordinates from a valid NetCDF file."""
    model = Nc_model(valid_model_path)
    
    # Load coordinates
    lon, lat, depth = model.load_coordinates()
    
    # Check return values
    assert isinstance(lon, np.ndarray)
    assert isinstance(lat, np.ndarray)
    assert isinstance(depth, np.ndarray)
    
    # Check that instance variables are populated
    np.testing.assert_array_equal(model.lon, lon)
    np.testing.assert_array_equal(model.lat, lat)
    np.testing.assert_array_equal(model.depth, depth)
     # Check expected values (from create_valid_model_nc)
    expected_lon = np.array([-120, -119, -118, -117, -116], dtype='f4')
    expected_lat = np.array([40, 41, 42, 43], dtype='f4')
    # Since positive='down' and geospatial_vertical_positive='down', depth gets inverted (multiplied by -1)
    expected_depth = np.array([0, -10, -20], dtype='f4')

    np.testing.assert_array_equal(lon, expected_lon)
    np.testing.assert_array_equal(lat, expected_lat)
    np.testing.assert_array_equal(depth, expected_depth)
    
    model.dataset.close()

def test_load_coordinates_depth_positive_up(depth_positive_up_model_path):
    """Test loading coordinates when depth has positive='up' attribute."""
    model = Nc_model(depth_positive_up_model_path)
    
    # Load coordinates
    lon, lat, depth = model.load_coordinates()
    
    # The original depth values are [0, 10, 20] with positive='up'
    # The logic checks if 'up' is in the positive attribute, and if so, keeps values as-is
    # If 'up' is NOT in the positive attribute, it multiplies by -1
    # Since positive='up', we expect the original values [0, 10, 20]
    expected_depth = np.array([0, 10, 20], dtype='f4')
    np.testing.assert_array_equal(depth, expected_depth)
    
    model.dataset.close()

# Test load_variable() method
def test_load_variable_basic(valid_model_path):
    """Test loading a basic variable from a valid NetCDF file."""
    model = Nc_model(valid_model_path)
    model.load_coordinates()
    
    # Load the vsv variable
    model_array = model.load_variable('vsv')
    
    # Check return type
    assert isinstance(model_array, Model_array)
    assert model_array.name == 'vsv'
    assert isinstance(model_array.values, np.ndarray)
    
    # Check shape - should be (lat, lon, depth) after transposition
    expected_shape = (4, 5, 3)  # lat=4, lon=5, depth=3
    assert model_array.values.shape == expected_shape
    
    # Check that values are reasonable (not all NaN)
    assert not np.all(np.isnan(model_array.values))
    
    model.dataset.close()

def test_load_variable_with_mask():
    """Test loading a variable with masked data."""
    filepath = os.path.join(TEST_DATA_DIR, "masked_data_model.nc")
    model = Nc_model(filepath)
    model.load_coordinates()
    
    model_array = model.load_variable('vsv')
    
    # Check that masked values are converted to NaN
    assert np.isnan(model_array.values[0, 0, 0])  # First masked element
    assert np.isnan(model_array.values[2, 3, 1])  # Second masked element (after transposition)
    
    # Check that non-masked values are not NaN
    assert not np.isnan(model_array.values[0, 1, 0])  # Should be valid
    
    model.dataset.close()

def test_load_variable_with_missing_value():
    """Test loading a variable with missing_value attribute."""
    filepath = os.path.join(TEST_DATA_DIR, "missing_value_model.nc")
    model = Nc_model(filepath)
    model.load_coordinates()
    
    model_array = model.load_variable('vsv')
    
    # Check that missing values are converted to NaN
    assert np.isnan(model_array.values[0, 0, 0])  # First missing element
    assert np.isnan(model_array.values[2, 3, 1])  # Second missing element (after transposition)
    
    # Check that valid values are preserved
    assert not np.isnan(model_array.values[0, 1, 0])  # Should be valid
    
    model.dataset.close()

def test_load_variable_rho_g_cm3_conversion():
    """Test density unit conversion from g/cm³ to kg/m³."""
    filepath = os.path.join(TEST_DATA_DIR, "rho_g_cm3_model.nc")
    model = Nc_model(filepath)
    model.load_coordinates()
    
    model_array = model.load_variable('rho')
    
    # Original data was 2.5 g/cm³, should be converted to 2500 kg/m³
    expected_value = 2500.0
    assert np.allclose(model_array.values, expected_value)
    
    model.dataset.close()

def test_load_variable_rho_kg_m3_no_conversion():
    """Test that density in kg/m³ is not converted."""
    filepath = os.path.join(TEST_DATA_DIR, "rho_kg_m3_model.nc")
    model = Nc_model(filepath)
    model.load_coordinates()
    
    model_array = model.load_variable('rho')
    
    # Original data was 2500 kg/m³, should remain unchanged
    expected_value = 2500.0
    assert np.allclose(model_array.values, expected_value)
    
    model.dataset.close()

def test_load_variable_rho_ambiguous_conversion():
    """Test density unit conversion fallback based on median value."""
    filepath = os.path.join(TEST_DATA_DIR, "rho_ambiguous_low_model.nc")
    model = Nc_model(filepath)
    model.load_coordinates()
    
    model_array = model.load_variable('rho')
    
    # Original data was 2.5 (ambiguous units), median < 50, should be converted to 2500
    expected_value = 2500.0
    assert np.allclose(model_array.values, expected_value)
    
    model.dataset.close()

def test_load_variable_nonexistent():
    """Test loading a non-existent variable raises KeyError."""
    filepath = os.path.join(TEST_DATA_DIR, "valid_model.nc")
    model = Nc_model(filepath)
    model.load_coordinates()
    
    with pytest.raises(KeyError):
        model.load_variable('non_existent_variable')
    
    model.dataset.close()

# Test Model_array class
def test_model_array_creation():
    """Test Model_array creation and basic functionality."""
    name = "test_var"
    values = np.array([1, 2, 3, 4, 5])
    
    model_array = Model_array(name, values)
    
    assert model_array.name == name
    np.testing.assert_array_equal(model_array.values, values)
