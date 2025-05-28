\
import netCDF4
import numpy as np
import os

# Define the output directory for the test NetCDF files
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

def create_valid_model_nc(filepath):
    """
    Creates a standard, correctly formatted NetCDF file.
    - Contains latitude, longitude, depth variables.
    - Contains a sample data variable 'vsv'.
    - Depth variable has a 'positive': 'down' attribute.
    - 'vsv' variable has dimensions (depth, latitude, longitude).
    """
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping creation.")
        return

    with netCDF4.Dataset(filepath, 'w', format='NETCDF4') as ncfile:
        # Define dimensions
        ncfile.createDimension('depth', 3)
        ncfile.createDimension('latitude', 4)
        ncfile.createDimension('longitude', 5)

        # Create coordinate variables
        depth = ncfile.createVariable('depth', 'f4', ('depth',))
        depth[:] = np.array([0, 10, 20], dtype='f4')
        depth.units = 'km'
        depth.positive = 'down' # As per IRIS EMC convention often implying positive downwards

        latitude = ncfile.createVariable('latitude', 'f4', ('latitude',))
        latitude[:] = np.array([40, 41, 42, 43], dtype='f4')
        latitude.units = 'degrees_north'

        longitude = ncfile.createVariable('longitude', 'f4', ('longitude',))
        longitude[:] = np.array([-120, -119, -118, -117, -116], dtype='f4')
        longitude.units = 'degrees_east'
        
        # Create data variable 'vsv'
        vsv = ncfile.createVariable('vsv', 'f4', ('depth', 'latitude', 'longitude'), fill_value=-99999.0)
        vsv_data = np.arange(3 * 4 * 5, dtype='f4').reshape(3, 4, 5)
        vsv[:] = vsv_data
        vsv.units = 'km/s'
        vsv.long_name = 'Shear-wave velocity'
        
        # Add some global attributes (optional, but good practice)
        ncfile.title = "Valid Test Model"
        ncfile.Conventions = "CF-1.6"
        ncfile.geospatial_vertical_positive = "down" # Common attribute

    print(f"Created: {filepath}")

def create_missing_latitude_nc(filepath):
    """
    Creates a NetCDF file missing the 'latitude' coordinate variable.
    This should trigger an AssertionError in Nc_model.load_ncfile().
    """
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping creation.")
        return

    with netCDF4.Dataset(filepath, 'w', format='NETCDF4') as ncfile:
        # Define dimensions
        ncfile.createDimension('depth', 3)
        ncfile.createDimension('longitude', 5)
        # Note: No latitude dimension or variable

        # Create coordinate variables (missing latitude)
        depth = ncfile.createVariable('depth', 'f4', ('depth',))
        depth[:] = np.array([0, 10, 20], dtype='f4')
        depth.units = 'km'
        depth.positive = 'down'

        longitude = ncfile.createVariable('longitude', 'f4', ('longitude',))
        longitude[:] = np.array([-120, -119, -118, -117, -116], dtype='f4')
        longitude.units = 'degrees_east'
        
        # Add some data variable (though it won't be used due to missing latitude)
        vsv = ncfile.createVariable('vsv', 'f4', ('depth', 'longitude'), fill_value=-99999.0)
        vsv_data = np.arange(3 * 5, dtype='f4').reshape(3, 5)
        vsv[:] = vsv_data
        vsv.units = 'km/s'
        
        ncfile.title = "Test Model Missing Latitude"

    print(f"Created: {filepath}")

def create_depth_positive_up_nc(filepath):
    """
    Creates a NetCDF file where depth has positive='up' attribute.
    This tests the depth coordinate inversion logic.
    """
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping creation.")
        return

    with netCDF4.Dataset(filepath, 'w', format='NETCDF4') as ncfile:
        # Define dimensions
        ncfile.createDimension('depth', 3)
        ncfile.createDimension('latitude', 4)
        ncfile.createDimension('longitude', 5)

        # Create coordinate variables
        depth = ncfile.createVariable('depth', 'f4', ('depth',))
        depth[:] = np.array([0, 10, 20], dtype='f4')  # Positive values
        depth.units = 'km'
        depth.positive = 'up'  # This should trigger inversion

        latitude = ncfile.createVariable('latitude', 'f4', ('latitude',))
        latitude[:] = np.array([40, 41, 42, 43], dtype='f4')
        latitude.units = 'degrees_north'

        longitude = ncfile.createVariable('longitude', 'f4', ('longitude',))
        longitude[:] = np.array([-120, -119, -118, -117, -116], dtype='f4')
        longitude.units = 'degrees_east'
        
        # Create data variable
        vsv = ncfile.createVariable('vsv', 'f4', ('depth', 'latitude', 'longitude'), fill_value=-99999.0)
        vsv_data = np.arange(3 * 4 * 5, dtype='f4').reshape(3, 4, 5)
        vsv[:] = vsv_data
        vsv.units = 'km/s'
        
        ncfile.title = "Test Model Depth Positive Up"
        ncfile.geospatial_vertical_positive = "up"

    print(f"Created: {filepath}")

def create_masked_data_nc(filepath):
    """
    Creates a NetCDF file with masked data variables.
    This tests the mask handling in load_variable().
    """
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping creation.")
        return

    with netCDF4.Dataset(filepath, 'w', format='NETCDF4') as ncfile:
        # Define dimensions
        ncfile.createDimension('depth', 3)
        ncfile.createDimension('latitude', 4)
        ncfile.createDimension('longitude', 5)

        # Create coordinate variables
        depth = ncfile.createVariable('depth', 'f4', ('depth',))
        depth[:] = np.array([0, 10, 20], dtype='f4')
        depth.units = 'km'
        depth.positive = 'down'

        latitude = ncfile.createVariable('latitude', 'f4', ('latitude',))
        latitude[:] = np.array([40, 41, 42, 43], dtype='f4')
        latitude.units = 'degrees_north'

        longitude = ncfile.createVariable('longitude', 'f4', ('longitude',))
        longitude[:] = np.array([-120, -119, -118, -117, -116], dtype='f4')
        longitude.units = 'degrees_east'
        
        # Create data variable with mask
        vsv = ncfile.createVariable('vsv', 'f4', ('depth', 'latitude', 'longitude'), fill_value=-99999.0)
        vsv_data = np.arange(3 * 4 * 5, dtype='f4').reshape(3, 4, 5)
        # Create a mask - mask some values
        mask = np.zeros_like(vsv_data, dtype=bool)
        mask[0, 0, 0] = True  # Mask first element
        mask[1, 2, 3] = True  # Mask another element
        
        # Apply mask to data
        vsv_masked = np.ma.masked_array(vsv_data, mask=mask)
        vsv[:] = vsv_masked
        vsv.units = 'km/s'
        
        ncfile.title = "Test Model with Masked Data"
        ncfile.geospatial_vertical_positive = "down"

    print(f"Created: {filepath}")

def create_missing_value_data_nc(filepath):
    """
    Creates a NetCDF file with a missing_value attribute instead of masks.
    This tests the missing_value handling in load_variable().
    """
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping creation.")
        return

    with netCDF4.Dataset(filepath, 'w', format='NETCDF4') as ncfile:
        # Define dimensions
        ncfile.createDimension('depth', 3)
        ncfile.createDimension('latitude', 4)
        ncfile.createDimension('longitude', 5)

        # Create coordinate variables
        depth = ncfile.createVariable('depth', 'f4', ('depth',))
        depth[:] = np.array([0, 10, 20], dtype='f4')
        depth.units = 'km'
        depth.positive = 'down'

        latitude = ncfile.createVariable('latitude', 'f4', ('latitude',))
        latitude[:] = np.array([40, 41, 42, 43], dtype='f4')
        latitude.units = 'degrees_north'

        longitude = ncfile.createVariable('longitude', 'f4', ('longitude',))
        longitude[:] = np.array([-120, -119, -118, -117, -116], dtype='f4')
        longitude.units = 'degrees_east'
        
        # Create data variable with missing_value
        vsv = ncfile.createVariable('vsv', 'f4', ('depth', 'latitude', 'longitude'))
        vsv_data = np.arange(3 * 4 * 5, dtype='f4').reshape(3, 4, 5)
        # Set some values to missing_value
        missing_val = -99999.0
        vsv_data[0, 0, 0] = missing_val
        vsv_data[1, 2, 3] = missing_val
        vsv[:] = vsv_data
        vsv.missing_value = missing_val
        vsv.units = 'km/s'
        
        ncfile.title = "Test Model with Missing Values"
        ncfile.geospatial_vertical_positive = "down"

    print(f"Created: {filepath}")

def create_rho_g_cm3_nc(filepath):
    """
    Creates a NetCDF file with density in g/cm³ units.
    This tests the unit conversion in load_variable().
    """
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping creation.")
        return

    with netCDF4.Dataset(filepath, 'w', format='NETCDF4') as ncfile:
        # Define dimensions
        ncfile.createDimension('depth', 3)
        ncfile.createDimension('latitude', 4)
        ncfile.createDimension('longitude', 5)

        # Create coordinate variables
        depth = ncfile.createVariable('depth', 'f4', ('depth',))
        depth[:] = np.array([0, 10, 20], dtype='f4')
        depth.units = 'km'
        depth.positive = 'down'

        latitude = ncfile.createVariable('latitude', 'f4', ('latitude',))
        latitude[:] = np.array([40, 41, 42, 43], dtype='f4')
        latitude.units = 'degrees_north'

        longitude = ncfile.createVariable('longitude', 'f4', ('longitude',))
        longitude[:] = np.array([-120, -119, -118, -117, -116], dtype='f4')
        longitude.units = 'degrees_east'
        
        # Create density variable in g/cm³
        rho = ncfile.createVariable('rho', 'f4', ('depth', 'latitude', 'longitude'))
        rho_data = np.full((3, 4, 5), 2.5, dtype='f4')  # Typical rock density in g/cm³
        rho[:] = rho_data
        rho.units = 'g/cm^3'
        rho.long_name = 'Density'
        
        ncfile.title = "Test Model with Density in g/cm³"
        ncfile.geospatial_vertical_positive = "down"

    print(f"Created: {filepath}")

def create_rho_kg_m3_nc(filepath):
    """
    Creates a NetCDF file with density in kg/m³ units.
    This tests that no conversion occurs when units are already SI.
    """
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping creation.")
        return

    with netCDF4.Dataset(filepath, 'w', format='NETCDF4') as ncfile:
        # Define dimensions
        ncfile.createDimension('depth', 3)
        ncfile.createDimension('latitude', 4)
        ncfile.createDimension('longitude', 5)

        # Create coordinate variables
        depth = ncfile.createVariable('depth', 'f4', ('depth',))
        depth[:] = np.array([0, 10, 20], dtype='f4')
        depth.units = 'km'
        depth.positive = 'down'

        latitude = ncfile.createVariable('latitude', 'f4', ('latitude',))
        latitude[:] = np.array([40, 41, 42, 43], dtype='f4')
        latitude.units = 'degrees_north'

        longitude = ncfile.createVariable('longitude', 'f4', ('longitude',))
        longitude[:] = np.array([-120, -119, -118, -117, -116], dtype='f4')
        longitude.units = 'degrees_east'
        
        # Create density variable in kg/m³
        rho = ncfile.createVariable('rho', 'f4', ('depth', 'latitude', 'longitude'))
        rho_data = np.full((3, 4, 5), 2500.0, dtype='f4')  # Same density in kg/m³
        rho[:] = rho_data
        rho.units = 'kg/m^3'
        rho.long_name = 'Density'
        
        ncfile.title = "Test Model with Density in kg/m³"
        ncfile.geospatial_vertical_positive = "down"

    print(f"Created: {filepath}")

def create_rho_ambiguous_low_nc(filepath):
    """
    Creates a NetCDF file with density without clear units, low values (< 50).
    This tests the fallback unit detection based on median values.
    """
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping creation.")
        return

    with netCDF4.Dataset(filepath, 'w', format='NETCDF4') as ncfile:
        # Define dimensions
        ncfile.createDimension('depth', 3)
        ncfile.createDimension('latitude', 4)
        ncfile.createDimension('longitude', 5)

        # Create coordinate variables
        depth = ncfile.createVariable('depth', 'f4', ('depth',))
        depth[:] = np.array([0, 10, 20], dtype='f4')
        depth.units = 'km'
        depth.positive = 'down'

        latitude = ncfile.createVariable('latitude', 'f4', ('latitude',))
        latitude[:] = np.array([40, 41, 42, 43], dtype='f4')
        latitude.units = 'degrees_north'

        longitude = ncfile.createVariable('longitude', 'f4', ('longitude',))
        longitude[:] = np.array([-120, -119, -118, -117, -116], dtype='f4')
        longitude.units = 'degrees_east'
        
        # Create density variable with low values (likely g/cm³)
        rho = ncfile.createVariable('rho', 'f4', ('depth', 'latitude', 'longitude'))
        rho_data = np.full((3, 4, 5), 2.5, dtype='f4')  # Low values, should be converted
        rho[:] = rho_data
        # Intentionally no units or ambiguous units
        rho.long_name = 'Density'
        
        ncfile.title = "Test Model with Ambiguous Low Density"
        ncfile.geospatial_vertical_positive = "down"

    print(f"Created: {filepath}")

def main():
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Generate the NetCDF files
    create_valid_model_nc(os.path.join(OUTPUT_DIR, "valid_model.nc"))
    create_missing_latitude_nc(os.path.join(OUTPUT_DIR, "missing_latitude_model.nc"))
    create_depth_positive_up_nc(os.path.join(OUTPUT_DIR, "depth_positive_up_model.nc"))
    create_masked_data_nc(os.path.join(OUTPUT_DIR, "masked_data_model.nc"))
    create_missing_value_data_nc(os.path.join(OUTPUT_DIR, "missing_value_model.nc"))
    create_rho_g_cm3_nc(os.path.join(OUTPUT_DIR, "rho_g_cm3_model.nc"))
    create_rho_kg_m3_nc(os.path.join(OUTPUT_DIR, "rho_kg_m3_model.nc"))
    create_rho_ambiguous_low_nc(os.path.join(OUTPUT_DIR, "rho_ambiguous_low_model.nc"))

if __name__ == "__main__":
    main()
