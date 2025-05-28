import os

class ConfigValidationError(Exception):
    pass

def auto_detect_utm_from_extent(extent):
    """
    Auto-detect UTM zone and hemisphere from geographic extent.
    
    Parameters
    ----------
    extent : list
        [min_lon, max_lon, min_lat, max_lat] in decimal degrees
        
    Returns
    -------
    tuple
        (utm_zone, hemisphere) where utm_zone is int and hemisphere is 'N' or 'S'
        
    Raises
    ------
    ConfigValidationError
        If extent doesn't appear to be in geographic coordinates
    """
    if len(extent) != 4:
        raise ConfigValidationError("extent must be a list of 4 numbers [min_lon, max_lon, min_lat, max_lat]")
    
    min_lon, max_lon, min_lat, max_lat = extent
    
    # Check if coordinates look like geographic coordinates (longitude/latitude)
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180 and 
            -90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ConfigValidationError(
            "Extent values don't appear to be in geographic coordinates (longitude/latitude). "
            "Auto-detection only works with geographic coordinates. "
            f"Got extent: {extent}"
        )
    
    # Use center point to determine UTM zone
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    
    # Calculate UTM zone from longitude
    utm_zone = int((center_lon + 180) / 6) + 1
    
    # Determine hemisphere from latitude
    hemisphere = 'N' if center_lat >= 0 else 'S'
    
    return utm_zone, hemisphere

def is_geographic_extent(extent):
    """
    Check if extent appears to be in geographic coordinates (longitude/latitude).
    
    Parameters
    ----------
    extent : list
        [min_x, max_x, min_y, max_y] coordinates
        
    Returns
    -------
    bool
        True if extent appears to be in geographic coordinates
    """
    if len(extent) != 4:
        return False
    
    # First check if all values are numeric
    if not all(isinstance(val, (int, float)) for val in extent):
        return False
    
    min_x, max_x, min_y, max_y = extent
    
    # Geographic coordinates should be within valid longitude/latitude ranges
    return (-180 <= min_x <= 180 and -180 <= max_x <= 180 and 
            -90 <= min_y <= 90 and -90 <= max_y <= 90)

def validate_config(config):
    # Required fields
    required = [
        ('data_path', str),
        ('dx', (int, float)),
        ('dy', (int, float)),
        ('dz', (int, float)),
        ('z_min', (int, float)),
        ('z_max', (int, float)),
        ('variable', (str, list)),
    ]
    
    for key, typ in required:
        if key not in config:
            raise ConfigValidationError(f"Missing required config option: {key}")
        if not isinstance(config[key], typ):
            raise ConfigValidationError(f"Config option '{key}' must be of type {typ}, got {type(config[key])}")
    
    # Early validation of extent (before UTM logic that uses it)
    if 'extent' in config and config['extent'] is not None:
        if not (isinstance(config['extent'], list) and len(config['extent']) == 4):
            raise ConfigValidationError("extent must be a list of 4 numbers")
        for val in config['extent']:
            if not isinstance(val, (int, float)):
                raise ConfigValidationError("extent values must be numbers")
        # New: extent must NOT be in geographic coordinates (lat/lon)
        if is_geographic_extent(config['extent']):
            raise ConfigValidationError("'extent' must be specified in UTM coordinates, not geographic (lat/lon). Please provide UTM coordinates and specify utm_zone and utm_hemisphere.")
    
    # UTM validation logic
    utm_zone = config.get('utm_zone')
    utm_hemisphere = config.get('utm_hemisphere')
    extent = config.get('extent')
    use_gui = config.get('use_gui', False)
    
    # Check if UTM values are missing or null
    utm_zone_is_null = utm_zone is None
    utm_hemisphere_is_null = utm_hemisphere is None
    extent_is_null = extent is None
    
    # UTM fields can only be null if: no extent is specified AND GUI is activated
    if (utm_zone_is_null or utm_hemisphere_is_null):
        if not extent_is_null and not use_gui:
            # Extent is provided but no GUI - can only be null if extent is geographic for auto-detection
            if extent is not None and is_geographic_extent(extent):
                # This is OK - we can auto-detect from geographic extent
                pass
            else:
                raise ConfigValidationError(
                    "UTM zone and hemisphere can only be null when:\n"
                    "1. No extent is specified AND use_gui is True (GUI will determine projection), OR\n"
                    "2. Extent is provided in geographic coordinates (longitude/latitude) for auto-detection.\n"
                    f"Current config: utm_zone={utm_zone}, utm_hemisphere={utm_hemisphere}, "
                    f"extent={'null' if extent is None else 'provided'}, use_gui={use_gui}"
                )
        elif extent_is_null and not use_gui:
            # No extent and no GUI - UTM must be specified
            raise ConfigValidationError(
                "UTM zone and hemisphere are required when no extent is specified and use_gui is False.\n"
                "Either specify utm_zone and utm_hemisphere, or set use_gui to True."
            )
    
    # UTM fields validation (only when present and not None)
    if utm_zone is not None:
        if not isinstance(utm_zone, int):
            raise ConfigValidationError(f"utm_zone must be an integer, got {type(utm_zone)}")
    
    if utm_hemisphere is not None:
        if not isinstance(utm_hemisphere, str):
            raise ConfigValidationError(f"utm_hemisphere must be a string, got {type(utm_hemisphere)}")
    
    # Check for potential issue: extent in UTM coordinates but no UTM zone specified
    if (extent is not None and 
        not is_geographic_extent(extent) and
        (utm_zone is None or utm_hemisphere is None)):
        print("Warning: extent appears to be in UTM coordinates but utm_zone/utm_hemisphere not specified.")
        print("Consider specifying utm_zone and utm_hemisphere, or provide extent in geographic coordinates (longitude/latitude).")
    
    # File existence
    if not os.path.isfile(config['data_path']):
        raise ConfigValidationError(f"data_path file does not exist: {config['data_path']}")
    # Value checks
    for k in ['dx', 'dy', 'dz']:
        if config[k] <= 0:
            raise ConfigValidationError(f"{k} must be > 0")
    if config['z_min'] >= config['z_max']:
        raise ConfigValidationError("z_min must be less than z_max")
    
    # UTM validation only if UTM fields are present and not None
    if 'utm_zone' in config and config['utm_zone'] is not None:
        if not (1 <= config['utm_zone'] <= 60):
            raise ConfigValidationError("utm_zone must be between 1 and 60")
    if 'utm_hemisphere' in config and config['utm_hemisphere'] is not None:
        if config['utm_hemisphere'] not in ['N', 'S']:
            raise ConfigValidationError("utm_hemisphere must be 'N' or 'S'")
    # variable
    if isinstance(config['variable'], list):
        if not config['variable']:
            raise ConfigValidationError("variable list must not be empty")
        for v in config['variable']:
            if not isinstance(v, str):
                raise ConfigValidationError("All elements of variable list must be strings")
    # fill_nan
    if 'fill_nan' in config and config['fill_nan'] not in [None, 'vertical', 'horizontal']:
        raise ConfigValidationError("fill_nan must be 'vertical', 'horizontal', or null")
    # Mesh options
    if config.get('generate_mesh', False):
        mesh_required = [
            ('mesh_output_dir', str),
            ('max_depth', (int, float)),
            ('dx_target_km', (int, float)),
            ('dz_target_km', (int, float)),
            ('max_cpu', int),
            ('doubling_layers', list),
        ]
        for key, typ in mesh_required:
            if key not in config:
                raise ConfigValidationError(f"Missing mesh config option: {key}")
            if not isinstance(config[key], typ):
                raise ConfigValidationError(f"Mesh config option '{key}' must be of type {typ}")
        # All mesh depths must be positive (positive-down convention)
        if config['max_depth'] <= 0 or config['dx_target_km'] <= 0 or config['dz_target_km'] <= 0 or config['max_cpu'] <= 0:
            raise ConfigValidationError("Mesh numeric options must be > 0 (positive-down convention)")
        if not all(isinstance(x, (int, float)) for x in config['doubling_layers']):
            raise ConfigValidationError("All doubling_layers must be numbers (in km)")
        if not all(0 <= x <= config['max_depth'] for x in config['doubling_layers']):
            raise ConfigValidationError("All doubling_layers (in km) must be within the max_depth range [0, max_depth] (in km) and positive (positive-down convention)")
    # Topography options
    if config.get('generate_topography', False):
        topo_required = [
            ('topography_output_dir', str),
            ('slope_thresholds', list),
            ('smoothing_sigma', (str, int, float)),
        ]
        for key, typ in topo_required:
            if key not in config:
                raise ConfigValidationError(f"Missing topography config option: {key}")
            if not isinstance(config[key], typ):
                raise ConfigValidationError(f"Topography config option '{key}' must be of type {typ}")
        if not all(isinstance(x, (int, float)) for x in config['slope_thresholds']):
            raise ConfigValidationError("All slope_thresholds must be numbers")
        if not (isinstance(config['smoothing_sigma'], (int, float)) or config['smoothing_sigma'] == 'auto'):
            raise ConfigValidationError("smoothing_sigma must be a number or 'auto'")
    # Tomography output
    if 'tomo_output_dir' in config and not isinstance(config['tomo_output_dir'], str):
        raise ConfigValidationError("tomo_output_dir must be a string")
    # Plotting
    if 'plot_color_by' in config and config['plot_color_by'] not in ['vp', 'vs', 'rho']:
        raise ConfigValidationError("plot_color_by must be 'vp', 'vs', or 'rho'")
    # All checks passed
    return True
