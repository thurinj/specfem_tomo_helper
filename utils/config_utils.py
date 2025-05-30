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
    # Rest of the validation logic will be added here