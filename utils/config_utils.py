# config_validator.py

class ConfigValidationError(Exception):
    pass

def is_geographic_extent(extent):
    # Dummy implementation for the sake of example
    # Replace with the actual logic to determine if the extent is in geographic coordinates
    return False

def validate_config(config):
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

    # ... rest of the validation logic ...