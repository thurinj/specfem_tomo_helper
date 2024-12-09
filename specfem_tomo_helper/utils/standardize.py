#!/usr/bin/env python

unit_mapping = {
    'km': 1000,  # Convert kilometers to meters
    'm': 1,
    'km/s': 1000,  # Convert kilometers/seconds to meters/seconds
    'm/s': 1
}

def standardize_units(value, unit):
    if unit in unit_mapping:
        return value * unit_mapping[unit]
    else:
        raise ValueError(f"Unknown unit: {unit}")

