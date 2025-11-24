import pytest
import numpy as np
import pyproj
from unittest.mock import Mock
from specfem_tomo_helper.utils.topo_processor import TopographyProcessor


class MockInterpolator:
    """Mock interpolator for testing."""
    def __init__(self, x_coords, y_coords):
        self.x_interp_coordinates = np.array(x_coords)
        self.y_interp_coordinates = np.array(y_coords)


class MockMeshProcessor:
    """Mock mesh processor for testing."""
    def __init__(self, dx=None, dy=None):
        if dx is not None and dy is not None:
            self.selected_config = {'dx': dx, 'dy': dy}
        else:
            self.selected_config = None


def test_topo_resolution_priority_1_user():
    """Test priority 1: Use user-specified resolution."""
    # Setup
    x_coords = np.linspace(0, 100000, 21)  # 5000m spacing
    y_coords = np.linspace(0, 100000, 21)
    interpolator = MockInterpolator(x_coords, y_coords)
    
    # Mesh with different resolution (2500m)
    mesh_processor = MockMeshProcessor(dx=2500, dy=2500)
    
    proj = pyproj.Proj(proj='utm', zone=33, datum='WGS84')
    
    # Create TopographyProcessor with user-specified resolution (takes priority over mesh)
    topo = TopographyProcessor(
        interpolator,
        proj,
        mesh_processor=mesh_processor,
        topo_res=1000
    )
    
    # Check that user resolution is used (not mesh)
    dx_actual = topo.x_interp[1] - topo.x_interp[0]
    dy_actual = topo.y_interp[1] - topo.y_interp[0]
    
    assert abs(dx_actual - 1000) < 1, f"Expected dx=1000, got {dx_actual}"
    assert abs(dy_actual - 1000) < 1, f"Expected dy=1000, got {dy_actual}"
    assert len(topo.x_interp) == 101  # 100000 / 1000 + 1


def test_topo_resolution_priority_2_mesh():
    """Test priority 2: Use mesh resolution when no user resolution specified."""
    # Setup
    x_coords = np.linspace(0, 100000, 21)  # 5000m spacing
    y_coords = np.linspace(0, 100000, 21)
    interpolator = MockInterpolator(x_coords, y_coords)
    
    # Mesh with 2500m resolution
    mesh_processor = MockMeshProcessor(dx=2500, dy=2500)
    
    proj = pyproj.Proj(proj='utm', zone=33, datum='WGS84')
    
    # Create TopographyProcessor without user resolution
    topo = TopographyProcessor(
        interpolator,
        proj,
        mesh_processor=mesh_processor
    )
    
    # Check that mesh resolution is used
    dx_actual = topo.x_interp[1] - topo.x_interp[0]
    dy_actual = topo.y_interp[1] - topo.y_interp[0]
    
    assert abs(dx_actual - 2500) < 1, f"Expected dx=2500, got {dx_actual}"
    assert abs(dy_actual - 2500) < 1, f"Expected dy=2500, got {dy_actual}"
    assert len(topo.x_interp) == 41  # 100000 / 2500 + 1


def test_topo_resolution_priority_3_etopo():
    """Test priority 3: Use ETOPO1 native resolution as default."""
    # Setup
    x_coords = np.linspace(0, 100000, 21)  # 5000m spacing
    y_coords = np.linspace(0, 100000, 21)
    interpolator = MockInterpolator(x_coords, y_coords)
    
    proj = pyproj.Proj(proj='utm', zone=33, datum='WGS84')
    
    # Create TopographyProcessor without mesh or user resolution
    topo = TopographyProcessor(
        interpolator,
        proj
    )
    
    # Check that ETOPO1 resolution (~1800m) is used
    dx_actual = topo.x_interp[1] - topo.x_interp[0]
    dy_actual = topo.y_interp[1] - topo.y_interp[0]
    
    # ETOPO1 resolution is 1800m - allow for rounding to integer grid points
    assert abs(dx_actual - 1800) < 50, f"Expected dx≈1800, got {dx_actual}"
    assert abs(dy_actual - 1800) < 50, f"Expected dy≈1800, got {dy_actual}"


def test_topo_resolution_priority_override():
    """Test that user resolution has highest priority."""
    # Setup
    x_coords = np.linspace(0, 100000, 21)
    y_coords = np.linspace(0, 100000, 21)
    interpolator = MockInterpolator(x_coords, y_coords)
    
    # Mesh with 2500m resolution
    mesh_processor = MockMeshProcessor(dx=2500, dy=2500)
    
    proj = pyproj.Proj(proj='utm', zone=33, datum='WGS84')
    
    # User wants 1000m - this should take priority over mesh
    topo = TopographyProcessor(
        interpolator,
        proj,
        mesh_processor=mesh_processor,
        topo_res=1000
    )
    
    # Check that USER resolution is used (priority 1 beats priority 2)
    dx_actual = topo.x_interp[1] - topo.x_interp[0]
    dy_actual = topo.y_interp[1] - topo.y_interp[0]
    
    assert abs(dx_actual - 1000) < 1, f"Expected user dx=1000 to take priority, got {dx_actual}"
    assert abs(dy_actual - 1000) < 1, f"Expected user dy=1000 to take priority, got {dy_actual}"


def test_topo_resolution_mesh_without_config():
    """Test that mesh processor without config falls back to next priority."""
    # Setup
    x_coords = np.linspace(0, 100000, 21)
    y_coords = np.linspace(0, 100000, 21)
    interpolator = MockInterpolator(x_coords, y_coords)
    
    # Mesh processor exists but no config generated yet
    mesh_processor = MockMeshProcessor()  # selected_config is None
    
    proj = pyproj.Proj(proj='utm', zone=33, datum='WGS84')
    
    # Create with user-specified resolution
    topo = TopographyProcessor(
        interpolator,
        proj,
        mesh_processor=mesh_processor,
        topo_res=3000
    )
    
    # Should use user resolution (priority 1)
    # Allow some tolerance for integer grid point rounding
    dx_actual = topo.x_interp[1] - topo.x_interp[0]
    dy_actual = topo.y_interp[1] - topo.y_interp[0]
    
    assert abs(dx_actual - 3000) < 50, f"Expected user dx≈3000, got {dx_actual}"
    assert abs(dy_actual - 3000) < 50, f"Expected user dy≈3000, got {dy_actual}"


def test_topo_grid_extent_preserved():
    """Test that topography grid extent matches interpolator extent regardless of resolution."""
    # Setup with specific extent
    x_coords = np.linspace(50000, 150000, 21)
    y_coords = np.linspace(4000000, 4100000, 21)
    interpolator = MockInterpolator(x_coords, y_coords)
    
    proj = pyproj.Proj(proj='utm', zone=33, datum='WGS84')
    
    # Create with different resolution
    topo = TopographyProcessor(
        interpolator,
        proj,
        topo_res=2000
    )
    
    # Check extent is preserved
    assert abs(topo.x_interp[0] - 50000) < 1
    assert abs(topo.x_interp[-1] - 150000) < 1
    assert abs(topo.y_interp[0] - 4000000) < 1
    assert abs(topo.y_interp[-1] - 4100000) < 1
