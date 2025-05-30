#!/usr/bin/env python

import pytest
import tempfile
import os
import yaml
from unittest.mock import Mock, patch
import numpy as np

from specfem_tomo_helper.cli import is_anisotropic_model, validate_anisotropic_variables
from specfem_tomo_helper.utils.config_utils import validate_config, ConfigValidationError


class TestAnisotropicWorkflow:
    """Test anisotropic model support in the CLI workflow."""
    
    def test_anisotropic_model_detection(self):
        """Test detection of anisotropic models."""
        # Test isotropic variables
        assert not is_anisotropic_model(['vp', 'vs', 'rho'])
        assert not is_anisotropic_model('vsv')
        
        # Test anisotropic variables
        assert is_anisotropic_model(['c11', 'c12', 'c13'])
        assert is_anisotropic_model('c44')
        
        # Test mixed case
        assert is_anisotropic_model(['C11', 'C22', 'rho'])
        
        # Test full anisotropic set
        full_anisotropic = [
            'c11', 'c12', 'c13', 'c14', 'c15', 'c16',
            'c22', 'c23', 'c24', 'c25', 'c26',
            'c33', 'c34', 'c35', 'c36',
            'c44', 'c45', 'c46',
            'c55', 'c56',
            'c66', 'rho'
        ]
        assert is_anisotropic_model(full_anisotropic)

    def test_anisotropic_variables_complete(self):
        """Test validation passes with complete anisotropic variable set."""
        full_anisotropic = [
            'c11', 'c12', 'c13', 'c14', 'c15', 'c16',
            'c22', 'c23', 'c24', 'c25', 'c26',
            'c33', 'c34', 'c35', 'c36',
            'c44', 'c45', 'c46',
            'c55', 'c56',
            'c66', 'rho'
        ]
        assert validate_anisotropic_variables(full_anisotropic) == True

    def test_anisotropic_variables_incomplete(self):
        """Test validation fails with incomplete anisotropic variable set."""
        incomplete_anisotropic = ['c11', 'c12', 'c13', 'rho']  # Missing many components
        
        with pytest.raises(ConfigValidationError) as excinfo:
            validate_anisotropic_variables(incomplete_anisotropic)
        
        assert "missing required components" in str(excinfo.value)
        assert "c14" in str(excinfo.value)  # Should mention missing components

    def test_anisotropic_config_validation(self):
        """Test config validation for anisotropic models."""
        # Create a complete anisotropic config
        anisotropic_config = {
            'data_path': '/tmp/test.nc',  # Will be mocked
            'dx': 10000,
            'dy': 10000,
            'dz': 1000,
            'z_min': -100,
            'z_max': 0,
            'variable': [
                'c11', 'c12', 'c13', 'c14', 'c15', 'c16',
                'c22', 'c23', 'c24', 'c25', 'c26',
                'c33', 'c34', 'c35', 'c36',
                'c44', 'c45', 'c46',
                'c55', 'c56',
                'c66', 'rho'
            ],
            'utm_zone': 4,
            'utm_hemisphere': 'N',
            'extent': [441718, 941718, 6484812, 6884812],
            'plot_color_by': 'c11'
        }
        
        # Mock file existence check
        with patch('os.path.isfile', return_value=True):
            assert validate_config(anisotropic_config) == True

    def test_anisotropic_config_incomplete(self):
        """Test config validation fails for incomplete anisotropic models."""
        incomplete_config = {
            'data_path': '/tmp/test.nc',
            'dx': 10000,
            'dy': 10000,
            'dz': 1000,
            'z_min': -100,
            'z_max': 0,
            'variable': ['c11', 'c12', 'rho'],  # Incomplete anisotropic set
            'utm_zone': 4,
            'utm_hemisphere': 'N',
            'extent': [441718, 941718, 6484812, 6884812]
        }
        
        with patch('os.path.isfile', return_value=True):
            with pytest.raises(ConfigValidationError) as excinfo:
                validate_config(incomplete_config)
            
            assert "missing required components" in str(excinfo.value)

    def test_anisotropic_plot_color_by(self):
        """Test plot_color_by validation for anisotropic models."""
        anisotropic_config = {
            'data_path': '/tmp/test.nc',
            'dx': 10000,
            'dy': 10000,
            'dz': 1000,
            'z_min': -100,
            'z_max': 0,
            'variable': [
                'c11', 'c12', 'c13', 'c14', 'c15', 'c16',
                'c22', 'c23', 'c24', 'c25', 'c26',
                'c33', 'c34', 'c35', 'c36',
                'c44', 'c45', 'c46',
                'c55', 'c56',
                'c66', 'rho'
            ],
            'utm_zone': 4,
            'utm_hemisphere': 'N',
            'extent': [441718, 941718, 6484812, 6884812]
        }
        
        # Test valid anisotropic plot variables
        with patch('os.path.isfile', return_value=True):
            for var in ['c11', 'c44', 'c66', 'rho']:
                anisotropic_config['plot_color_by'] = var
                assert validate_config(anisotropic_config) == True
        
        # Test invalid plot variable for anisotropic model
        with patch('os.path.isfile', return_value=True):
            anisotropic_config['plot_color_by'] = 'vp'  # Not in variable list
            with pytest.raises(ConfigValidationError) as excinfo:
                validate_config(anisotropic_config)
            assert "must be one of the variables" in str(excinfo.value)

    def test_isotropic_plot_color_by(self):
        """Test that isotropic models still work with original plot_color_by validation."""
        isotropic_config = {
            'data_path': '/tmp/test.nc',
            'dx': 5000,
            'dy': 5000,
            'dz': 5000,
            'z_min': -40,
            'z_max': 0,
            'variable': ['vp', 'vs'],
            'utm_zone': 36,
            'utm_hemisphere': 'N',
            'extent': [441718, 941718, 6484812, 6884812],
            'plot_color_by': 'vp'
        }
        
        with patch('os.path.isfile', return_value=True):
            assert validate_config(isotropic_config) == True
        
        # Test invalid plot variable for isotropic model
        with patch('os.path.isfile', return_value=True):
            isotropic_config['plot_color_by'] = 'c11'  # Not valid for isotropic
            with pytest.raises(ConfigValidationError) as excinfo:
                validate_config(isotropic_config)
            assert "must be 'vp', 'vs', or 'rho' for isotropic models" in str(excinfo.value)

    def test_anisotropic_template_generation(self):
        """Test that anisotropic config template can be generated."""
        from specfem_tomo_helper.cli import main
        import sys
        from io import StringIO
        
        # Test creating anisotropic template
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, 'test_anisotropic.yaml')
            
            # Mock sys.argv for the test
            test_args = ['cli.py', '--create-config', '--anisotropic', '--output', output_file]
            
            with patch.object(sys, 'argv', test_args):
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    try:
                        main()
                    except SystemExit:
                        pass  # Expected when --create-config is used
            
            # Check that file was created
            assert os.path.exists(output_file)
            
            # Check content of generated file
            with open(output_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Verify it contains all anisotropic components
            variables = config['variable']
            assert len(variables) == 22  # 21 Cij + rho
            assert 'c11' in variables
            assert 'c66' in variables
            assert 'rho' in variables
            assert config['fill_nan'] == 'vertical'
            assert config['float_format'] == '%.8f'
