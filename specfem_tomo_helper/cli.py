import sys
import os
import argparse
from ruamel.yaml import YAML
import numpy as np
import matplotlib.pyplot as plt
from specfem_tomo_helper.projection import vp2rho, vp2vs, vs2vp, define_utm_projection
from specfem_tomo_helper.fileimport import Nc_model
from specfem_tomo_helper.utils import maptool, trilinear_interpolator, write_tomo_file, write_anisotropic_tomo_file, TopographyProcessor, MeshProcessor
from specfem_tomo_helper.utils.change_of_basis import transform
from specfem_tomo_helper.utils.config_utils import validate_config, ConfigValidationError, auto_detect_utm_from_extent, is_geographic_extent
from specfem_tomo_helper.utils.download import download_if_missing
from specfem_tomo_helper import __version__

# Global verbose flag
VERBOSE = False

def verbose_print(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if VERBOSE:
        print(*args, **kwargs)

def is_anisotropic_model(variables):
    """
    Determine if the model is anisotropic based on the variable list.
    """
    if isinstance(variables, str):
        variables = [variables]
    anisotropic_components = [
        'c11', 'c12', 'c13', 'c14', 'c15', 'c16',
        'c22', 'c23', 'c24', 'c25', 'c26',
        'c33', 'c34', 'c35', 'c36',
        'c44', 'c45', 'c46',
        'c55', 'c56',
        'c66'
    ]
    var_lower = [v.lower() for v in variables if isinstance(v, str)]
    return any(comp in var_lower for comp in anisotropic_components)

def validate_anisotropic_variables(variables):
    """
    Validate that all required anisotropic components are present.
    """
    if isinstance(variables, str):
        variables = [variables]
    required_components = [
        'c11', 'c12', 'c13', 'c14', 'c15', 'c16',
        'c22', 'c23', 'c24', 'c25', 'c26',
        'c33', 'c34', 'c35', 'c36',
        'c44', 'c45', 'c46',
        'c55', 'c56',
        'c66', 'rho'
    ]
    var_lower = [v.lower() for v in variables if isinstance(v, str)]
    missing = [comp for comp in required_components if comp not in var_lower]
    if missing:
        raise ConfigValidationError(
            f"Anisotropic model detected but missing required components: {missing}. "
            f"For anisotropic models, all 21 stiffness tensor components (c11-c66) plus density (rho) are required."
        )
    return True

def main():
    global VERBOSE
    
    parser = argparse.ArgumentParser(
        description=(
            'Run specfem_tomo_helper workflow from config file or create a template config.\n\n'
            'Workflow summary:\n'
            '  - Validates the provided YAML config file.\n'
            '  - Loads NetCDF tomography model and extracts coordinates.\n'
            '  - Handles UTM projection and extent (auto-detects if needed).\n'
            '  - Interpolates model variables onto a regular grid.\n'
            '  - Writes tomography files (isotropic or anisotropic).\n'
            '  - Optionally generates mesh and topography files.\n'
            '  - Optionally visualizes the outer shell of the model.\n\n'
            'Use --create-config to generate a commented template config file.'
        ),
        epilog=(
            'Config file arguments (YAML):\n'
            '  data_path:         Path to input NetCDF model file.\n'
            '  variable:          List of variables to extract (e.g. vp, vs, rho, or c11..c66, rho for anisotropic).\n'
            '\nModel grid and extent parameters:\n'
            '  extent:            [xmin, xmax, ymin, ymax] in UTM coordinates (optional, can use GUI).\n'
            '  utm_zone, utm_hemisphere: UTM zone and hemisphere (required if extent is set; auto-detected if not provided).\n'
            '  dx, dy, dz:        Grid spacing for the model (meters).\n'
            '  z_min, z_max:      Depth range for the model (meters).\n'
            '\nMesh resolution parameters (for mesh generation):\n'
            '  dx_target_km, dz_target_km: Target mesh spacing (kilometers).\n'
            '  doubling_layers:   List of depths (km) for mesh doubling (refinement layers).\n'
            '  max_depth:         Maximum mesh depth (meters).\n'
            '  max_cpu:           Max CPUs for mesh partitioning.\n'
            '\nTopography processing parameters:\n'
            '  smoothing_sigma:   Gaussian smoothing for topography (number or "auto").\n'
            '  slope_thresholds:  List of slope thresholds for topography filtering.\n'
            '      If smoothing_sigma is "auto", topography is smoothed until all slopes are below the minimum slope_threshold.\n'
            '      Otherwise, smoothing_sigma sets the smoothing before filtering by slope_thresholds.\n'
            '  filter_topography: true/false, whether to filter topography by slope.\n'
            '\nOutput and other parameters:\n'
            '  tomo_output_dir:   Output directory for tomography files.\n'
            '  mesh_output_dir:   Output directory for mesh files.\n'
            '  topography_output_dir: Output directory for topography files.\n'
            '  generate_mesh:     true/false, whether to generate mesh files.\n'
            '  generate_topography: true/false, whether to generate topography files.\n'
            '  plot_outer_shell:  true/false, plot outer shell of the model.\n'
            '  plot_color_by:     Which variable to color by in plot (vp, vs, rho).\n'
            '\nFor more details, please have a look at the template config file generated with: --create-config.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', '-c', type=str, help='Path to YAML config file')
    parser.add_argument('--create-config', action='store_true', help='Create a template config YAML file and exit')
    parser.add_argument('--output', '-o', type=str, default='config_example.yaml', help='Output path for the template config file (used with --create-config)')
    parser.add_argument('--anisotropic', action='store_true', help='Create a template config YAML file for anisotropic tomography model (used with --create-config)')
    parser.add_argument('--version', action='version', version=f'specfem-tomo-helper {__version__}')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    # Set global verbose flag
    VERBOSE = args.verbose

    if args.create_config:
        import shutil
        if args.anisotropic:
            template_name = 'config_anisotropic_template.yaml'
        else:
            template_name = 'config_example_template.yaml'
        template_path = os.path.join(os.path.dirname(__file__), f'templates/{template_name}')
        output_path = args.output
        shutil.copyfile(template_path, output_path)
        print(f"Template config written to {output_path}")
        return

    config_path = args.config
    if not config_path:
        print('Error: Please provide a config file with --config <path>.')
        sys.exit(1)
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path, 'r') as f:
        config = yaml.load(f)
    try:
        validate_config(config)
    except ConfigValidationError as e:
        print(f"Config validation error: {e}")
        sys.exit(1)

    # Model and grid parameters
    path = config['data_path']
    model_url = config.get('model_url', None)
    if not os.path.isfile(path):
        if model_url:
            download_if_missing(path, model_url)
        else:
            print(f"Error: Model file '{path}' not found and no model_url provided in config.")
            sys.exit(1)
    dx = config['dx']
    dy = config['dy']
    dz = config['dz']
    z_min = config['z_min']
    z_max = config['z_max']
    variable = config['variable']

    # Load the netCDF model and variables
    verbose_print(f"Loading NetCDF model from: {path}")
    nc_model = Nc_model(path)
    lon, lat, depth = nc_model.load_coordinates()
    verbose_print(f"Loaded coordinates from NetCDF model")

    # UTM projection setup and extent validation
    utm_zone = config.get('utm_zone')
    utm_hemisphere = config.get('utm_hemisphere')
    extent = config.get('extent')

    if extent is not None:
        if is_geographic_extent(extent):
            print("Error: 'extent' must be specified in UTM coordinates, not geographic (lat/lon). Please provide UTM coordinates and specify utm_zone and utm_hemisphere.")
            sys.exit(1)
        if utm_zone is None or utm_hemisphere is None:
            print("Error: When 'extent' is provided, both 'utm_zone' and 'utm_hemisphere' must also be specified in the config.")
            sys.exit(1)
    else:
        if utm_zone is None or utm_hemisphere is None:
            verbose_print("No extent or UTM zone/hemisphere provided. Attempting to auto-detect from NetCDF model coordinates...")
            try:
                min_lon, max_lon = float(np.nanmin(lon)), float(np.nanmax(lon))
                min_lat, max_lat = float(np.nanmin(lat)), float(np.nanmax(lat))
                data_extent = [min_lon, max_lon, min_lat, max_lat]
                verbose_print(f"Auto-detecting UTM zone and hemisphere from model lon/lat extent: {data_extent}")
                detected_zone, detected_hemisphere = auto_detect_utm_from_extent(data_extent)
                utm_zone = detected_zone
                utm_hemisphere = detected_hemisphere
                config['utm_zone'] = utm_zone
                config['utm_hemisphere'] = utm_hemisphere
                print(f"Detected UTM zone: {utm_zone}, hemisphere: {utm_hemisphere}")
            except Exception as e:
                print(f"Error: Could not auto-detect UTM from NetCDF model coordinates: {e}")
                print("Please specify utm_zone and utm_hemisphere manually in your config file.")
                sys.exit(1)

    myProj = None
    if utm_zone is not None and utm_hemisphere is not None:
        myProj = define_utm_projection(utm_zone, utm_hemisphere)

    if config['extent'] is None or config.get('use_gui', False):
        gui_parameters = maptool(nc_model, myProj)
        extent = gui_parameters.extent
        config['extent'] = extent.tolist() if hasattr(extent, 'tolist') else list(extent)
        if hasattr(gui_parameters, 'projection') and hasattr(gui_parameters.projection, 'crs'):
            utm_zone = getattr(gui_parameters.projection.crs, 'utm_zone', None)
            if utm_zone is not None:
                if isinstance(utm_zone, str):
                    utm_zone_numeric = ''.join(filter(str.isdigit, utm_zone))
                    if utm_zone_numeric:
                        config['utm_zone'] = int(utm_zone_numeric)
                else:
                    config['utm_zone'] = int(utm_zone)
        if myProj is None:
            if config.get('utm_zone') is None or config.get('utm_hemisphere') is None:
                if is_geographic_extent(config['extent']):
                    verbose_print("Auto-detecting UTM zone and hemisphere from GUI-selected extent...")
                    detected_zone, detected_hemisphere = auto_detect_utm_from_extent(config['extent'])
                    config['utm_zone'] = detected_zone
                    config['utm_hemisphere'] = detected_hemisphere
                    print(f"Detected UTM zone: {detected_zone}, hemisphere: {detected_hemisphere}")
                else:
                    print("Error: Cannot auto-detect UTM from non-geographic extent. Please specify utm_zone and utm_hemisphere manually.")
                    sys.exit(1)
            myProj = define_utm_projection(config['utm_zone'], config['utm_hemisphere'])
        with open(args.config, 'w') as f:
            yaml.dump(config, f)
    else:
        extent = config['extent']
        class Dummy:
            pass
        gui_parameters = Dummy()
        gui_parameters.extent = extent
        gui_parameters.projection = myProj

    interpolator = trilinear_interpolator(nc_model, gui_parameters.projection)
    interpolator.interpolation_parameters(extent[0], extent[1], dx,
                                          extent[2], extent[3], dy,
                                          z_min, z_max, dz)

    variables = config['variable']
    if isinstance(variables, str):
        variables = [variables]

    is_anisotropic = is_anisotropic_model(variables)
    var_arrays = []
    var_names = []
    if is_anisotropic:
        verbose_print("Detected anisotropic model - processing full stiffness tensor")
        validate_anisotropic_variables(variables)
        for v in variables:
            arr = nc_model.load_variable(v, fill_nan=config.get('fill_nan', 'vertical'))
            var_arrays.append(arr)
            var_names.append(v.lower())
    else:
        def canonical_var_name(name):
            name = name.lower()
            if 'vs' in name:
                return 'vs'
            if 'vp' in name:
                return 'vp'
            if 'rho' in name or 'density' in name:
                return 'rho'
            return name
        for v in variables:
            arr = nc_model.load_variable(v, fill_nan=config.get('fill_nan', 'vertical'))
            var_arrays.append(arr)
            var_names.append(canonical_var_name(v))

    interpolated = interpolator.interpolate(var_arrays)
    coordinates = interpolated[:, 0:3]
    if is_anisotropic:
        param = interpolated[:, 3:]
        # apply coordinate transformation if needed
        basis = config['basis']
        if basis is not None:
            if basis.lower() != "east_north_up":
                verbose_print(
                    f"Transforming coordinates from basis: '{basis}' to"
                    f" basis: 'east_north_up'"
                )
                for i, c_vec in enumerate(param[:, :-1]):
                    param[i, :-1] = transform(c_vec, basis)
        tomo = np.hstack((coordinates, param))
    else:
        param_dict = {name: interpolated[:, 3 + i] for i, name in enumerate(var_names)}
        if 'vp' not in param_dict and 'vs' in param_dict:
            param_dict['vp'] = vs2vp(param_dict['vs'])
        if 'vs' not in param_dict and 'vp' in param_dict:
            param_dict['vs'] = vp2vs(param_dict['vp'])
        if 'rho' not in param_dict and 'vp' in param_dict:
            param_dict['rho'] = vp2rho(param_dict['vp'])
        output_params = []
        for key in ['vp', 'vs', 'rho']:
            if key in param_dict:
                output_params.append(param_dict[key])
            else:
                output_params.append(np.full(coordinates.shape[0], np.nan))
        param = np.vstack(output_params).T
        tomo = np.hstack((coordinates, param))

    tomo_output_dir = config.get('tomo_output_dir', './')
    os.makedirs(tomo_output_dir, exist_ok=True)
    verbose_print(f"Writing tomography files to: {tomo_output_dir}")
    if is_anisotropic:
        float_format = config.get('float_format', '%.8f')
        write_anisotropic_tomo_file(tomo, interpolator, tomo_output_dir, float_format=float_format)
        print(f"Anisotropic tomography files written to: {tomo_output_dir}")
        print("WARNING: Cij values in the output file have been converted to SI units (Pa)!")
    else:
        write_tomo_file(tomo, interpolator, tomo_output_dir)
        print(f"Tomography files written to: {tomo_output_dir}")

    # --- Mesh Generation (identical for both cases) ---
    mesh_output_dir = config.get('mesh_output_dir', './mesh_output')
    if config.get('generate_mesh', False):
        verbose_print(f"Generating mesh files in: {mesh_output_dir}")
        mesh = MeshProcessor(
            interpolated_tomography=tomo,
            projection=gui_parameters.projection,
            save_dir=mesh_output_dir
        )
        mesh.suggest_horizontal_configs(dx_target_km=config['dx_target_km'], max_cpu=config['max_cpu'], mode='choice', n_doublings=len(config['doubling_layers']))
        neg_doubling_layers_km = [-dl for dl in config['doubling_layers']]
        doubling_layers_m = [dl * 1000.0 for dl in neg_doubling_layers_km]
        mesh.generate_dynamic_mesh_config(dz_target_km=config['dz_target_km'], max_depth=config['max_depth'], doubling_layers=doubling_layers_m)
        print(f"Mesh configuration generated in: {mesh_output_dir}")
    if config.get('generate_topography', False):
        verbose_print(f"Generating topography files in: {config['topography_output_dir']}")
        smoothing_sigma = config.get('smoothing_sigma', 0)
        mesh_arg = mesh if config.get('generate_mesh', False) else None
        neg_doubling_layers_km = [-dl for dl in config['doubling_layers']] if 'doubling_layers' in config else None
        doubling_layers_m = [dl * 1000.0 for dl in neg_doubling_layers_km] if neg_doubling_layers_km is not None else None
        topo = TopographyProcessor(
            interpolator,
            gui_parameters.projection,
            save_dir=config['topography_output_dir'],
            smoothing_sigma=0 if smoothing_sigma == 'auto' else (smoothing_sigma or 0),
            mesh_processor=mesh_arg,
            doubling_depth=doubling_layers_m if mesh_arg is None else None
        )
        if smoothing_sigma == 'auto':
            min_slope = min(config['slope_thresholds']) if config.get('slope_thresholds') else 10
            Xi, Yi, Zi = topo.interpolate_topography()
            Zi_smoothed = topo.auto_smooth_topography(Zi, Xi, Yi, min_slope)
            def interpolate_topography_override():
                return Xi, Yi, Zi_smoothed
            topo.interpolate_topography = interpolate_topography_override
        if config.get('filter_topography', True):
            topo.write_all_outputs(slope_thresholds=config['slope_thresholds'])
        else:
            if hasattr(topo, 'write_all_outputs_no_filter'):
                topo.write_all_outputs_no_filter()
            else:
                topo.write_all_outputs(slope_thresholds=None)
        print(f"Topography files generated in: {config['topography_output_dir']}")
    if config.get('generate_mesh', False):
        mesh.write_parfile_easy(output_dir=mesh_output_dir)
        print(f"Mesh parameter file written to: {mesh_output_dir}")

    # --- Visualization (identical for both cases, but color_idx logic only for isotropic) ---
    if config.get('plot_outer_shell', False):
        x_bounds = {np.min(tomo[:, 0]), np.max(tomo[:, 0])}
        y_bounds = {np.min(tomo[:, 1]), np.max(tomo[:, 1])}
        z_bounds = {np.min(tomo[:, 2]), np.max(tomo[:, 2])}
        outer_shell = tomo[np.isin(tomo[:, 0], list(x_bounds)) |
                           np.isin(tomo[:, 1], list(y_bounds)) |
                           np.isin(tomo[:, 2], list(z_bounds))]
        if is_anisotropic:
            color_idx = 3  # e.g., c11 or first parameter after coordinates
        else:
            color_idx = {'vp': 3, 'vs': 4, 'rho': 5}[config.get('plot_color_by', 'vp')]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(outer_shell[:, 0], outer_shell[:, 1], outer_shell[:, 2], c=outer_shell[:, color_idx])
        plt.show()

if __name__ == '__main__':
    main()
