import sys
import os
import argparse
from ruamel.yaml import YAML
import numpy as np
import matplotlib.pyplot as plt
from specfem_tomo_helper.projection import vp2rho, vp2vs, vs2vp, define_utm_projection
from specfem_tomo_helper.fileimport import Nc_model
from specfem_tomo_helper.utils import maptool, trilinear_interpolator, write_tomo_file, TopographyProcessor, MeshProcessor
from specfem_tomo_helper.utils.config_utils import validate_config, ConfigValidationError, auto_detect_utm_from_extent, is_geographic_extent

def main():
    parser = argparse.ArgumentParser(description='Run tomography workflow from config file or create a template config.')
    parser.add_argument('--config', '-c', type=str, help='Path to YAML config file')
    parser.add_argument('--create-config', action='store_true', help='Create a template config YAML file and exit')
    parser.add_argument('--output', '-o', type=str, default='config_example.yaml', help='Output path for the template config file (used with --create-config)')
    args = parser.parse_args()

    if args.create_config:
        import shutil
        template_path = os.path.join(os.path.dirname(__file__), 'templates/config_example_template.yaml')
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
    dx = config['dx']
    dy = config['dy']
    dz = config['dz']
    z_min = config['z_min']
    z_max = config['z_max']
    variable = config['variable']

    # Load the netCDF model and variables
    nc_model = Nc_model(path)
    lon, lat, depth = nc_model.load_coordinates()
    var = nc_model.load_variable(variable, fill_nan='vertical')

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
            print("No extent or UTM zone/hemisphere provided. Attempting to auto-detect from NetCDF model coordinates...")
            try:
                min_lon, max_lon = float(np.nanmin(lon)), float(np.nanmax(lon))
                min_lat, max_lat = float(np.nanmin(lat)), float(np.nanmax(lat))
                data_extent = [min_lon, max_lon, min_lat, max_lat]
                print(f"Auto-detecting UTM zone and hemisphere from model lon/lat extent: {data_extent}")
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
                    print("Auto-detecting UTM zone and hemisphere from GUI-selected extent...")
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

    def canonical_var_name(name):
        name = name.lower()
        if 'vs' in name:
            return 'vs'
        if 'vp' in name:
            return 'vp'
        if 'rho' in name or 'density' in name:
            return 'rho'
        return name

    var_arrays = []
    var_names = []
    for v in variables:
        arr = nc_model.load_variable(v, fill_nan=config.get('fill_nan', 'vertical'))
        var_arrays.append(arr)
        var_names.append(canonical_var_name(v))

    interpolated = interpolator.interpolate(var_arrays)
    coordinates = interpolated[:, 0:3]
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
    write_tomo_file(tomo, interpolator, tomo_output_dir)

    mesh_output_dir = config.get('mesh_output_dir', './mesh_output')
    if config.get('generate_mesh', False):
        mesh = MeshProcessor(
            interpolated_tomography=tomo,
            projection=gui_parameters.projection,
            save_dir=mesh_output_dir
        )
        mesh.suggest_horizontal_configs(dx_target_km=config['dx_target_km'], max_cpu=config['max_cpu'], mode='choice', n_doublings=len(config['doubling_layers']))
        # Convert doubling_layers from km (positive down) to negative for internal use
        neg_doubling_layers_km = [-dl for dl in config['doubling_layers']]
        doubling_layers_m = [dl * 1000.0 for dl in neg_doubling_layers_km]
        mesh.generate_dynamic_mesh_config(dz_target_km=config['dz_target_km'], max_depth=config['max_depth'], doubling_layers=doubling_layers_m)
    if config.get('generate_topography', False):
        smoothing_sigma = config.get('smoothing_sigma', 0)
        mesh_arg = mesh if config.get('generate_mesh', False) else None
        # Convert doubling_layers from km (positive down) to negative for TopographyProcessor if mesh is not used
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
    if config.get('generate_mesh', False):
        mesh.write_parfile_easy(output_dir=mesh_output_dir)

    if config.get('plot_outer_shell', False):
        x_bounds = {np.min(tomo[:, 0]), np.max(tomo[:, 0])}
        y_bounds = {np.min(tomo[:, 1]), np.max(tomo[:, 1])}
        z_bounds = {np.min(tomo[:, 2]), np.max(tomo[:, 2])}
        outer_shell = tomo[np.isin(tomo[:, 0], list(x_bounds)) |
                           np.isin(tomo[:, 1], list(y_bounds)) |
                           np.isin(tomo[:, 2], list(z_bounds))]
        color_idx = {'vp': 3, 'vs': 4, 'rho': 5}[config.get('plot_color_by', 'vp')]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(outer_shell[:, 0], outer_shell[:, 1], outer_shell[:, 2], c=outer_shell[:, color_idx])
        plt.show()

if __name__ == '__main__':
    main()
