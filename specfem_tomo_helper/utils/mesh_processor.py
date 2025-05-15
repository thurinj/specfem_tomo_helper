import numpy as np
import pandas as pd
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sklearn.cluster import KMeans
import importlib.resources


def _pkg_template_path(filename="mesh_template.j2",
                       subdir="templates",
                       package="specfem_tomo_helper"):
    """Return <installed-pkg>/<subdir>/<filename> as a Path()."""
    pkg_dir = Path(importlib.import_module(package).__file__).parent
    path = pkg_dir / subdir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path

class MeshConfigError(RuntimeError):
    pass

class MeshProcessor:
    """
    Build and validate SPECFEM3D mesh parameters, then write a Par_file.
    """

    # --------------------------------------------------------------------- #
    #  Constructor & basic utilities                                        #
    # --------------------------------------------------------------------- #
    def __init__(self, interpolated_tomography, src_half_duration=0.5,
                 save_dir="./mesh_output"):
        """
        Parameters
        ----------
        interpolated_tomography : ndarray (N, ≥5)
            Columns : x, y, z[km], Vp, Vs, ...
        src_half_duration : float
            Half-duration (s) of the source used to size the mesh.
        save_dir : str
            Directory for interface txt files and other artefacts.
        """
        self.model = np.asarray(interpolated_tomography)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.min_vs = self.model[:, 4].min()
        self.src_half_duration = src_half_duration
        self.desired_dx, msg = self.estimate_element_size()
        print(msg)

        self.selected_config: dict | None = None      # from suggest_horizontal_configs
        self._vertical_cache: dict | None = None      # from generate_dynamic_mesh_config

    # ------------------------------------------------------------------ #
    #  Mesh-sizing helpers                                               #
    # ------------------------------------------------------------------ #
    def estimate_element_size(self, points_per_wavelength=5, gll_order=5):
        """
        Returns
        -------
        h_max : float   (metres)
        msg   : str     pretty message
        """
        f_max = 1.0 / (2.0 * self.src_half_duration)
        lambda_min = (self.min_vs * 1_000.0) / f_max
        h_max = lambda_min / points_per_wavelength * (gll_order - 1)

        msg = (f"Target dx ≈ {h_max:,.1f} m  "
               f"(Vs={self.min_vs:.4f} km/s,  h_dur={self.src_half_duration}s)")
        return h_max, msg

    # ------------------------------------------------------------------ #
    #  Horizontal search                                                 #
    # ------------------------------------------------------------------ #
    def suggest_horizontal_configs(self, dx_target_km=None, max_cpu=64,
                                   alpha=1.0, beta=1.0, mode="best"):
        """
        Populates self.selected_config and returns it.
        """
        # ---- *unchanged enumerator* ----------------------------------- #
        if dx_target_km is None:
            dx_target_km = self.desired_dx / 1000.0

        dx_target = dx_target_km * 1_000.0
        x0, x1 = self.model[:, 0].min(), self.model[:, 0].max()
        y0, y1 = self.model[:, 1].min(), self.model[:, 1].max()
        Lx, Ly = x1 - x0, y1 - y0
        domain_ratio = Lx / Ly

        candidates = []
        for nproc_xi in range(1, max_cpu + 1):
            for nproc_eta in range(1, max_cpu // nproc_xi + 1):
                tot = nproc_xi * nproc_eta
                for kx in range(1, 100):
                    nex_xi = 8 * nproc_xi * kx
                    dx = Lx / nex_xi
                    if not 0.5 * dx_target <= dx <= 2.0 * dx_target:
                        if dx < 0.5 * dx_target:
                            break
                        continue
                    for ky in range(1, 100):
                        nex_eta = 8 * nproc_eta * ky
                        dy = Ly / nex_eta
                        if not 0.5 * dx_target <= dy <= 2.0 * dx_target:
                            if dy < 0.5 * dx_target:
                                break
                            continue
                        res_ratio = max(dx, dy) / min(dx, dy)
                        proc_ratio = nproc_xi / nproc_eta
                        shape_ratio = max(domain_ratio, proc_ratio) / \
                                      min(domain_ratio, proc_ratio)
                        delta = 0.5 * (abs(dx - dx_target) + abs(dy - dx_target)) / dx_target
                        candidates.append((tot, nproc_xi, nproc_eta, nex_xi, nex_eta,
                                           dx, dy, res_ratio, shape_ratio, delta))

        df = pd.DataFrame(candidates, columns=[
            "total_cpu", "nproc_xi", "nproc_eta", "nex_xi", "nex_eta",
            "dx", "dy", "res_ratio", "load_ratio", "delta"
        ])
        df = df[df["total_cpu"] >= 0.9 * df["total_cpu"].max()]

        # cost
        scale = {
            "delta": np.percentile(df["delta"], 90),
            "square": np.percentile(df["res_ratio"] - 1, 90),
            "shape": np.percentile(df["load_ratio"] - 1, 90)
        }
        df["cost"] = (beta * (df["delta"] / scale["delta"])
                      + (df["res_ratio"] - 1) / scale["square"]
                      + alpha * (df["load_ratio"] - 1) / scale["shape"])
        df = df.sort_values("cost").reset_index(drop=True)

        if mode == "best":
            self.selected_config = df.iloc[0].to_dict()
        elif mode == "choice":
            print(df.head(10)[["total_cpu", "nproc_xi", "nproc_eta", "nex_xi", "nex_eta", "dx", "dy", "res_ratio"]])
            try:
                sel = int(input("Index ➜ "))
                if sel < 0 or sel >= len(df):
                    raise ValueError("Invalid index")
                self.selected_config = df.iloc[sel].to_dict()
            except (ValueError, IndexError):
                print("Invalid input. Falling back to the best configuration.")
                self.selected_config = df.iloc[0].to_dict()
        else:
            raise ValueError("mode must be 'best' or 'choice'")
        return self.selected_config

    def horizontal_dict(self):
        if self.selected_config is None:
            raise MeshConfigError("Run suggest_horizontal_configs() first")
        c = self.selected_config
        return dict(NPROC_XI=int(c["nproc_xi"]), NPROC_ETA=int(c["nproc_eta"]),
                    NEX_XI=int(c["nex_xi"]), NEX_ETA=int(c["nex_eta"]))

    # ------------------------------------------------------------------ #
    #  Vertical doubling + regions                                       #
    # ------------------------------------------------------------------ #
    def _suggest_vertical_resolution_with_doubling(self, dz_target_km=1.0, total_depth_km=10.0, doubling_layers=None):
        """
        Suggests the optimal number of vertical elements considering multiple doubling layers.

        Parameters:
        - dz_target_km: Target element size in the vertical direction (in km).
        - total_depth_km: Total depth of the model (in km).
        - doubling_layers: List of depths (in km) where doubling occurs, sorted from shallowest to deepest.

        Returns:
        - total_elements: Total number of vertical elements.
        - element_distribution: List of element counts for each layer.
        """
        if not doubling_layers:
            # No doubling, calculate elements normally
            total_elements = int(np.ceil(total_depth_km / dz_target_km))
            return total_elements, [total_elements]

        # Ensure doubling layers are sorted
        doubling_layers = sorted(doubling_layers)

        element_distribution = []
        previous_depth = 0.0
        current_dz = dz_target_km

        for depth in doubling_layers + [total_depth_km]:
            layer_thickness = depth - previous_depth
            elements_in_layer = int(np.ceil(layer_thickness / current_dz))
            element_distribution.append(elements_in_layer)
            previous_depth = depth
            current_dz *= 2  # Double the element size for the next layer

        total_elements = sum(element_distribution)
        return total_elements, element_distribution[::-1]

    def generate_dynamic_mesh_config(self, dz_target_km=1.0,
                                     max_depth=10.0,
                                     doubling_layers=None):
        doubling_layers = doubling_layers or []
        
        if doubling_layers:
            doubling_layers = sorted([abs(x) / 1000 for x in doubling_layers])

        nz_tot, elem_dist = self._suggest_vertical_resolution_with_doubling(
            dz_target_km, max_depth, doubling_layers)

        ndoublings = len(doubling_layers)
        nz_doublings = np.cumsum(elem_dist[:-1]).tolist()

        # regions
        regions = []
        nz0 = 1
        for i, n in enumerate(elem_dist, start=1):
            nz1 = nz0 + n - 1
            regions.append((1, self.horizontal_dict()["NEX_XI"],
                            1, self.horizontal_dict()["NEX_ETA"],
                            nz0, nz1, -1))
            nz0 = nz1 + 1

        self._vertical_cache = dict(
            USE_REGULAR_MESH=False,
            NDOUBLINGS=ndoublings,
            NZ_DOUBLINGS=nz_doublings,
            NREGIONS=len(regions),
            REGIONS=regions,
            TOTAL_NZ=nz_tot
        )
        return self._vertical_cache

    # ------------------------------------------------------------------ #
    #  Validation                                                        #
    # ------------------------------------------------------------------ #
    def _validate(self):
        if self._vertical_cache is None or self.selected_config is None:
            raise MeshConfigError("Run horizontal and vertical generators first")

        h = self.horizontal_dict()
        v = self._vertical_cache

        if h["NEX_XI"] % (8 * h["NPROC_XI"]) != 0:
            raise MeshConfigError("NEX_XI not multiple of 8*NPROC_XI")
        if h["NEX_ETA"] % (8 * h["NPROC_ETA"]) != 0:
            raise MeshConfigError("NEX_ETA not multiple of 8*NPROC_ETA")
        if v["NDOUBLINGS"] != len(v["NZ_DOUBLINGS"]):
            raise MeshConfigError("Inconsistent doubling count")

    # ------------------------------------------------------------------ #
    #  Flatten for template                                              #
    # ------------------------------------------------------------------ #
    def parfile_dict(self, **latlon_kwargs):
        """
        Parameters
        ----------
        latlon_kwargs : keys that the caller must provide
            LATITUDE_MIN, LATITUDE_MAX, LONGITUDE_MIN, LONGITUDE_MAX,
            DEPTH_BLOCK_KM, UTM_PROJECTION_ZONE, SUPPRESS_UTM_PROJECTION,
            INTERFACES_FILE, TOMO_FILENAME
        """
        self._validate()
        out = {}
        out.update(self.horizontal_dict())
        out.update(self._vertical_cache)
        out.update(latlon_kwargs)

        # unpack regions to dict list
        reg_dicts = []
        for r in out["REGIONS"]:
            reg_dicts.append(dict(xi0=r[0], xi1=r[1], eta0=r[2], eta1=r[3],
                                  z0=r[4], z1=r[5], mat=r[6]))
        out["regions"] = reg_dicts
        out["nz_doublings"] = out["NZ_DOUBLINGS"]
        return out

    # ------------------------------------------------------------------ #
    #  Render Par_file                                                   #
    # ------------------------------------------------------------------ #
    def to_parfile(self, template_path: str | Path | None = None,
                   output_path="Par_file", **latlon):
        """
        Render the Par_file.

        Parameters
        ----------
        template_path : str or Path or None
            • None  → use the package’s default template  
            • str/Path → path to a custom Jinja2 template
        output_path  : where to write the Par_file
        **latlon     : the extra keyword args required by the template
        """
        if template_path is None:
            template_path = _pkg_template_path()    # ← automatic default

        env = Environment(loader=FileSystemLoader(Path(template_path).parent),
                          autoescape=select_autoescape([]),
                          trim_blocks=True, lstrip_blocks=True)
        tmpl = env.get_template(Path(template_path).name)
        txt = tmpl.render(**self.parfile_dict(**latlon))
        Path(output_path).write_text(txt)
        print(f"✅  {output_path} written")
        return output_path

    # ------------------------------------------------------------------ #
    #  Vs/Vp-based automatic layer break                                 #
    # ------------------------------------------------------------------ #
    def detect_doubling(self, n_clusters=3):
        z_vals = np.unique(self.model[:, 2])        # depth km, shallow→deep
        vp_vs = np.column_stack([
            [self.model[np.isclose(self.model[:, 2], z), 3].mean() for z in z_vals],
            [self.model[np.isclose(self.model[:, 2], z), 4].mean() for z in z_vals]
        ])
        kmeans = KMeans(n_clusters, n_init=10, random_state=0)
        labels = kmeans.fit_predict(vp_vs)

        transitions = z_vals[1:][labels[1:] != labels[:-1]]
        return transitions.tolist()
