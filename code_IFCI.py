# -*- coding: utf-8 -*-
"""
Green fragmentation (Connectivity/Aggregation/Structure)
Multi-scale: bub (full raster) + buffers (500/1000/5000/10000m) around venue points

+ NEW: 3-period summaries (pre5 / prep / post5) for indices:
  - CFI / AFI / SFI / IFCI
  - mean + error bars (SD/SE/95%CI)

+ NEW: City-level 3-period summaries
  - Aggregate venue-year indices -> city-year indices (mean across venues in same city_id)
  - Then compute period means & errorbars for each city_id

Outputs (in OUT_DIR):
- fragmentation_metrics_raw_scales.csv
- fragmentation_for_plotting_scales.csv
- fragmentation_period_venue_long.csv
- fragmentation_period_venue_wide.csv
- fragmentation_period_overall_long.csv
- fragmentation_period_city_long.csv
- fragmentation_period_city_wide.csv
- fragmentation_period_city_overall_long.csv
- fragmentation_report_scales.xlsx (multiple sheets incl. venue+city period sheets)

IMPORTANT FIX:
- bub buffer_m is set to 0 (not NaN), so bub included in all groupby/pivot.
"""

import re
import math
import warnings
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.mask import mask as rio_mask

from skimage.measure import label, regionprops
from scipy.spatial import cKDTree

import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import transform as shp_transform
from shapely.geometry import mapping as shp_mapping
from pyproj import CRS, Transformer

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ===================== USER SETTINGS =====================
ROOT = r"F:\nc_World Cup\01venue_bub\2002_japan_south_korea_venues_bub\DDD_control\05_data\gs"
OUT_DIR = r"F:\nc_World Cup\01venue_bub\Table data\fragmentation_outputs\2026\DID"

# Venue point file: supports shp/geojson/gpkg; CSV is also supported if longitude/latitude columns are provided.
POINTS_PATH = r"F:\nc_World Cup\01venue_bub\2026_usa_canda_mexico_venue_bub\06_DID\2026_DID_poi.shp"

# Candidate field names for point IDs and names, designed to be compatible with existing POI/venue tables.
POINT_ID_FIELD_CANDIDATES = ["venue_id", "VENUE_ID", "id", "ID", "poi_id", "POI_ID"]
POINT_NAME_FIELD_CANDIDATES = ["venue_name", "VENUE_NAME", "name", "NAME", "stadium", "STADIUM", "venue", "VENUE"]

# Candidate city field names, used for three-period outputs by city ID.
CITY_ID_FIELD_CANDIDATES = ["city_id", "CITY_ID", "cityid", "CITYID", "host_city_id", "HOSTCITYID",
                            "citycode", "CITYCODE", "CITY_CODE", "city_code"]
CITY_NAME_FIELD_CANDIDATES = ["city_name", "CITY_NAME", "city", "CITY", "host_city", "HOST_CITY",
                              "hostcity", "HOSTCITY", "City", "CITYNAME", "cityname"]

# If the point file is a CSV, one longitude column and one latitude column are required.
CSV_LON_CANDIDATES = ["lon", "longitude", "LON", "LONG", "x", "X"]
CSV_LAT_CANDIDATES = ["lat", "latitude", "LAT", "y", "Y"]
CSV_CRS = "EPSG:4326"  # Default CRS for CSV coordinates.

# Buffer radii in meters.
BUFFERS_M = [500, 1000, 5000, 10000]

# If raster already binary (green=1, non-green=0), keep THRESHOLD=None.
# If raster is continuous (e.g., NDVI), set THRESHOLD and green := value >= THRESHOLD.
THRESHOLD = None  # e.g., 0.3

# How to treat NODATA:
MASK_NODATA = False

# Whether to count the buffer boundary as a landscape boundary when calculating edge counts (FN).
COUNT_BUFFER_BOUNDARY_AS_EDGE = False

# ===== Tournament timing (for 3-phase summaries) =====
AWARD_YEAR = 2018   # Year when hosting rights were awarded (e.g., 1996 for the 2002 Korea/Japan World Cup).
HOST_YEAR  = 2026   # Host/opening year (e.g., 2002).
POST_INCLUDE_HOST_YEAR = False  # False: 2003–2007；True: 2002–2006
# ========================================================


# -------------------------
# Helpers: IO & parsing
# -------------------------
def find_rasters(root: str):
    root_path = Path(root)
    tif_list = sorted(list(root_path.rglob("*.tif")) + list(root_path.rglob("*.tiff")))
    return tif_list


def infer_venue_year(fp: Path):
    """
    Parse venue/year from filename like: 3_Korea_Green_2007.tif
    - year: last 4-digit at the end of stem
    - venue: stem without the trailing _YYYY
    - venue_id, venue_name: split leading integer if exists
    """
    stem = fp.stem
    m = re.search(r"(19|20)\d{2}$", stem)
    if not m:
        raise ValueError(f"Cannot parse year from filename: {fp.name}")

    year = int(m.group())
    venue = stem[:m.start()].rstrip("_")

    venue_id = None
    venue_name = venue
    m2 = re.match(r"(\d+)_+(.*)$", venue)
    if m2:
        venue_id = int(m2.group(1))
        venue_name = m2.group(2)

    return venue, venue_id, venue_name, year


def _pick_first_existing_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def load_venue_points(points_path: str) -> gpd.GeoDataFrame:
    """
    Load venue points from vector file or CSV.
    Ensures columns:
      - geometry (Point)
      - venue_id (Int64, may be NA)
      - venue_name (str)
      - city_id (str/int as str, may fallback)
      - city_name (str)
    """
    p = Path(points_path)
    if not p.exists():
        raise FileNotFoundError(f"POINTS_PATH not found: {points_path}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p, encoding="utf-8-sig")
        lon_col = _pick_first_existing_col(df.columns.tolist(), CSV_LON_CANDIDATES)
        lat_col = _pick_first_existing_col(df.columns.tolist(), CSV_LAT_CANDIDATES)
        if lon_col is None or lat_col is None:
            raise ValueError(f"CSV points must have lon/lat columns. Found columns: {list(df.columns)}")

        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=CSV_CRS
        )
    else:
        gdf = gpd.read_file(p)

    if gdf.empty:
        raise ValueError("Venue points file is empty.")

    if gdf.geometry.isna().any():
        gdf = gdf.dropna(subset=["geometry"]).copy()

    gdf = gdf[gdf.geometry.geom_type == "Point"].copy()
    if gdf.empty:
        raise ValueError("No Point geometries found in venue points file.")

    # venue id/name
    id_col = _pick_first_existing_col(gdf.columns.tolist(), POINT_ID_FIELD_CANDIDATES)
    name_col = _pick_first_existing_col(gdf.columns.tolist(), POINT_NAME_FIELD_CANDIDATES)

    if id_col is not None:
        gdf["venue_id"] = pd.to_numeric(gdf[id_col], errors="coerce").astype("Int64")
    else:
        gdf["venue_id"] = pd.Series([pd.NA] * len(gdf), dtype="Int64")

    if name_col is not None:
        gdf["venue_name"] = gdf[name_col].astype(str)
    else:
        gdf["venue_name"] = pd.Series([""] * len(gdf), dtype=str)

    # city id/name
    city_id_col = _pick_first_existing_col(gdf.columns.tolist(), CITY_ID_FIELD_CANDIDATES)
    city_name_col = _pick_first_existing_col(gdf.columns.tolist(), CITY_NAME_FIELD_CANDIDATES)

    if city_id_col is not None:
        # Convert city_id to string to avoid Int64/float type inconsistencies.
        gdf["city_id"] = gdf[city_id_col].astype(str)
    else:
        # Fallback: use venue_id as city_id to avoid dropping the record.
        gdf["city_id"] = gdf["venue_id"].astype("Int64").astype(str)

    if city_name_col is not None:
        gdf["city_name"] = gdf[city_name_col].astype(str)
    else:
        # Fallback: use venue_name.
        gdf["city_name"] = gdf["venue_name"].astype(str)

    return gdf


# -------------------------
# Buffer & raster read
# -------------------------
def build_buffer_geom_in_raster_crs(point_geom: Point, radius_m: float, raster_crs) -> Point:
    """
    Return buffer polygon geometry in raster CRS.
    - If raster CRS is projected: buffer directly (convert meters->CRS units if needed).
    - If raster CRS is geographic: local AEQD projection to buffer in meters, transform back.
    """
    if raster_crs is None:
        raise ValueError("Raster has no CRS; cannot create meter-based buffers.")

    crs = CRS.from_user_input(raster_crs)

    if crs.is_projected:
        unit_name = ""
        try:
            unit_name = (crs.axis_info[0].unit_name or "").lower()
        except Exception:
            unit_name = ""

        if "metre" in unit_name or "meter" in unit_name:
            radius_units = radius_m
        elif "foot" in unit_name or "feet" in unit_name:
            radius_units = radius_m / 0.3048
        else:
            radius_units = radius_m

        return point_geom.buffer(radius_units)

    lon, lat = float(point_geom.x), float(point_geom.y)
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")

    fwd = Transformer.from_crs(crs, aeqd, always_xy=True).transform
    inv = Transformer.from_crs(aeqd, crs, always_xy=True).transform

    pt_m = shp_transform(fwd, point_geom)
    buf_m = pt_m.buffer(radius_m)
    buf_back = shp_transform(inv, buf_m)
    return buf_back


def read_binary_green_with_geom(fp: Path, geom, threshold=None, mask_nodata=False):
    """Mask raster by geometry -> binary green (1/0), valid mask, resolution."""
    with rasterio.open(fp) as src:
        if src.crs is None:
            raise ValueError(f"Raster missing CRS (cannot buffer): {fp}")

        out, _ = rio_mask(src, [shp_mapping(geom)], crop=True, filled=False)
        arr = out[0]
        data = np.asarray(arr.data)
        outside_mask = np.asarray(arr.mask).astype(bool)

        nodata = src.nodata
        resx, resy = src.res

    nodata_mask = (data == nodata) if nodata is not None else np.zeros(data.shape, dtype=bool)
    inside = ~outside_mask
    valid = inside & (~nodata_mask) if mask_nodata else inside

    green = (data == 1).astype(np.uint8) if threshold is None else (data >= threshold).astype(np.uint8)
    green = np.where(valid, green, 0).astype(np.uint8)
    return green, valid, float(resx), float(resy)


def read_binary_green_full(fp: Path, threshold=None, mask_nodata=False):
    """Full raster -> binary green (1/0), valid mask, resolution."""
    with rasterio.open(fp) as src:
        arr = src.read(1)
        nodata = src.nodata
        resx, resy = src.res

    nodata_mask = (arr == nodata) if nodata is not None else np.zeros(arr.shape, dtype=bool)
    green = (arr == 1).astype(np.uint8) if threshold is None else (arr >= threshold).astype(np.uint8)

    if mask_nodata:
        valid = ~nodata_mask
        green = np.where(valid, green, 0).astype(np.uint8)
    else:
        valid = np.ones(arr.shape, dtype=bool)

    return green, valid, float(resx), float(resy)


# -------------------------
# Metrics
# -------------------------
def compute_adjacencies_4n(binary, valid=None, count_boundary_as_edge=False):
    """4-neighbor adjacency FF and FN (both directions)."""
    b = binary.astype(np.uint8)
    v = np.ones(b.shape, dtype=bool) if valid is None else valid.astype(bool)

    left = b[:, :-1]
    right = b[:, 1:]
    v_lr = np.ones(left.shape, dtype=bool) if count_boundary_as_edge else (v[:, :-1] & v[:, 1:])
    ff_h = np.sum(((left == 1) & (right == 1)) & v_lr)
    fn_h = np.sum(((left == 1) & (right == 0)) & v_lr) + np.sum(((left == 0) & (right == 1)) & v_lr)

    up = b[:-1, :]
    down = b[1:, :]
    v_ud = np.ones(up.shape, dtype=bool) if count_boundary_as_edge else (v[:-1, :] & v[1:, :])
    ff_v = np.sum(((up == 1) & (down == 1)) & v_ud)
    fn_v = np.sum(((up == 1) & (down == 0)) & v_ud) + np.sum(((up == 0) & (down == 1)) & v_ud)

    return int(ff_h + ff_v), int(fn_h + fn_v)


def compute_edge_density(binary, valid, pixel_size, landscape_area):
    _, fn = compute_adjacencies_4n(binary, valid=valid, count_boundary_as_edge=COUNT_BUFFER_BOUNDARY_AS_EDGE)
    edge_length = fn * pixel_size
    return edge_length / landscape_area if landscape_area > 0 else np.nan


def compute_pladj(binary, valid):
    ff, fn = compute_adjacencies_4n(binary, valid=valid, count_boundary_as_edge=COUNT_BUFFER_BOUNDARY_AS_EDGE)
    denom = ff + fn
    return 100.0 * ff / denom if denom > 0 else np.nan


def compute_ai(binary, valid=None):
    """AI = FF / maxFF * 100 (maxFF from compact block)."""
    b = binary.astype(np.uint8)
    n = int(b.sum())
    if n <= 1:
        return np.nan

    ff_obs, _ = compute_adjacencies_4n(b, valid=valid, count_boundary_as_edge=COUNT_BUFFER_BOUNDARY_AS_EDGE)

    a = max(int(math.floor(math.sqrt(n))), 1)
    bdim = int(math.ceil(n / a))
    ref = np.zeros((a, bdim), dtype=np.uint8)
    ref.flat[:n] = 1
    ff_ref, _ = compute_adjacencies_4n(ref, valid=None, count_boundary_as_edge=True)

    return 100.0 * ff_obs / ff_ref if ff_ref > 0 else np.nan


def compute_core_area(binary, valid, pixel_area):
    """TCA: 3x3 window all green AND all valid."""
    b = binary.astype(np.uint8)
    v = valid.astype(np.uint8)
    if b.sum() == 0:
        return 0.0

    pb = np.pad(b, 1, mode="constant", constant_values=0)
    pv = np.pad(v, 1, mode="constant", constant_values=0)

    wbg = (
        pb[:-2, :-2] + pb[:-2, 1:-1] + pb[:-2, 2:] +
        pb[1:-1, :-2] + pb[1:-1, 1:-1] + pb[1:-1, 2:] +
        pb[2:, :-2] + pb[2:, 1:-1] + pb[2:, 2:]
    )
    wbv = (
        pv[:-2, :-2] + pv[:-2, 1:-1] + pv[:-2, 2:] +
        pv[1:-1, :-2] + pv[1:-1, 1:-1] + pv[1:-1, 2:] +
        pv[2:, :-2] + pv[2:, 1:-1] + pv[2:, 2:]
    )
    core = ((wbg == 9) & (wbv == 9)).astype(np.uint8)
    return float(core.sum()) * pixel_area


def compute_patches(binary, connectivity=2):
    return label(binary.astype(bool), connectivity=connectivity)


def compute_patch_metrics(binary, pixel_area):
    """NP, MPA, largest area, ENN (pixel), patch areas."""
    lab = compute_patches(binary, connectivity=2)
    if lab.max() == 0:
        return dict(NP=0, MPA=np.nan, LPI_largest_area=0.0, ENN_pix=np.nan, patch_areas=[])

    props = regionprops(lab)
    areas_cells = np.array([p.area for p in props], dtype=float)
    areas = areas_cells * pixel_area

    npatches = int(len(areas))
    mpa = float(np.mean(areas)) if npatches > 0 else np.nan
    largest_area = float(np.max(areas))

    centroids = np.array([p.centroid for p in props], dtype=float)
    if len(centroids) <= 1:
        enn_pix = np.nan
    else:
        tree = cKDTree(centroids)
        dists, _ = tree.query(centroids, k=2)
        enn_pix = float(np.mean(dists[:, 1]))

    return dict(NP=npatches, MPA=mpa, LPI_largest_area=largest_area, ENN_pix=enn_pix, patch_areas=areas)


def compute_ldi(patch_areas, landscape_area):
    """LDI = 1 - Σ (a_i/A)^2"""
    if landscape_area <= 0:
        return np.nan
    if len(patch_areas) == 0:
        return 0.0
    fracs = np.array(patch_areas, dtype=float) / float(landscape_area)
    return float(1.0 - np.sum(fracs ** 2))


# -------------------------
# Stats helpers
# -------------------------
def bring_front(df, front_cols):
    front_cols = [c for c in front_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in front_cols]
    return df[front_cols + other_cols]


def zscore_series(s: pd.Series) -> pd.Series:
    v = s.astype(float)
    mu = np.nanmean(v.values)
    sd = np.nanstd(v.values, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (v - mu) / sd


def safe_linear_slope(years, values):
    years = np.array(years, dtype=float)
    values = np.array(values, dtype=float)
    mask = np.isfinite(years) & np.isfinite(values)
    years = years[mask]
    values = values[mask]
    if len(years) < 2:
        return np.nan
    x = years - years.mean()
    y = values - values.mean()
    denom = np.sum(x * x)
    return float(np.sum(x * y) / denom) if denom != 0 else np.nan


def minmax_norm_by_group(df: pd.DataFrame, metric_cols: List[str], group_col: str) -> pd.DataFrame:
    """Per-group min-max normalization to [0,1] for each metric column."""
    out = df.copy()
    for c in metric_cols:
        def _norm(g):
            v = g[c].astype(float).values
            vmin = np.nanmin(v)
            vmax = np.nanmax(v)
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax == vmin):
                return pd.Series([np.nan] * len(g), index=g.index)
            return (g[c].astype(float) - vmin) / (vmax - vmin)
        out[c + "_n01"] = out.groupby(group_col, group_keys=False).apply(_norm)
    return out


def scale_sort_key(scale: str) -> Tuple[int, int]:
    """Sort: bub first, then r500/r1000/... ascending."""
    if scale == "bub":
        return (0, 0)
    m = re.match(r"r(\d+)$", str(scale))
    if m:
        return (1, int(m.group(1)))
    return (9, 999999)


def build_lookups(points_gdf: gpd.GeoDataFrame):
    """
    Return dicts keyed by venue_id and venue_name:
    - id2meta: venue_id -> dict(geom, venue_name, city_id, city_name)
    - name2meta: venue_name -> same dict
    """
    id2meta: Dict[int, Dict] = {}
    name2meta: Dict[str, Dict] = {}

    for _, r in points_gdf.iterrows():
        meta = {
            "geom": r.geometry,
            "venue_name": str(r.get("venue_name", "")),
            "city_id": str(r.get("city_id", "")),
            "city_name": str(r.get("city_name", "")),
        }
        if pd.notna(r.get("venue_id")):
            vid = int(r["venue_id"])
            id2meta[vid] = meta

        nm = str(r.get("venue_name", "")).strip()
        if nm:
            name2meta[nm] = meta

    return id2meta, name2meta


# -------------------------
# Period + error bars
# -------------------------
def assign_period(year: int, award_year: int, host_year: int, post_include_host_year: bool = False) -> Optional[str]:
    """
    pre5: [award-5, award-1]
    prep: [award, host-1]
    post5:
      - if post_include_host_year=False: [host+1, host+5]
      - else: [host, host+4]
    """
    y = int(year)

    pre_start, pre_end = award_year - 5, award_year - 1
    prep_start, prep_end = award_year, host_year - 1
    if post_include_host_year:
        post_start, post_end = host_year, host_year + 4
    else:
        post_start, post_end = host_year + 1, host_year + 5

    if pre_start <= y <= pre_end:
        return "pre5"
    if prep_start <= y <= prep_end:
        return "prep"
    if post_start <= y <= post_end:
        return "post5"
    return None


def add_errorbars(df_long: pd.DataFrame, group_cols: List[str], value_col: str = "value") -> pd.DataFrame:
    """Compute n/mean/sd/se/ci95 for each group from raw values."""
    g = df_long.groupby(group_cols, dropna=False)[value_col]
    out = g.agg(
        n="count",
        mean="mean",
        sd=lambda x: x.std(ddof=1)
    ).reset_index()
    out["se"] = out["sd"] / np.sqrt(out["n"])
    out["ci95_low"] = out["mean"] - 1.96 * out["se"]
    out["ci95_high"] = out["mean"] + 1.96 * out["se"]
    return out


def pivot_period_wide(period_long: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
    """Long -> wide columns like mean_CFI, sd_CFI, ci95_low_CFI ..."""
    wide = period_long.pivot_table(
        index=id_cols,
        columns="index",
        values=["mean", "sd", "se", "ci95_low", "ci95_high", "n"],
        aggfunc="first"
    )
    wide.columns = [f"{stat}_{idx}" for stat, idx in wide.columns]
    wide = wide.reset_index()
    return wide


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    rasters = find_rasters(ROOT)
    if not rasters:
        raise FileNotFoundError(f"No tif/tiff found under: {ROOT}")

    pts_base = load_venue_points(POINTS_PATH)
    if pts_base.crs is None:
        pts_base = pts_base.set_crs("EPSG:4326", allow_override=True)

    # cache points per raster CRS
    pts_cache: Dict[str, Tuple[gpd.GeoDataFrame, Dict[int, Dict], Dict[str, Dict]]] = {}

    rows = []
    for fp in rasters:
        venue, venue_id, venue_name, year = infer_venue_year(fp)

        with rasterio.open(fp) as src:
            raster_crs = src.crs
        if raster_crs is None:
            print(f"[Skip] Raster CRS missing: {fp}")
            continue

        crs_key = CRS.from_user_input(raster_crs).to_string()
        if crs_key not in pts_cache:
            pts_in = pts_base if CRS.from_user_input(pts_base.crs) == CRS.from_user_input(raster_crs) else pts_base.to_crs(raster_crs)
            id2meta, name2meta = build_lookups(pts_in)
            pts_cache[crs_key] = (pts_in, id2meta, name2meta)

        _, id2meta, name2meta = pts_cache[crs_key]

        # ---- match venue point + get city info ----
        meta = None
        if venue_id is not None and venue_id in id2meta:
            meta = id2meta[venue_id]
        elif venue_name in name2meta:
            meta = name2meta[venue_name]
        elif venue in name2meta:
            meta = name2meta[venue]
        else:
            print(f"[Skip] No matching point for raster: {fp.name} | venue_id={venue_id} venue_name={venue_name}")
            continue

        pt = meta["geom"]
        city_id = meta.get("city_id", "")
        city_name = meta.get("city_name", "")

        # -------- A) bub (full raster) --------
        green, valid, resx, resy = read_binary_green_full(fp, threshold=THRESHOLD, mask_nodata=MASK_NODATA)
        pixel_area = abs(resx * resy)
        pixel_size = float((abs(resx) + abs(resy)) / 2.0)
        valid_cells = int(valid.sum())
        landscape_area = float(valid_cells) * pixel_area

        tca = compute_core_area(green, valid, pixel_area)
        patch_m = compute_patch_metrics(green, pixel_area)
        npatch = patch_m["NP"]
        mpa = patch_m["MPA"]
        lpi = 100.0 * (patch_m["LPI_largest_area"] / landscape_area) if landscape_area > 0 else np.nan
        enn = patch_m["ENN_pix"] * pixel_size if np.isfinite(patch_m["ENN_pix"]) else np.nan
        ldi = compute_ldi(patch_m["patch_areas"], landscape_area)
        ed = compute_edge_density(green, valid, pixel_size, landscape_area)
        pladj = compute_pladj(green, valid)
        ai = compute_ai(green, valid=valid)
        green_cells = float(green.sum())
        green_area = green_cells * pixel_area
        green_frac = (green_cells / float(valid_cells)) if valid_cells > 0 else np.nan

        rows.append({
            "venue": venue, "venue_id": venue_id, "venue_name": venue_name,
            "city_id": city_id, "city_name": city_name,
            "year": year,
            "scale": "bub", "buffer_m": 0,  # bub fixed as 0
            "file": str(fp),

            "TCA": tca, "LPI": lpi, "LDI": ldi, "AI": ai, "PLADJ": pladj,
            "ENN": enn, "NP": npatch, "MPA": mpa, "ED": ed,

            "pixel_size": pixel_size, "pixel_area": pixel_area,
            "valid_cells": valid_cells, "landscape_area": landscape_area,
            "green_area": green_area, "green_frac": green_frac,
        })

        # -------- B) buffers --------
        for buf_m in BUFFERS_M:
            buf_geom = build_buffer_geom_in_raster_crs(pt, float(buf_m), raster_crs)
            green, valid, resx, resy = read_binary_green_with_geom(fp, buf_geom, threshold=THRESHOLD, mask_nodata=MASK_NODATA)

            pixel_area = abs(resx * resy)
            pixel_size = float((abs(resx) + abs(resy)) / 2.0)
            valid_cells = int(valid.sum())
            landscape_area = float(valid_cells) * pixel_area

            tca = compute_core_area(green, valid, pixel_area)
            patch_m = compute_patch_metrics(green, pixel_area)
            npatch = patch_m["NP"]
            mpa = patch_m["MPA"]
            lpi = 100.0 * (patch_m["LPI_largest_area"] / landscape_area) if landscape_area > 0 else np.nan
            enn = patch_m["ENN_pix"] * pixel_size if np.isfinite(patch_m["ENN_pix"]) else np.nan
            ldi = compute_ldi(patch_m["patch_areas"], landscape_area)
            ed = compute_edge_density(green, valid, pixel_size, landscape_area)
            pladj = compute_pladj(green, valid)
            ai = compute_ai(green, valid=valid)
            green_cells = float(green.sum())
            green_area = green_cells * pixel_area
            green_frac = (green_cells / float(valid_cells)) if valid_cells > 0 else np.nan

            rows.append({
                "venue": venue, "venue_id": venue_id, "venue_name": venue_name,
                "city_id": city_id, "city_name": city_name,
                "year": year,
                "scale": f"r{int(buf_m)}", "buffer_m": int(buf_m),
                "file": str(fp),

                "TCA": tca, "LPI": lpi, "LDI": ldi, "AI": ai, "PLADJ": pladj,
                "ENN": enn, "NP": npatch, "MPA": mpa, "ED": ed,

                "pixel_size": pixel_size, "pixel_area": pixel_area,
                "valid_cells": valid_cells, "landscape_area": landscape_area,
                "green_area": green_area, "green_frac": green_frac,
            })

    df_raw = pd.DataFrame(rows)
    if df_raw.empty:
        raise RuntimeError("No results computed. Check POINTS_PATH matching, raster CRS, and filenames.")

    df_raw["__scale_order"] = df_raw["scale"].astype(str).apply(scale_sort_key)
    df_raw = df_raw.sort_values(["city_id", "venue_id", "venue", "__scale_order", "year"], na_position="last") \
                   .drop(columns="__scale_order").reset_index(drop=True)
    df_raw = bring_front(df_raw, ["city_id", "city_name", "venue", "venue_id", "venue_name", "year", "scale", "buffer_m", "file"])

    raw_csv = out_dir / "fragmentation_metrics_raw_scales.csv"
    df_raw.to_csv(raw_csv, index=False, encoding="utf-8-sig")

    # ===================== 2) Normalize 0-1 (per scale) =====================
    metric_cols = ["TCA", "LPI", "LDI", "AI", "PLADJ", "ENN", "NP", "MPA", "ED"]
    df_norm = minmax_norm_by_group(df_raw, metric_cols, group_col="scale")
    df_norm = bring_front(df_norm, ["city_id", "city_name", "venue", "venue_id", "venue_name", "year", "scale", "buffer_m", "file"])

    # ===================== 3) Unify direction: higher => more fragmented =====================
    df_frag = df_norm.copy()
    less_frag_higher = ["TCA", "LPI", "AI", "PLADJ", "MPA"]
    more_frag_higher = ["LDI", "ENN", "NP", "ED"]

    for c in less_frag_higher:
        df_frag[c + "_frag"] = 1.0 - df_frag[c + "_n01"]
    for c in more_frag_higher:
        df_frag[c + "_frag"] = df_frag[c + "_n01"]

    # ===================== 4) CFI / AFI / SFI (yearly) =====================
    df_idx = df_frag.copy()
    df_idx["CFI"] = df_idx[["TCA_frag", "LPI_frag", "LDI_frag"]].mean(axis=1)
    df_idx["AFI"] = df_idx[["AI_frag", "PLADJ_frag", "ENN_frag"]].mean(axis=1)
    df_idx["SFI"] = df_idx[["NP_frag", "MPA_frag", "ED_frag"]].mean(axis=1)
    df_idx = bring_front(df_idx, ["city_id", "city_name", "venue", "venue_id", "venue_name", "year", "scale", "buffer_m", "file"])

    # ===================== 5) plotting_table (venue-year-scale long) =====================
    df_plot = df_idx[[
        "city_id", "city_name",
        "venue", "venue_id", "venue_name",
        "year", "scale", "buffer_m",
        "CFI", "AFI", "SFI"
    ]].copy()
    df_plot["IFCI"] = df_plot[["CFI", "AFI", "SFI"]].mean(axis=1)

    df_plot["__scale_order"] = df_plot["scale"].astype(str).apply(scale_sort_key)
    df_plot = df_plot.sort_values(["city_id", "venue_id", "venue", "__scale_order", "year"], na_position="last") \
                     .drop(columns="__scale_order").reset_index(drop=True)

    df_plot = bring_front(
        df_plot,
        ["city_id", "city_name", "venue", "venue_id", "venue_name", "year", "scale", "buffer_m",
         "CFI", "AFI", "SFI", "IFCI"]
    )

    plot_csv = out_dir / "fragmentation_for_plotting_scales.csv"
    df_plot.to_csv(plot_csv, index=False, encoding="utf-8-sig")

    # ===================== 6) 3-period stats (VENUE-level) =====================
    df_period = df_plot.copy()
    df_period["period"] = df_period["year"].apply(
        lambda y: assign_period(y, AWARD_YEAR, HOST_YEAR, POST_INCLUDE_HOST_YEAR)
    )
    df_period = df_period[df_period["period"].notna()].copy()

    period_value_cols = ["CFI", "AFI", "SFI", "IFCI"]
    df_period_long = df_period.melt(
        id_vars=["city_id", "city_name", "venue", "venue_id", "venue_name", "scale", "buffer_m", "year", "period"],
        value_vars=period_value_cols,
        var_name="index",
        value_name="value"
    )
    df_period_long["value"] = pd.to_numeric(df_period_long["value"], errors="coerce")

    # venue-level errorbars: across YEARS within period
    venue_group_cols = ["city_id", "city_name", "venue", "venue_id", "venue_name", "scale", "buffer_m", "period", "index"]
    df_period_venue_long = add_errorbars(df_period_long, group_cols=venue_group_cols, value_col="value")
    df_period_venue_wide = pivot_period_wide(
        df_period_venue_long,
        id_cols=["city_id", "city_name", "venue", "venue_id", "venue_name", "scale", "buffer_m", "period"]
    )

    # overall across venues (optional)
    df_venue_means = df_period_venue_long.rename(columns={"mean": "venue_mean"}).copy()
    df_period_overall_long = df_venue_means.groupby(["scale", "buffer_m", "period", "index"], dropna=False)["venue_mean"].agg(
        n_venues="count",
        mean="mean",
        sd=lambda x: x.std(ddof=1)
    ).reset_index()
    df_period_overall_long["se"] = df_period_overall_long["sd"] / np.sqrt(df_period_overall_long["n_venues"])
    df_period_overall_long["ci95_low"] = df_period_overall_long["mean"] - 1.96 * df_period_overall_long["se"]
    df_period_overall_long["ci95_high"] = df_period_overall_long["mean"] + 1.96 * df_period_overall_long["se"]

    # save venue period csv
    period_venue_long_csv = out_dir / "fragmentation_period_venue_long.csv"
    period_venue_wide_csv = out_dir / "fragmentation_period_venue_wide.csv"
    period_overall_long_csv = out_dir / "fragmentation_period_overall_long.csv"
    df_period_venue_long.to_csv(period_venue_long_csv, index=False, encoding="utf-8-sig")
    df_period_venue_wide.to_csv(period_venue_wide_csv, index=False, encoding="utf-8-sig")
    df_period_overall_long.to_csv(period_overall_long_csv, index=False, encoding="utf-8-sig")

    # ===================== 7) 3-period stats (CITY-level) =====================
    # Step 1: aggregate venue-year -> city-year (mean across venues in same city_id)
    df_city_year = df_plot.groupby(["city_id", "city_name", "scale", "buffer_m", "year"], dropna=False).agg(
        CFI=("CFI", "mean"),
        AFI=("AFI", "mean"),
        SFI=("SFI", "mean"),
        IFCI=("IFCI", "mean"),
        n_venues=("venue_id", "nunique")
    ).reset_index()

    # Step 2: assign period
    df_city_year["period"] = df_city_year["year"].apply(lambda y: assign_period(y, AWARD_YEAR, HOST_YEAR, POST_INCLUDE_HOST_YEAR))
    df_city_year = df_city_year[df_city_year["period"].notna()].copy()

    # Step 3: long + errorbars across YEARS within each period (per city_id)
    df_city_long0 = df_city_year.melt(
        id_vars=["city_id", "city_name", "scale", "buffer_m", "year", "period", "n_venues"],
        value_vars=["CFI", "AFI", "SFI", "IFCI"],
        var_name="index",
        value_name="value"
    )
    df_city_long0["value"] = pd.to_numeric(df_city_long0["value"], errors="coerce")

    city_group_cols = ["city_id", "city_name", "scale", "buffer_m", "period", "index"]
    df_period_city_long = add_errorbars(df_city_long0, group_cols=city_group_cols, value_col="value")

    # add city-period average venue count (optional)
    nv = df_city_year.groupby(["city_id", "city_name", "scale", "buffer_m", "period"], dropna=False)["n_venues"].mean().reset_index()
    nv = nv.rename(columns={"n_venues": "mean_n_venues"})
    df_period_city_long = df_period_city_long.merge(nv, on=["city_id", "city_name", "scale", "buffer_m", "period"], how="left")

    df_period_city_wide = pivot_period_wide(
        df_period_city_long,
        id_cols=["city_id", "city_name", "scale", "buffer_m", "period", "mean_n_venues"]
    )

    # Step 4: overall across cities (errorbars across city means)
    df_city_means = df_period_city_long.rename(columns={"mean": "city_mean"}).copy()
    df_period_city_overall_long = df_city_means.groupby(["scale", "buffer_m", "period", "index"], dropna=False)["city_mean"].agg(
        n_cities="count",
        mean="mean",
        sd=lambda x: x.std(ddof=1)
    ).reset_index()
    df_period_city_overall_long["se"] = df_period_city_overall_long["sd"] / np.sqrt(df_period_city_overall_long["n_cities"])
    df_period_city_overall_long["ci95_low"] = df_period_city_overall_long["mean"] - 1.96 * df_period_city_overall_long["se"]
    df_period_city_overall_long["ci95_high"] = df_period_city_overall_long["mean"] + 1.96 * df_period_city_overall_long["se"]

    # save city period csv
    period_city_long_csv = out_dir / "fragmentation_period_city_long.csv"
    period_city_wide_csv = out_dir / "fragmentation_period_city_wide.csv"
    period_city_overall_long_csv = out_dir / "fragmentation_period_city_overall_long.csv"
    df_period_city_long.to_csv(period_city_long_csv, index=False, encoding="utf-8-sig")
    df_period_city_wide.to_csv(period_city_wide_csv, index=False, encoding="utf-8-sig")
    df_period_city_overall_long.to_csv(period_city_overall_long_csv, index=False, encoding="utf-8-sig")

    # ===================== 8) Excel report =====================
    xlsx_path = out_dir / "fragmentation_report_scales.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_raw.to_excel(writer, sheet_name="raw_metrics", index=False)
        df_norm.to_excel(writer, sheet_name="normalized_0_1", index=False)
        df_plot.to_excel(writer, sheet_name="plotting_table", index=False)

        # venue-level period sheets
        df_period_venue_long.to_excel(writer, sheet_name="period_venue_long", index=False)
        df_period_venue_wide.to_excel(writer, sheet_name="period_venue_wide", index=False)
        df_period_overall_long.to_excel(writer, sheet_name="period_venue_overall", index=False)

        # city-level period sheets (what you asked)
        df_period_city_long.to_excel(writer, sheet_name="period_city_long", index=False)
        df_period_city_wide.to_excel(writer, sheet_name="period_city_wide", index=False)
        df_period_city_overall_long.to_excel(writer, sheet_name="period_city_overall", index=False)

    print("DONE.")
    print(f"Raw CSV : {raw_csv}")
    print(f"Plot CSV: {plot_csv}")
    print(f"Venue period long   : {period_venue_long_csv}")
    print(f"City period long    : {period_city_long_csv}")
    print(f"Report              : {xlsx_path}")


if __name__ == "__main__":
    # pip install numpy pandas rasterio scikit-image scipy openpyxl geopandas shapely pyproj
    main()