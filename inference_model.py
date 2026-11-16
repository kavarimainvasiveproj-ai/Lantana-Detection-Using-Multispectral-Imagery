import json
from pathlib import Path
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling, ColorInterp
from rasterio.features import shapes
from rasterio.warp import transform_geom

from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
from shapely.geometry import shape, mapping

import torch
import torch.nn as nn


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Resolve project-relative paths from this script, not the terminal's cwd.
PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "lantana_net_model_converted.pth"

# Options: "multispectral" or "separate"
INPUT_MODE = "multispectral"

# Option A: Single multispectral TIFF
INPUT_TIF = PROJECT_DIR / "Site3_Oct_20261003_ms.tif"

# Option B: Separate spectral TIFFs
BAND_PATHS = {
    "green": PROJECT_DIR / "data" / "Green.tif",
    "red": PROJECT_DIR / "data" / "Red.tif",
    "rededge": PROJECT_DIR / "data" / "RedEdge.tif",
    "nir": PROJECT_DIR / "data" / "NIR.tif"
}

# Band positions in multispectral TIFF
BAND_INDEXES = {
    "green": 1,
    "red": 2,
    "rededge": 3,
    "nir": 4
}

# Alpha is detected from TIFF metadata; do not assume band 5 is alpha.

OUTPUT_DIR = PROJECT_DIR / "inference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PROB = OUTPUT_DIR / "lantana_probability.tif"
OUT_MASK = OUTPUT_DIR / "lantana_mask.tif"
OUT_JSON = OUTPUT_DIR / "lantana_polygons.geojson"

ROW_CHUNK = 256
GPU_BATCH = 131072

# Reference post-processing configuration
REFERENCE_RESOLUTION = 0.01922
REFERENCE_MIN_PIXELS = 150
REFERENCE_CLOSE_KERNEL = 3
REFERENCE_OPEN_KERNEL = 2

# Resolution-aware post-processing
RESOLUTION_AWARE = True

# Polygon configuration
SIMPLIFY_METERS = 0.05
MIN_POLYGON_AREA_M2 = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# ============================================================
# 2. MODEL ARCHITECTURE
# ============================================================

class LantanaNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

# Safely load the checkpoint. These NumPy globals are allowlisted because
# the converted checkpoint created by chec.py still contains NumPy metadata.
# Only use a checkpoint from a source you trust.
safe_numpy_globals = [
    np._core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
]
if hasattr(np, "dtypes") and hasattr(np.dtypes, "Float32DType"):
    safe_numpy_globals.append(np.dtypes.Float32DType)

torch.serialization.add_safe_globals(safe_numpy_globals)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model = LantanaNet().to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

mu = np.asarray(checkpoint["mu"], dtype=np.float32)
std = np.asarray(checkpoint["std"], dtype=np.float32)

EPS = float(checkpoint["eps"])

PROB_THRESHOLD = float(
    checkpoint.get("prob_threshold", 0.9)
)

assert mu.shape == (4,)
assert std.shape == (4,)
assert np.all(std > 0)

print("Model loaded")
print("Probability threshold:", PROB_THRESHOLD)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def compute_features(red, green, nir, rededge):

    ndvi = (nir - red) / (nir + red + EPS)

    ndre = (nir - rededge) / (nir + rededge + EPS)

    gndvi = (nir - green) / (nir + green + EPS)

    re_ratio = nir / (rededge + EPS)

    ndvi = np.clip(ndvi, -1, 1)
    ndre = np.clip(ndre, -1, 1)
    gndvi = np.clip(gndvi, -1, 1)
    re_ratio = np.clip(re_ratio, 0, 10)

    return np.stack(
        [ndvi, ndre, gndvi, re_ratio],
        axis=-1
    ).astype(np.float32)


# ============================================================
# 5. GPU INFERENCE
# ============================================================

def predict_batch(features):

    predictions = np.empty(
        len(features),
        dtype=np.float32
    )

    with torch.inference_mode():

        for start in range(0, len(features), GPU_BATCH):

            end = min(start + GPU_BATCH, len(features))

            batch = features[start:end]

            normalized = (
                (batch - mu) / std
            ).astype(np.float32)

            xb = torch.from_numpy(normalized).to(DEVICE)

            logits = model(xb)

            probabilities = torch.sigmoid(logits)

            predictions[start:end] = (
                probabilities.cpu().numpy()
            )

    return predictions


# ============================================================
# 6. UNIFIED MULTISPECTRAL DATA LOADER
# ============================================================

class SpectralReader:

    def __init__(self, stack):

        self.datasets = {}
        self.stack = stack

        if INPUT_MODE == "multispectral":

            self.reference = stack.enter_context(
                rasterio.open(INPUT_TIF)
            )

            for name in ("green", "red", "rededge", "nir"):

                idx = BAND_INDEXES[name]

                if not 1 <= idx <= self.reference.count:
                    raise ValueError(f"Missing band: {name}")

                description = self.reference.descriptions[idx - 1]

                if description:

                    actual = (
                        description.lower()
                        .replace(" ", "")
                        .replace("_", "")
                        .replace("-", "")
                    )

                    if actual != name:
                        raise ValueError(
                            f"Expected {name}, found {description}"
                        )

            print("Input: single multispectral TIFF")

        elif INPUT_MODE == "separate":

            originals = {}

            for name, path in BAND_PATHS.items():

                originals[name] = stack.enter_context(
                    rasterio.open(path)
                )

                if originals[name].crs is None:
                    raise ValueError(f"Missing CRS: {path}")

                if originals[name].count != 1:
                    raise ValueError(
                        f"Expected single-band TIFF: {path}"
                    )

            # NIR defines the target spatial grid
            self.reference = originals["nir"]

            for name, src in originals.items():

                aligned = (
                    src.crs == self.reference.crs
                    and src.transform == self.reference.transform
                    and src.width == self.reference.width
                    and src.height == self.reference.height
                )

                if aligned:

                    self.datasets[name] = src

                else:

                    print("Aligning band:", name)

                    self.datasets[name] = stack.enter_context(
                        WarpedVRT(
                            src,
                            crs=self.reference.crs,
                            transform=self.reference.transform,
                            width=self.reference.width,
                            height=self.reference.height,
                            resampling=Resampling.bilinear,
                            dtype="float32",
                            nodata=np.nan
                        )
                    )

            print("Input: separate spectral TIFFs")

        else:

            raise ValueError("Invalid INPUT_MODE")

    def read(self, window):

        result = {}
        valid = None

        for name in ("green", "red", "rededge", "nir"):

            if INPUT_MODE == "multispectral":

                arr = self.reference.read(
                    BAND_INDEXES[name],
                    window=window,
                    masked=True,
                    out_dtype="float32"
                )

            else:

                arr = self.datasets[name].read(
                    1,
                    window=window,
                    masked=True,
                    out_dtype="float32"
                )

            data = arr.filled(np.nan)

            band_valid = (
                ~np.ma.getmaskarray(arr)
                & np.isfinite(data)
            )

            result[name] = data

            if valid is None:
                valid = band_valid.copy()
            else:
                valid &= band_valid

        # Apply alpha masking only when TIFF metadata identifies an alpha band.
        # A fifth band is not automatically an alpha band; it may be another
        # spectral band, and masking it as alpha can discard valid pixels.
        if INPUT_MODE == "multispectral":
            alpha_index = None

            for band_number, color_interp in enumerate(
                self.reference.colorinterp, start=1
            ):
                description = self.reference.descriptions[band_number - 1]
                description_is_alpha = (
                    description is not None
                    and description.strip().lower() == "alpha"
                )

                if color_interp == ColorInterp.alpha or description_is_alpha:
                    alpha_index = band_number
                    break

            if alpha_index is not None:
                alpha = self.reference.read(
                    alpha_index,
                    window=window,
                    masked=True
                )
                alpha_data = alpha.filled(0)
                valid &= (~np.ma.getmaskarray(alpha)) & (alpha_data > 0)

        return result, valid


# ============================================================
# 7. RUN MULTISPECTRAL INFERENCE
# ============================================================

with ExitStack() as stack:

    reader = SpectralReader(stack)
    src = reader.reference

    H = src.height
    W = src.width

    CRS = src.crs
    TRANSFORM = src.transform

    if CRS is None:
        raise ValueError("Input raster has no CRS")

    if not CRS.is_projected or CRS.linear_units != "metre":
        raise ValueError("Expected projected CRS in metres")

    PIXEL_AREA = abs(
        TRANSFORM.a * TRANSFORM.e
        - TRANSFORM.b * TRANSFORM.d
    )

    RESOLUTION = np.sqrt(PIXEL_AREA)

    print("Width:", W)
    print("Height:", H)
    print("CRS:", CRS)
    print("Resolution:", src.res)
    print("Pixel area:", PIXEL_AREA)

    # --------------------------------------------------------
    # Resolution-aware post-processing parameters
    # --------------------------------------------------------

    if RESOLUTION_AWARE:

        reference_area = (
            REFERENCE_MIN_PIXELS
            * REFERENCE_RESOLUTION ** 2
        )

        MIN_PIXELS = max(
            1,
            int(np.ceil(reference_area / PIXEL_AREA))
        )

        CLOSE_KERNEL = max(
            1,
            round(
                REFERENCE_CLOSE_KERNEL
                * REFERENCE_RESOLUTION
                / RESOLUTION
            )
        )

        OPEN_KERNEL = max(
            1,
            round(
                REFERENCE_OPEN_KERNEL
                * REFERENCE_RESOLUTION
                / RESOLUTION
            )
        )

    else:

        MIN_PIXELS = REFERENCE_MIN_PIXELS
        CLOSE_KERNEL = REFERENCE_CLOSE_KERNEL
        OPEN_KERNEL = REFERENCE_OPEN_KERNEL

    print("Minimum patch pixels:", MIN_PIXELS)
    print("Closing kernel:", CLOSE_KERNEL)
    print("Opening kernel:", OPEN_KERNEL)

    # --------------------------------------------------------
    # Output probability raster
    # --------------------------------------------------------

    prob_profile = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": 1,
        "dtype": "float32",
        "crs": CRS,
        "transform": TRANSFORM,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER"
    }

    total_valid = 0
    total_positive = 0

    with rasterio.open(
        OUT_PROB,
        "w",
        **prob_profile
    ) as dst:

        dst.set_band_description(
            1, "Lantana probability"
        )

        for y0 in range(0, H, ROW_CHUNK):

            y1 = min(y0 + ROW_CHUNK, H)

            window = Window(
                0, y0, W, y1 - y0
            )

            bands, valid = reader.read(window)

            prob_chunk = np.full(
                (y1 - y0, W),
                np.nan,
                dtype=np.float32
            )

            if valid.any():

                features = compute_features(
                    bands["red"][valid],
                    bands["green"][valid],
                    bands["nir"][valid],
                    bands["rededge"][valid]
                )

                finite = np.isfinite(features).all(axis=1)

                p = np.full(
                    len(features),
                    np.nan,
                    dtype=np.float32
                )

                if finite.any():

                    p[finite] = predict_batch(
                        features[finite]
                    )

                prob_chunk[valid] = p

                total_valid += int(
                    np.isfinite(p).sum()
                )

                total_positive += int(
                    np.count_nonzero(p >= PROB_THRESHOLD)
                )

            dst.write(
                prob_chunk,
                1,
                window=window
            )

            print(f"Processed rows {y0}-{y1}/{H}")

print("Probability map saved:", OUT_PROB)


# ============================================================
# 8. POST-PROCESSING
# ============================================================

print("Starting post-processing")

with rasterio.open(OUT_PROB) as src:

    probability = src.read(1)

valid_mask = np.isfinite(probability)

mask = (
    valid_mask
    & (probability >= PROB_THRESHOLD)
)

del probability

# Remove small connected regions
mask = remove_small_objects(
    mask,
    min_size=MIN_PIXELS,
    connectivity=2
)

# Morphological closing
if CLOSE_KERNEL > 1:

    mask = ndi.binary_closing(
        mask,
        structure=np.ones(
            (CLOSE_KERNEL, CLOSE_KERNEL),
            dtype=bool
        )
    )

    mask &= valid_mask

# Morphological opening
if OPEN_KERNEL > 1:

    mask = ndi.binary_opening(
        mask,
        structure=np.ones(
            (OPEN_KERNEL, OPEN_KERNEL),
            dtype=bool
        )
    )

# Fill enclosed holes
mask = ndi.binary_fill_holes(mask)

# Exclude invalid image regions
mask &= valid_mask

# Final minimum area filtering
mask = remove_small_objects(
    mask,
    min_size=MIN_PIXELS,
    connectivity=2
)

positive_pixels = int(mask.sum())

lantana_area_m2 = positive_pixels * PIXEL_AREA

print("Final positive pixels:", positive_pixels)
print("Lantana area (m2):", lantana_area_m2)


# ============================================================
# 9. SAVE CLASSIFICATION TIFF
# ============================================================

mask_profile = prob_profile.copy()

mask_profile.update(
    dtype="uint8",
    nodata=255,
    predictor=2
)

with rasterio.open(
    OUT_MASK,
    "w",
    **mask_profile
) as dst:

    dst.set_band_description(
        1, "Lantana classification"
    )

    for y0 in range(0, H, ROW_CHUNK):

        y1 = min(y0 + ROW_CHUNK, H)

        valid_chunk = valid_mask[y0:y1]
        positive_chunk = mask[y0:y1]

        output = np.full(
            (y1 - y0, W),
            255,
            dtype=np.uint8
        )

        output[valid_chunk] = 0
        output[positive_chunk] = 1

        dst.write(
            output,
            1,
            window=Window(0, y0, W, y1-y0)
        )

print("Classification TIFF saved:", OUT_MASK)

del valid_mask


# ============================================================
# 10. POLYGON EXTRACTION AND GEOJSON EXPORT
# ============================================================

print("Generating Lantana polygons")

total_polygons = 0
polygon_area_total = 0.0

with rasterio.open(OUT_MASK) as src:

    polygons = shapes(
        rasterio.band(src, 1),
        mask=mask,
        transform=src.transform,
        connectivity=4
    )

    # Stream GeoJSON features to disk instead of
    # storing every polygon in RAM.

    with open(OUT_JSON, "w", encoding="utf-8") as f:

        f.write(
            '{"type":"FeatureCollection","features":['
        )

        first_feature = True

        for geom, value in polygons:

            if value != 1:
                continue

            polygon = shape(geom)

            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            if polygon.is_empty:
                continue

            area_m2 = float(polygon.area)

            if area_m2 < MIN_POLYGON_AREA_M2:
                continue

            if SIMPLIFY_METERS > 0:

                polygon = polygon.simplify(
                    SIMPLIFY_METERS,
                    preserve_topology=True
                )

            if polygon.is_empty:
                continue

            # Convert UTM coordinates into WGS84
            geometry = transform_geom(
                src.crs,
                "EPSG:4326",
                mapping(polygon)
            )

            total_polygons += 1
            polygon_area_total += area_m2

            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "id": total_polygons,
                    "class": "Lantana",
                    "area_m2": round(area_m2, 4),
                    "area_hectares": round(
                        area_m2 / 10000, 8
                    ),
                    "classification_threshold": PROB_THRESHOLD
                }
            }

            if not first_feature:
                f.write(",")

            json.dump(feature, f, allow_nan=False)

            first_feature = False

        f.write("]}")

print("GeoJSON saved:", OUT_JSON)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

summary = {
    "input_mode": INPUT_MODE,
    "model": str(MODEL_PATH),
    "resolution_m": float(RESOLUTION),
    "pixel_area_m2": float(PIXEL_AREA),
    "probability_threshold": PROB_THRESHOLD,
    "total_valid_pixels": total_valid,
    "positive_pixels_before_cleaning": total_positive,
    "positive_pixels": positive_pixels,
    "total_polygons": total_polygons,
    "lantana_area_m2": round(lantana_area_m2, 4),
    "lantana_area_hectares": round(
        lantana_area_m2 / 10000, 8
    ),
    "polygon_area_m2": round(polygon_area_total, 4),
    "outputs": {
        "probability": str(OUT_PROB),
        "classification": str(OUT_MASK),
        "polygons": str(OUT_JSON)
    }
}

with open(
    OUTPUT_DIR / "summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(summary, f, indent=4)

print("\n" + "=" * 60)
print("LANTANA INFERENCE COMPLETED")
print("=" * 60)

print(json.dumps(summary, indent=4))