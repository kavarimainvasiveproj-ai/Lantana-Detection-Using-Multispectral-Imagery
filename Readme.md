# Lantana Detection Using Multispectral Imagery

## Project Overview

An automated vegetation detection pipeline that identifies invasive Lantana vegetation using multispectral remote sensing imagery and a trained PyTorch neural network.

The system performs pixel-wise classification using vegetation indices and exports the detected regions as GIS-compatible raster and vector datasets.

## Technologies Used

- Python
- PyTorch
- Rasterio
- NumPy
- SciPy
- Scikit-image
- Shapely
- Remote Sensing
- GIS / QGIS

## Key Features

- Multispectral GeoTIFF processing
- Support for single and separate spectral TIFF inputs
- Vegetation index calculation
- NDVI, NDRE, GNDVI and NIR/Red-edge ratio
- Deep learning-based pixel classification
- CPU and NVIDIA GPU inference
- Window-based raster processing
- Probability map generation
- Morphological image processing
- Raster-to-vector polygon conversion
- GeoJSON export
- Lantana area calculation

## Model Architecture

Feed-forward neural network:

- Input layer: 4 features
- Hidden layer 1: 32 neurons
- Hidden layer 2: 16 neurons
- Output layer: 1 neuron

## Output Files

| File | Description |
|---|---|
| lantana_probability.tif | Pixel-wise probability map |
| lantana_mask.tif | Binary classification mask |
| lantana_polygons.geojson | Detected Lantana polygons |
| summary.json | Processing statistics |

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Vegetation_Inference
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Execution

Configure the model checkpoint and input raster paths in the inference configuration.

Run:

```bash
python inference_model.py
```

## Visualization

Open the generated TIFF and GeoJSON outputs in QGIS for visualization and geospatial analysis.

## Applications

- Invasive vegetation monitoring
- Land-cover classification
- Remote sensing analysis
- Vegetation mapping
- Environmental monitoring
