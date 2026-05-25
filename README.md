# AegisSphere AI — Disaster Intelligence & Early Warning Platform

AegisSphere AI is an advanced, production-scale disaster prediction and early-warning intelligence platform. The system is designed to consume multimodal input streams—including Sentinel-1 Synthetic Aperture Radar (SAR), Sentinel-2 multispectral imagery, Digital Elevation Models (DEM), and live news bulletins—to predict flash flooding, wildfires, and landslides in real-time, alongside DeBERTa/DistilBERT news NLP classification.

---

## Key Features

1. **Geospatial Hazard Segmentation (UNet)**:
   - A custom Multi-Encoder Attention UNet with double convolutions, residual skip-connections, Atrous Spatial Pyramid Pooling (ASPP), and Grid Attention gates.
   - Outputs highly accurate hazard prediction masks (inundation, burn areas, slope instability).
   
2. **Probabilistic Risk Forecasting (Temporal Transformer)**:
   - Temporal Fusion Transformer-style causal sequence forecaster predicting future risk steps (Quantile 10, Median 50, and Upper-bound 90).
   - Utilizes Monte Carlo Dropout during inference to dynamically generate prediction uncertainty intervals.

3. **Multilingual NLP News Classification (PEFT/LoRA)**:
   - News headline classifier powered by a DistilBERT/DeBERTa backbone with native Low-Rank Adaptation (LoRA) modules injected into query/value layers.
   - Automatically parses streams (GDELT, ReliefWeb, CrisisNLP) and triggers alerts when confidence thresholds are exceeded.

4. **FastAPI Inference Server**:
   - Integrates pre-compiled ONNX models via ONNX Runtime sessions with standard PyTorch eager fallbacks.
   - Contains JWT authentication, rate limiting, and real-time WebSocket connection broadcasters.

5. **Premium Glassmorphic GIS Dashboard**:
   - Web interface built with Tailwind CSS, React, Leaflet GIS mapping overlays, and Chart.js forecast visualizations.

---

## Directory Structure

```
AI Disaster System/
├── backend/
│   └── main.py                     # FastAPI backend, JWT Auth, WebSockets & API endpoints
├── cache/
│   ├── unet.onnx                   # Compiled Colab binary segmenter
│   ├── forecaster.onnx             # Compiled Colab binary temporal forecaster
│   └── classifier.onnx             # Compiled Colab binary news classifier
├── configs/
│   └── config.py                   # Pydantic Settings environment configurations
├── datasets/
│   ├── satellite_dataset.py        # Aligned Sentinel-1/2 & DEM loader
│   └── news_dataset.py             # Multilingual news loader with local fallback
├── frontend/
│   └── index.html                  # Single Page React dashboard (GIS + Chart.js)
├── inference/
│   └── onnx_exporter.py            # ONNX export scripts & runtime inference sessions
├── models/
│   ├── unet.py                     # Multi-Encoder Attention UNet PyTorch architecture
│   ├── forecaster.py               # TFT Causal Forecaster & MC Dropout PyTorch model
│   └── classifier.py               # LoRA fine-tuning wrapper & Transformer classifier
├── notebooks/
│   └── aegissphere_colab.ipynb     # Google Colab GPU hyper-scale training runbook
├── tests/
│   └── verify.py                   # System integration verification suite
└── training/
    └── trainer.py                  # Cosine Warmup, SWA, EMA, and AMP training loop
```

---

## Installation & Setup

Ensure you have **Python 3.10+** installed on your system.

### 1. Initialize Virtual Environment & Dependencies
Open your shell (e.g., PowerShell on Windows) and run:
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install requirements
pip install fastapi uvicorn pydantic-settings PyJWT requests numpy torch transformers onnx onnxruntime
```

### 2. Run System Integration Quality Gate
Before launching the service, verify all models, configuration boundaries, and dataset pipelines run successfully:
```powershell
python tests/verify.py
```
*Expected Output:* `[SUCCESS] ALL QUALITY GATE TESTS PASSED successfully. Code ready for distribution.`

---

## Running the Platform

### 1. Launch FastAPI Backend
Start the uvicorn development server on localhost:
```powershell
.\.venv\Scripts\uvicorn.exe backend.main:app --host 127.0.0.1 --port 8000
```
Upon startup, the server loads the pre-compiled ONNX sessions:
```
[*] ONNX Session started for model: cache/unet.onnx
[*] ONNX Session started for model: cache/forecaster.onnx
[*] ONNX Session started for model: cache/classifier.onnx
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 2. Access the Interactive GIS Dashboard
1. Open your web browser and navigate to:
   [http://localhost:8000/](http://localhost:8000/)
2. Log in using the system admin credentials:
   - **Username**: `aegis_admin`
   - **Password**: `aegis_omega_password`
3. Request forecasts, test news classification headers, and monitor the live alert feed.
