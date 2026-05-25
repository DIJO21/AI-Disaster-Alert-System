import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import config
from models.unet import MultiEncoderUNet, HybridDisasterLoss, compute_segmentation_metrics
from models.forecaster import TemporalForecaster, QuantileLoss
from models.classifier import NewsClassifier
from datasets.satellite_dataset import SentinelDataset
from datasets.news_dataset import NewsDataset
from inference.onnx_exporter import AegisONNXExporter

def verify_all():
    print("=== AEGISSPHERE SYSTEM INTEGRATION TEST ===")
    
    # 1. Verify Configuration Paths
    print(f"[*] Config Project Root: {config.PATHS.PROJECT_ROOT}")
    assert config.MODEL.UNET_IN_CHANNELS == 3, "Invalid UNet channel configuration"
    print("[OK] Configurations verified successfully.")

    # 2. Verify Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Testing target hardware: {device}")

    # UNet Verification
    unet = MultiEncoderUNet(in_channels=config.MODEL.UNET_IN_CHANNELS, num_classes=config.MODEL.UNET_CLASSES).to(device)
    dummy_img = torch.randn(2, config.MODEL.UNET_IN_CHANNELS, 256, 256).to(device)
    unet.train()
    outputs = unet(dummy_img)
    assert "out" in outputs and "ds4" in outputs, "UNet training forward pass failed"
    unet.eval()
    outputs_eval = unet(dummy_img)
    assert outputs_eval["out"].shape == (2, config.MODEL.UNET_CLASSES, 256, 256), f"Invalid evaluation shape: {outputs_eval['out'].shape}"
    print("[OK] Multi-Encoder Attention UNet forward pass verified.")

    # Forecaster Verification
    forecaster = TemporalForecaster(input_size=config.MODEL.FORECASTER_INPUT_SIZE).to(device)
    dummy_seq = torch.randn(2, 30, config.MODEL.FORECASTER_INPUT_SIZE).to(device)
    forecaster.eval()
    forecast_out = forecaster(dummy_seq)
    assert forecast_out.shape == (2, 30, 3), f"Invalid forecasting shape: {forecast_out.shape}"
    
    # Check uncertainty MC Dropout
    uncertainty_res = forecaster.forecast_with_uncertainty(dummy_seq, num_mc_runs=5)
    assert "forecast" in uncertainty_res and "uncertainty" in uncertainty_res, "MC Dropout forecast failed"
    print("[OK] Temporal Fusion Transformer Forecaster verified.")

    # Classifier Verification
    classifier = NewsClassifier(num_classes=5, use_lora=True, lora_r=4).to(device)
    dummy_ids = torch.randint(0, 1000, (2, 64)).to(device)
    dummy_mask = torch.ones(2, 64).to(device)
    classifier.eval()
    classifier_out = classifier(dummy_ids, dummy_mask)
    assert classifier_out["logits"].shape == (2, 5), f"Invalid classifier shape: {classifier_out['logits'].shape}"
    print("[OK] Multilingual News Classifier verified.")

    # 3. Verify Datasets
    print("[*] Generating test caches for dataset verification...")
    sat_dataset = SentinelDataset(cache_dir="cache/test_sat", size=5)
    x, y = sat_dataset[0]
    assert x.shape == (15, 256, 256), f"Invalid dataset sample shape: {x.shape}"
    assert y.shape == (4, 256, 256), f"Invalid dataset label shape: {y.shape}"
    print("[OK] Sentinel Geospatial Dataset aligned loader verified.")

    news_dataset = NewsDataset(size=10)
    input_ids, attention_mask, label = news_dataset[0]
    assert input_ids.shape == (128,), f"Invalid news token shape: {input_ids.shape}"
    print("[OK] News NLP Dataset tokenization verified.")

    # 4. Verify ONNX Exporter
    print("[*] Verifying ONNX compilation runtimes...")
    try:
        exporter = AegisONNXExporter(output_dir="cache/test_onnx")
        unet_path = exporter.export_unet(unet)
        assert os.path.exists(unet_path), "UNet ONNX export failed"
        print("[OK] ONNX Exporter pipeline verified successfully.")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[Warning] ONNX export dependencies missing ({e}). Skipping compilation checks (eager inference runtime is still operational).")

    print("\n[SUCCESS] ALL QUALITY GATE TESTS PASSED successfully. Code ready for distribution.")

if __name__ == "__main__":
    verify_all()
