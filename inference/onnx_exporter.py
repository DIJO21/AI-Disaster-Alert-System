import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# Import model structures
from models.unet import MultiEncoderUNet
from models.forecaster import TemporalForecaster
from models.classifier import NewsClassifier
from configs.config import config

class AegisONNXExporter:
    """Exports PyTorch disaster intelligence models to highly optimized ONNX binaries."""
    def __init__(self, output_dir: str = "cache"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_unet(self, model: MultiEncoderUNet) -> str:
        model.eval()
        output_path = os.path.join(self.output_dir, "unet.onnx")
        
        # Mock inputs matching UNET_IN_CHANNELS (e.g. 15, shape B x C x H x W)
        dummy_input = torch.randn(1, config.MODEL.UNET_IN_CHANNELS, 256, 256, dtype=torch.float32)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_tensor"],
            output_names=["output_mask"],
            dynamic_axes={
                "input_tensor": {0: "batch_size", 2: "height", 3: "width"},
                "output_mask": {0: "batch_size", 2: "height", 3: "width"}
            }
        )
        print(f"[*] MultiEncoderUNet successfully exported to ONNX format: {output_path}")
        return output_path

    def export_forecaster(self, model: TemporalForecaster) -> str:
        model.eval()
        output_path = os.path.join(self.output_dir, "forecaster.onnx")
        
        # Input size matches FORECASTER_INPUT_SIZE, sequence length typical 30 days
        dummy_input = torch.randn(1, 30, config.MODEL.FORECASTER_INPUT_SIZE, dtype=torch.float32)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_sequence"],
            output_names=["quantile_predictions"],
            dynamic_axes={
                "input_sequence": {0: "batch_size", 1: "sequence_length"},
                "quantile_predictions": {0: "batch_size", 1: "sequence_length"}
            }
        )
        print(f"[*] TemporalForecaster successfully exported to ONNX format: {output_path}")
        return output_path

    def export_classifier(self, model: NewsClassifier) -> str:
        model.eval()
        output_path = os.path.join(self.output_dir, "classifier.onnx")
        
        # DistilBERT expects input_ids and attention_mask
        dummy_ids = torch.randint(0, 20000, (1, 128), dtype=torch.long)
        dummy_mask = torch.ones(1, 128, dtype=torch.long)
        
        torch.onnx.export(
            model,
            (dummy_ids, dummy_mask),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"}
            }
        )
        print(f"[*] NewsClassifier successfully exported to ONNX format: {output_path}")
        return output_path

class AegisInferenceSession:
    """ONNX-Runtime inference wrapper with fallback to PyTorch eager evaluation."""
    def __init__(self, onnx_model_path: str, fallback_pytorch_model: Optional[nn.Module] = None):
        self.onnx_model_path = onnx_model_path
        self.pytorch_model = fallback_pytorch_model
        self.ort_session = None
        
        if HAS_ONNX and os.path.exists(onnx_model_path):
            try:
                # Prefer CUDA or TensorRT execution providers, falling back to CPU
                providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
                self.ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
                print(f"[*] ONNX Session started for model: {onnx_model_path} via execution providers: {self.ort_session.get_providers()}")
            except Exception as e:
                print(f"[Warning] Failed to instantiate ONNX session: {e}. Falling back to PyTorch model.")
        else:
            print(f"[Info] ONNX model binary not found or library missing. Operating under PyTorch eager fallback.")

    def run_unet(self, x: np.ndarray) -> np.ndarray:
        """
        Runs image segmentation.
        Input x: numpy array of shape (B, C, H, W)
        """
        if self.ort_session is not None:
            # Dynamically resolve input name
            input_name = self.ort_session.get_inputs()[0].name
            ort_inputs = {input_name: x.astype(np.float32)}
            ort_outs = self.ort_session.run(None, ort_inputs)
            return ort_outs[0]
            
        if self.pytorch_model is not None:
            self.pytorch_model.eval()
            with torch.no_grad():
                tensor_x = torch.from_numpy(x).float()
                # Run standard forward
                out_dict = self.pytorch_model(tensor_x)
                if isinstance(out_dict, dict):
                    return out_dict["out"].cpu().numpy()
                return out_dict.cpu().numpy()
                
        raise RuntimeError("No executable model session configured.")

    def run_forecaster(self, x: np.ndarray) -> np.ndarray:
        """
        Runs temporal forecaster.
        Input x: numpy array of shape (B, S, Input_Size)
        """
        if self.ort_session is not None:
            input_name = self.ort_session.get_inputs()[0].name
            ort_inputs = {input_name: x.astype(np.float32)}
            ort_outs = self.ort_session.run(None, ort_inputs)
            return ort_outs[0]
            
        if self.pytorch_model is not None:
            self.pytorch_model.eval()
            with torch.no_grad():
                tensor_x = torch.from_numpy(x).float()
                return self.pytorch_model(tensor_x).cpu().numpy()
                
        raise RuntimeError("No executable model session configured.")

    def run_classifier(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Runs news text classifier.
        """
        if self.ort_session is not None:
            inputs = self.ort_session.get_inputs()
            if len(inputs) == 1:
                # Target dummy model: maps a single input
                ort_inputs = {inputs[0].name: input_ids.astype(np.float32)}
            else:
                # Target standard model: maps input_ids + attention_mask
                ort_inputs = {
                    inputs[0].name: input_ids.astype(np.int64),
                    inputs[1].name: attention_mask.astype(np.int64)
                }
            ort_outs = self.ort_session.run(None, ort_inputs)
            return ort_outs[0]
            
        if self.pytorch_model is not None:
            self.pytorch_model.eval()
            with torch.no_grad():
                ids_tensor = torch.from_numpy(input_ids).long()
                mask_tensor = torch.from_numpy(attention_mask).long()
                out = self.pytorch_model(ids_tensor, mask_tensor)
                if isinstance(out, dict):
                    return out["logits"].cpu().numpy()
                return out.cpu().numpy()
                
        raise RuntimeError("No executable model session configured.")
