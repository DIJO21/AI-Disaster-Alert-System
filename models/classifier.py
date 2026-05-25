import torch
import torch.nn as nn
import math
from typing import List, Dict, Optional, Any

try:
    from transformers import AutoModelForSequenceClassification, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class LoRALinear(nn.Module):
    """
    Native PyTorch Implementation of Low-Rank Adaptation (LoRA) for a Linear Layer.
    W_updated = W + (B @ A) * (alpha / r)
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # LoRA adapters
        self.lora_A = nn.Parameter(original_linear.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(original_linear.weight.new_zeros((out_features, r)))
        
        self.dropout = nn.Dropout(p=lora_dropout)
        
        # Freeze original linear layer weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
            
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A to Kaiming-uniform and B to zero (so LoRA starts as identity mapping)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original_linear(x)
        
        # Compute adapter modification
        lora_output = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return original_output + lora_output

class NewsClassifier(nn.Module):
    """
    Disaster news classifier based on HuggingFace transformers,
    featuring custom LoRA weight adapter injections.
    """
    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        num_classes: int = 5,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: float = 16.0
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_lora = use_lora
        
        if HAS_TRANSFORMERS:
            self.config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
            self.transformer = AutoModelForSequenceClassification.from_pretrained(
                model_name, config=self.config
            )
        else:
            # Fallback mock transformer layer if transformers library is missing (e.g. local unit tests)
            self.transformer = MockTransformerClassifier(num_classes)
            
        if self.use_lora:
            self.apply_lora(lora_r, lora_alpha)

    def apply_lora(self, r: int, alpha: float):
        """Injects LoRA adapters into target attention projection layers."""
        lora_targets = ["q_lin", "v_lin", "query", "value"]  # common names for attention projections
        replaced_count = 0
        
        # Traverse named modules to find target linear projections
        for name, module in self.transformer.named_modules():
            for target in lora_targets:
                if name.endswith(target) and isinstance(module, nn.Linear):
                    # Find the parent module to assign the LoRALinear replacement
                    parts = name.split('.')
                    parent = self.transformer
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    
                    # Create LoRA linear layer wrapping the original linear layer
                    lora_layer = LoRALinear(module, r=r, lora_alpha=alpha)
                    setattr(parent, parts[-1], lora_layer)
                    replaced_count += 1
                    
        # Freeze other parameters
        for name, param in self.transformer.named_parameters():
            if "lora_" not in name and "classifier" not in name:
                param.requires_grad = False
                
        print(f"[LoRA] Successfully injected {replaced_count} LoRA modules. Non-LoRA modules frozen.")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if HAS_TRANSFORMERS:
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            res = {"logits": outputs.logits}
            if labels is not None:
                res["loss"] = outputs.loss
            return res
        else:
            logits = self.transformer(input_ids)
            res = {"logits": logits}
            if labels is not None:
                res["loss"] = nn.CrossEntropyLoss()(logits, labels)
            return res

class MockTransformerClassifier(nn.Module):
    """Fallback classifier for testing without HF installed."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(30000, 128)
        # Mocking attention query & value linear layers
        self.q_lin = nn.Linear(128, 128)
        self.v_lin = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple pooling over sentence length
        embeds = self.embedding(x)
        # Apply projection layers
        q = self.q_lin(embeds)
        v = self.v_lin(embeds)
        # Pooling
        pooled = (q + v).mean(dim=1)
        return self.classifier(pooled)
