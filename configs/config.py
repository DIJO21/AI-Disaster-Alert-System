import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings

class PathSettings(BaseSettings):
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    BACKEND_DIR: Path = PROJECT_ROOT / "backend"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
    CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    CACHE_DIR: Path = PROJECT_ROOT / "cache"

    class Config:
        env_prefix = "AEGIS_"

class ModelSettings(BaseSettings):
    # UNet
    UNET_ENCODER: str = "resnet50"
    UNET_IN_CHANNELS: int = 3  # Matches Colab simplified channels (RGB/SAR)
    UNET_CLASSES: int = 1       # Matches Colab single binary hazard mask
    
    # Forecaster (Temporal Fusion Transformer-style)
    FORECASTER_INPUT_SIZE: int = 10  # Matches Colab dummy forecaster input dim
    FORECASTER_HIDDEN_SIZE: int = 64
    FORECASTER_NUM_ATTENTION_HEADS: int = 4
    FORECASTER_DROPOUT: float = 0.1
    FORECASTER_QUANTILE_OUTPUTS: List[float] = [0.1, 0.5, 0.9]
    
    # News Classifier
    CLASSIFIER_BACKBONE: str = "distilbert-base-multilingual-cased"
    CLASSIFIER_NUM_CLASSES: int = 5  # Earthquake, Flood, Wildfire, Hurricane, Other

    class Config:
        env_prefix = "AEGIS_MODEL_"

class TrainingSettings(BaseSettings):
    EPOCHS: int = 100
    BATCH_SIZE: int = 8
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-2
    GRADIENT_ACCUMULATION_STEPS: int = 4
    MIXED_PRECISION: bool = True
    EARLY_STOPPING_PATIENCE: int = 10
    USE_SWA: bool = True
    EMA_DECAY: float = 0.999

    class Config:
        env_prefix = "AEGIS_TRAIN_"

class APISettings(BaseSettings):
    COPENICUS_CLIENT_ID: Optional[str] = os.getenv("COPENICUS_CLIENT_ID")
    COPENICUS_CLIENT_SECRET: Optional[str] = os.getenv("COPENICUS_CLIENT_SECRET")
    NASA_POWER_API_URL: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
    OPENWEATHER_API_KEY: Optional[str] = os.getenv("OPENWEATHER_API_KEY")
    KAGGLE_USERNAME: Optional[str] = os.getenv("KAGGLE_USERNAME")
    KAGGLE_KEY: Optional[str] = os.getenv("KAGGLE_KEY")

    class Config:
        env_prefix = "AEGIS_API_"

class SecuritySettings(BaseSettings):
    JWT_SECRET: str = "super_secure_aegissphere_token_secret_key_129847192847"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    RATE_LIMIT_CALLS: int = 100
    RATE_LIMIT_PERIOD_SEC: int = 60

    class Config:
        env_prefix = "AEGIS_SEC_"

class AegisConfig(BaseSettings):
    PATHS: PathSettings = PathSettings()
    MODEL: ModelSettings = ModelSettings()
    TRAIN: TrainingSettings = TrainingSettings()
    API: APISettings = APISettings()
    SEC: SecuritySettings = SecuritySettings()

    class Config:
        env_nested_delimiter = "__"

# Ensure crucial directories exist
config = AegisConfig()
for path_attr in ["CHECKPOINTS_DIR", "LOGS_DIR", "CACHE_DIR"]:
    getattr(config.PATHS, path_attr).mkdir(parents=True, exist_ok=True)
