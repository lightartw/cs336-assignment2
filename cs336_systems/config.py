from pydantic import BaseModel

class ModelConfig(BaseModel):
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float

class OptimizerConfig(BaseModel):
    lr: float
    beta1: float
    beta2: float
    weight_decay: float
    eps: float

class TrainingConfig(BaseModel):
    batch_size: int
    max_norm: float    # gradient_clipping 需要的阈值
    precision: str

class Config(BaseModel):
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            return cls.model_validate_json(content)
        except Exception as e:
            raise ValueError(f"{e}") 