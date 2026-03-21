from pydantic import BaseModel

class ModelConfig(BaseModel):
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float

class TrainingConfig(BaseModel):
    batch_size: int
    precision: str
    device: str = "cuda"

class Config(BaseModel):
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            return cls.model_validate_json(content)
        except Exception as e:
            raise ValueError(f"{e}") 