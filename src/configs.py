from dataclasses import dataclass
import os

from peft import LoraConfig, TaskType

output_dir_path = os.getenv('WANDB_OUTPUT_DIR')

@dataclass
class LightningConfig:
    max_epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    num_workers_train: int = 8
    torch_device: str = 'gpu'
    dataset_path: str = None
    output_dir: str = output_dir_path
    include_mhc: bool = False
    mhc_groove_only: bool = False
    unique_epitopes: bool = False
    mask_seqs: bool = False
    mask_prob: float = 0.15
    swe_pooling: bool = False
    save_embed_path: str = None
    no_lora: bool = False
    mse_weight: float = 0.
    weigh_epitope_count: bool = False
    oversample: bool = False
    regular_ft: bool = False
    fewshot_ratio: float = None
    lr_scheduler: str = 'cos_anneal'


@dataclass
class EncoderProjectionConfigAbLang:
    epitope_input_dim: int = 1280
    receptor_input_dim: int = 1536
    projection_dim: int = 512
    temperature: float = 0.07
    receptor_model_name: str = 'ablang'

@dataclass
class EncoderProjectionConfigAbLang2:
    epitope_input_dim: int = 1280
    receptor_input_dim: int = 480
    projection_dim: int = 512
    temperature: float = 0.07
    receptor_model_name: str = 'ablang2'

@dataclass
class EncoderProjectionConfigAntiberta2:
    epitope_input_dim: int = 1280
    receptor_input_dim: int = 1024
    projection_dim: int = 512
    temperature: float = 0.07
    receptor_model_name: str = 'antiberta2'

@dataclass
class EncoderProjectionConfigTCRBert:
    epitope_input_dim: int = 1280
    receptor_input_dim: int = 1536
    hidden_dim: int = None
    projection_dim: int = 512
    temperature: float = 0.07
    receptor_model_name: str = 'tcrbert'

@dataclass
class EncoderProjectionConfigTCRLang:
    epitope_input_dim: int = 1280
    receptor_input_dim: int = 480
    hidden_dim: int = None
    projection_dim: int = 512
    temperature: float = 0.07
    receptor_model_name: str = 'tcrlang'

@dataclass
class EncoderProjectionConfigESM2:
    epitope_input_dim: int = 1280
    receptor_input_dim: int = 1280
    hidden_dim: int = None
    projection_dim: int = None
    temperature: float = 0.07
    receptor_model_name: str = 'esm2'

@dataclass
class EncoderProjectionConfigESM3:
    epitope_input_dim: int = 1536
    receptor_input_dim: int = 1536
    hidden_dim: int = None
    projection_dim: int = None
    temperature: float = 0.07
    receptor_model_name: str = 'esm3'

@dataclass
class EncoderProjectionConfigInHouse:
    epitope_input_dim: int = 1280
    receptor_input_dim: int = 768
    hidden_dim: int = None
    projection_dim: int = None
    temperature: float = 0.07
    receptor_model_name: str = 'inhouse'

@dataclass
class EncoderProjectionConfigOneHot:
    epitope_input_dim: int = 21
    receptor_input_dim: int = 21
    hidden_dim: int = None
    projection_dim: int = None
    temperature: float = 0.07
    receptor_model_name: str = 'inhouse'

# --------------------------------------------------
# PEFT configs:
peft_config_esm2 = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias='none',
    layers_to_transform=[32, 31, 30, 29, 28, 27, 26, 25],
    task_type=TaskType.FEATURE_EXTRACTION,
    target_modules=['attention.self.key', 'attention.self.value']
)

peft_config_esm3 = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias='none',
    layers_to_transform=[47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36],
    task_type=TaskType.FEATURE_EXTRACTION,
    target_modules=['attn.layernorm_qkv.1']
)

peft_config_ablang = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias='none',
    layers_to_transform=[11, 10, 9, 8],
    task_type=TaskType.FEATURE_EXTRACTION,
    target_modules=['attention.self.query', 'attention.self.value']
)

peft_config_ablang2 = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias='none',
    # layers_to_transform=[11, 10, 9, 8],
    task_type=TaskType.FEATURE_EXTRACTION,
    target_modules=".*(8|9|10|11).*[kv]_proj$"
)

peft_config_aberta2 = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias='none',
    layers_to_transform=[15, 14, 13, 12],
    task_type=TaskType.FEATURE_EXTRACTION,
    target_modules=["attention.self.query", "attention.self.value"]
)

peft_config_tcrbert = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias='none',
    layers_to_transform=[11, 10, 9, 8],
    task_type=TaskType.FEATURE_EXTRACTION,
    target_modules=["attention.self.key", "attention.self.value"]
)

peft_config_inhouse = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias='none',
    layers_to_transform=[11, 10, 9, 8],
    task_type=TaskType.FEATURE_EXTRACTION,
    target_modules=["attention.self.key", "attention.self.value"]
)
# --------------------------------------------------


def get_lightning_config(name='default'):
    if name == 'default':
        return LightningConfig()

def get_projection_config(name='ablang'):
    if name == 'ablang':
        return EncoderProjectionConfig()
    elif name == 'ablang2':
        return EncoderProjectionConfigAbLang2()
    elif name == 'antiberta2':
        return EncoderProjectionConfigAntiberta2()
    elif name == 'tcrbert':
        return EncoderProjectionConfigTCRBert()
    elif name == 'tcrlang':
        return EncoderProjectionConfigTCRLang()
    elif name == 'esm2':
        return EncoderProjectionConfigESM2()
    elif name == 'esm3':
        return EncoderProjectionConfigESM3()
    elif name == 'inhouse':
        return EncoderProjectionConfigInHouse()
    elif name == 'onehot':
        return EncoderProjectionConfigOneHot()
    else:
        raise ValueError(f"Invalid model name: {name}")


def build_lora_config(rank=4, alpha=32, dropout=0.1, bias='none', layers_to_transform=None, 
                      task_type=TaskType.FEATURE_EXTRACTION, target_modules=None):
    
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        layers_to_transform=layers_to_transform,
        task_type=task_type,
        target_modules=target_modules
    )
