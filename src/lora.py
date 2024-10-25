import torch
import torch.nn as nn

from peft import get_peft_model


def setup_peft_esm2(peft_config, no_lora = False, regular_ft=False):

    from transformers import EsmModel, EsmTokenizer

    # Load the pretrained ESM-2 model
    esm_model = EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D')
    esm_tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')

    if regular_ft:
        return esm_model, esm_tokenizer

    # Apply LoRA to the model
    peft_lm = get_peft_model(esm_model, peft_config)

    #NOT APPLYING LoRA to the model:
    if no_lora:
        for name, param in esm_model.named_parameters():
            param.requires_grad = False
        return esm_model, esm_tokenizer

    # freeze all the layers except the LoRA adapter matrices
    for name, param in peft_lm.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.required_grad = False

    return peft_lm, esm_tokenizer

def setup_peft_esm3(peft_config, no_lora = False):

    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
    from esm.tokenization import EsmSequenceTokenizer

    # Load the pretrained ESM-3 model
    esm3_model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1")
    esm3_tokenizer = EsmSequenceTokenizer()

    #NOT APPLYING LoRA to the model:
    if no_lora:
        for name, param in esm3_model.named_parameters():
            param.requires_grad = False
        return esm3_model, esm3_tokenizer
    
    # Apply LoRA to the model
    peft_lm = get_peft_model(esm3_model, peft_config)

    # freeze all the layers except the LoRA adapter matrices
    for name, param in esm3_model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.required_grad = False

    return esm3_model, esm3_tokenizer

def setup_peft_ablang(peft_config, chain="H"):

    from transformers import AutoTokenizer, AutoModelForMaskedLM

    if chain == "H":
        # Load the pretrained AbLang H model
        ablang_tokenizer = AutoTokenizer.from_pretrained("qilowoq/AbLang_heavy", trust_remote_code=True)
        ablang_model = AutoModelForMaskedLM.from_pretrained("qilowoq/AbLang_heavy", trust_remote_code=True)

    if chain == "L":
        # Load the pretrained AbLang L model
        ablang_tokenizer = AutoTokenizer.from_pretrained("qilowoq/AbLang_light", trust_remote_code=True)
        ablang_model = AutoModelForMaskedLM.from_pretrained("qilowoq/AbLang_light", trust_remote_code=True)

    # take out the decoder layer, which we don't need
    ablang_model = ablang_model.roberta

    # Apply LoRA to the model
    peft_lm = get_peft_model(ablang_model, peft_config)

    # freeze all the layers except the LoRA adapter matrices
    for name, param in peft_lm.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.required_grad = False

    return peft_lm, ablang_tokenizer

def setup_peft_ablang2(peft_config, receptor_type='BCR', device='cpu', no_lora=False):
    import ablang2

    # Load the pretrained AbLang2 model
    if receptor_type == 'TCR':
        ablang2_module = ablang2.pretrained(model_to_use='tcrlang-paired', random_init=False, device=device)
    elif receptor_type == 'BCR':
        ablang2_module = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, device=device)
    else:
        raise ValueError(f"Receptor type {receptor_type} not supported")
    ablang2_model = ablang2_module.AbRep

    # NOT APPLYING LoRA to the model:
    if no_lora:
        for name, param in ablang2_model.named_parameters():
            param.requires_grad = False
        return ablang2_model, ablang2_module.tokenizer

    # Apply LoRA to the model
    peft_lm = get_peft_model(ablang2_model, peft_config)

    # freeze all the layers except the LoRA adapter matrices
    lora_count = 0
    for name, param in ablang2_model.named_parameters():
        if "lora" in name:
            lora_count += 1
            param.requires_grad = True
        else:
            param.required_grad = False
    assert lora_count >= 4 # make sure we have LoRA adapter matrices

    return ablang2_model, ablang2_module.tokenizer

def setup_peft_aberta2(peft_config):
    from transformers import (
        RoFormerForMaskedLM, 
        RoFormerTokenizer, 
    )

    # Load the pretrained Aberta2 model
    aberta2_model = RoFormerForMaskedLM.from_pretrained("alchemab/antiberta2")
    aberta2_tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2")

    # only take the RoFormer module:
    aberta2_model = aberta2_model.roformer

    # Apply LoRA to the model
    peft_lm = get_peft_model(aberta2_model, peft_config)

    # freeze all the layers except the LoRA adapter matrices
    for name, param in peft_lm.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.required_grad = False
    
    return peft_lm, aberta2_tokenizer

def setup_peft_tcrbert(peft_config, no_lora=False, regular_ft=False):
    from transformers import (
        BertModel,
        AutoTokenizer,
    )

    # Load the pretrained TCRBert model
    tcrbert_model = BertModel.from_pretrained("wukevin/tcr-bert-mlm-only")
    tcrbert_tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert-mlm-only", trust_remote_code=True)

    if regular_ft:
        return tcrbert_model, tcrbert_tokenizer

    # Apply LoRA to the model
    peft_lm = get_peft_model(tcrbert_model, peft_config)

    # NOT APPLYING LoRA to the model:
    if no_lora:
        for name, param in tcrbert_model.named_parameters():
            param.requires_grad = False
        return tcrbert_model, tcrbert_tokenizer

    # freeze all the layers except the LoRA adapter matrices
    for name, param in peft_lm.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.required_grad = False

    return peft_lm, tcrbert_tokenizer

def setup_peft_inhouse(peft_config, no_lora=False, model_ckpt_path=None):
    from .pretrain.model import CdrBERT, getCdrTokenizer, MODEL_CONFIG

    # load the in-house TCR model:
    inhouse_tokenizer = getCdrTokenizer()
    inhouse_model = CdrBERT(MODEL_CONFIG, inhouse_tokenizer)
    inhouse_ckpt = torch.load(model_ckpt_path)
    # Remove "model." prefix from keys. Artifact of Pytorch Lightning
    new_state_dict = {}
    for key, value in inhouse_ckpt['state_dict'].items():
        new_key = key.replace("model.", "")
        new_state_dict[new_key] = value
    inhouse_model.load_state_dict(new_state_dict)

    # Apply LoRA to the model
    peft_lm = get_peft_model(inhouse_model, peft_config)

    # NOT APPLYING LoRA to the model:
    if no_lora:
        for name, param in inhouse_model.named_parameters():
            param.requires_grad = False
        return inhouse_model, inhouse_tokenizer
    
    # freeze all the layers except the LoRA adapter matrices
    for name, param in peft_lm.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.required_grad = False
    
    return peft_lm, inhouse_tokenizer