import timm
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

def get_timm_model(
    model_name,
    num_classes,
    adaptation_mode="",
    lora_r=8,
    lora_target_modules=None,
    lora_modules_to_save=None,
    lora_alpha = 1,
    lora_drop_adapters_path=None,
):
    # Initialize the base model
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    # Apply LoRA if specified
    if adaptation_mode == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            modules_to_save=lora_modules_to_save,
        )

        model = get_peft_model(model, config)
        model.peft_config["default"].base_model_name_or_path = model_name

        if lora_drop_adapters_path:
            print(f"[INFO] 🔄 Loading LoRA adapters from {lora_drop_adapters_path}")
            adapter_name = model.active_adapters[0]
            model.load_adapter(lora_drop_adapters_path, adapter_name, is_trainable=True)

    # If specified, freeze all layers
    elif adaptation_mode == "head_only":
        for param in model.parameters():
            param.requires_grad = False

    elif adaptation_mode != "full":
        raise ValueError(f"Received an unexpected adaptation mode:{adaptation_mode}")

    # Always unfreeze the parameters of the classification head
    # For transformers
    if hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True
    # For resnets
    elif hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True

    return model
