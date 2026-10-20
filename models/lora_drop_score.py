import torch
import torch.nn as nn

class LoraImportanceScorer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.lora_outputs = {}
        self.g_scores = {}
        self._output_hooks = []

    def _register_hooks(self):
        """Register hooks to capture the output of LoRA modules."""

        def make_hook(name):
            def hook(module, input, output):
                value = (output.detach() ** 2).sum().item()
                if name not in self.g_scores:
                    self.g_scores[name] = 0.0
                self.g_scores[name] += value
            return hook

        for name, module in self.model.named_modules():
            if hasattr(module, "lora_B") and "default" in module.lora_B and "patch_embed" not in name:
                hook = module.lora_B["default"].register_forward_hook(make_hook(name))
                self._output_hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._output_hooks:
            hook.remove()
        self._output_hooks = []

    def compute(
            self, 
            dataloader, 
            device):
        
        self.model.eval()
        self._register_hooks()

        with torch.no_grad():
            for batch in dataloader:
                x = batch["x"].to(device)
                _ = self.model(x)

        self._remove_hooks()
        total = sum(self.g_scores.values())
        importance_scores = {name: score / total for name, score in self.g_scores.items()}
        return importance_scores
