import torch
import torch.nn as nn


def score_function(A: torch.Tensor, eta: float=0.01) -> int:
    """
    Computes the score r(A) = min i s.t. sum_{j=i}^d lambda_j / sum_{j=1}^d lambda_j <= eta,
    where lambda_j are the singular values of A sorted in descending order.

    Args:
        A (torch.Tensor): 2D tensor (matrix)
        eta (float): threshold in (0,1)

    Returns:
        int: the minimum index i (1-based) satisfying the condition
    """
    # Compute singular values (descending order)
    # Note: torch.linalg.svdvals returns singular values sorted descending by default
    singular_vals = torch.linalg.svdvals(A).cpu()

    total_sum = singular_vals.sum()
    d = singular_vals.shape[0]

    # Compute tail sums from i to d
    # We'll iterate i from 1 to d (1-based indexing)
    # Compute cumulative sums from the end for efficiency:
    tail_sums = torch.flip(torch.cumsum(torch.flip(singular_vals, dims=[0]), dim=0), dims=[0])

    # Find minimal i (1-based) such that tail_sums[i-1]/total_sum <= eta
    for i in range(d):
        if tail_sums[i] / total_sum <= eta:
            return i + 1  # return 1-based index

    # If not found, return d (worst case)
    return d

class FirstOrderScorer:
    def __init__(self, model: nn.Module, target_blocks: list[str]):
        self.model = model
        self.target_blocks = target_blocks  # e.g., ['blocks.0.mlp', 'blocks.1.attn']
        self.activations = {}
        self.grad_outputs = {}

    def _get_module_by_name(self, name):
        """Get a module by its name using recursive attribute access."""
        # Support dotted paths like "blocks.0.mlp"
        module = self.model
        for attr in name.split('.'):
            module = getattr(module, attr)
        return module

    def _save_input(self, name):
        """Forward hook to save the input activations of a module."""
        def hook(module, input, output):
            self.activations[name] = input[0].detach()
        return hook

    def _save_grad_output(self, name):
        """Backward hook to save the gradient of the output of a module."""
        def hook(module, grad_input, grad_output):
            self.grad_outputs[name] = grad_output[0].detach()
        return hook

    def register_hooks(self):
        """Register all necessary hooks for the target blocks."""
        self.handles = []
        for name in self.target_blocks:
            mod = self._get_module_by_name(name)
            # Save the input during forward pass
            self.handles.append(mod.register_forward_hook(self._save_input(name)))
             # Save the gradient of the output during backward pass
            self.handles.append(mod.register_full_backward_hook(self._save_grad_output(name)))

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()

    def compute_scores(self, 
                       criterion,
                       data_loader, 
                       device, 
                       num_samples=512
                       ):
        self.model.eval()
        self.register_hooks()

        # Freeze all but necessary
        for p in self.model.parameters():
            p.requires_grad = False

        for name in self.target_blocks:
            module = self._get_module_by_name(name)
            for param_name, param in module.named_parameters():
                if "weight" in param_name:
                    param.requires_grad = True
                    
        self.model.zero_grad()
        count = 0

        # Accumulators for each module
        act_sum = {name: 0 for name in self.target_blocks}
        seen = {name: 0 for name in self.target_blocks}

        # Iterate over the data loader
        for batch in data_loader:
            x_batch, y_batch = batch["x"], batch["y"]

            for i in range(x_batch.size(0)):
                x = x_batch[i].unsqueeze(0).to(device)
                y = y_batch[i].unsqueeze(0).to(device)

                self.model.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()

                for name in self.target_blocks:
                    x_i = self.activations.get(name)
                    dy = self.grad_outputs.get(name)
                    if x_i is not None and dy is not None:
                        act_sum[name] += x_i.squeeze(0).T @ dy.squeeze(0)
                        seen[name] += 1
                    else:
                        print(f"Warning: Activation or gradient output for {name} is None.")

                count += 1
                if count >= num_samples:
                    break
            if count >= num_samples:
                break

        # Compute approximate scores
        scores = {}
        for name in self.target_blocks:
            if seen[name] > 0:
                agg = act_sum[name] / seen[name]
                scores[name] = score_function(agg)
            else:
                scores[name] = 0

        self.remove_hooks()
        return scores