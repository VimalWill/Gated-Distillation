"""
DEL (Deletion by Example Localization) unlearning method.

Reference: "Improved Localized Machine Unlearning Through the Lens of Memorization"
by Torkzadehmahani et al.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional


class DELUnlearning:
    """Deletion by Example Localization for machine unlearning."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.criticality_scores: Optional[Dict[str, torch.Tensor]] = None

    def compute_criticality_scores(
        self,
        forget_loader,
        layer_names: Optional[List[str]] = None,
        top_h: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-output-channel criticality: mean of top-h |θ·∇ℓ| scores per channel.

        Paper Algorithm 1 (Appendix A.5):
            s_j = |θ_j · g_j(θ, S)|
            c_oi = (1/h) * Σ top-h s_i[j] per output channel
        """
        if layer_names is None:
            layer_names = self._get_target_layers()

        # Pre-allocate accumulators with correct shapes (avoids defaultdict broadcast footgun)
        score_accum: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters()
            if any(ln in name for ln in layer_names)
        }

        # D1 fix: forward+backward outside torch.no_grad() so .grad is populated
        self.model.eval()
        for batch in forget_loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            self.model.zero_grad()
            outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss.backward()

            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in score_accum and param.grad is not None:
                        score_accum[name].add_(torch.abs(param.data * param.grad))

        self.model.zero_grad()

        # D3 fix: aggregate to output-channel level using mean of top-h scores per channel
        criticality: Dict[str, torch.Tensor] = {}
        for name, scores in score_accum.items():
            if scores.dim() >= 2:
                out_ch = scores.shape[0]
                flat = scores.view(out_ch, -1)             # (out_ch, in_feats)
                h = min(top_h, flat.shape[1])
                top_vals, _ = torch.topk(flat, h, dim=1)
                criticality[name] = top_vals.mean(dim=1)  # (out_ch,)
            else:
                criticality[name] = scores                 # bias: per-element

        self.criticality_scores = criticality
        return criticality

    def generate_mask(
        self,
        criticality_scores: Dict[str, torch.Tensor],
        budget_alpha: float = 0.25,
    ) -> Dict[str, torch.Tensor]:
        """
        Select top-budget output channels by criticality, mark their full rows.

        Budget is fraction of total output channels, not individual weights (D3 fix).
        """
        total_channels = sum(s.numel() for s in criticality_scores.values())
        budget = int(budget_alpha * total_channels)

        all_channels = [
            (criticality_scores[name][ch].item(), name, ch)
            for name in criticality_scores
            for ch in range(criticality_scores[name].numel())
        ]
        all_channels.sort(key=lambda x: x[0], reverse=True)
        selected = {(name, ch) for _, name, ch in all_channels[:budget]}

        masks: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if name not in criticality_scores:
                continue
            mask = torch.zeros_like(param, dtype=torch.bool, device=self.device)
            for ch in range(param.shape[0]):
                if (name, ch) in selected:
                    mask[ch] = True  # marks entire row for 2-D; element for 1-D
            masks[name] = mask

        return masks

    def reset_parameters(self, mask: Dict[str, torch.Tensor]):
        """
        Reset masked entries to the layer's own init distribution (D6 fix).

        Uses Kaiming-uniform for Linear weights and matching bias bound, not randn*1.0.
        """
        param_to_module: Dict[str, nn.Module] = {}
        for mname, module in self.model.named_modules():
            for pname, _ in module.named_parameters(recurse=False):
                full = f"{mname}.{pname}" if mname else pname
                param_to_module[full] = module

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in mask or not mask[name].any():
                    continue
                m = mask[name]
                init_vals = torch.empty_like(param)
                module = param_to_module.get(name)

                if isinstance(module, nn.Linear) and param.dim() == 2:
                    # PyTorch default Linear weight init
                    nn.init.kaiming_uniform_(init_vals, a=math.sqrt(5))
                elif isinstance(module, nn.Linear) and param.dim() == 1:
                    # PyTorch default Linear bias init
                    bound = 1.0 / math.sqrt(module.in_features) if module.in_features > 0 else 0
                    nn.init.uniform_(init_vals, -bound, bound)
                else:
                    nn.init.kaiming_uniform_(init_vals, a=math.sqrt(5))

                param.data[m] = init_vals[m]

    def finetune_masked_params(
        self,
        retain_loader,
        mask: Dict[str, torch.Tensor],
        learning_rate: float = 0.01,
        epochs: int = 1,
        optimizer_class=torch.optim.Adam,
    ):
        """
        Finetune only masked entries on the retain set.

        D5 fix: after backward, zero gradient at unmasked positions inside masked
        parameters so those entries are not updated by optimizer.step().
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = name in mask and bool(mask[name].any())

        optimizer = optimizer_class(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in retain_loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                optimizer.zero_grad()
                outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss.backward()

                # D5 fix: zero gradient at unmasked positions within masked parameters
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in mask:
                        param.grad[~mask[name]] = 0.0

                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(retain_loader):.4f}")

        for param in self.model.parameters():
            param.requires_grad = True

    def unlearn(
        self,
        forget_loader,
        retain_loader,
        budget_alpha: float = 0.25,
        top_h: int = 5,
        finetune_lr: float = 0.01,
        finetune_epochs: int = 1,
    ):
        """Full DEL pipeline: localize → reset → finetune."""
        print("[DEL] Computing criticality scores...")
        self.compute_criticality_scores(forget_loader, top_h=top_h)

        print(f"[DEL] Generating channel mask (α={budget_alpha})...")
        mask = self.generate_mask(self.criticality_scores, budget_alpha)
        n_params = int(sum(m.sum().item() for m in mask.values()))
        print(f"[DEL] Resetting {n_params} parameters...")
        self.reset_parameters(mask)

        print("[DEL] Finetuning masked parameters on retain set...")
        self.finetune_masked_params(
            retain_loader, mask, learning_rate=finetune_lr, epochs=finetune_epochs
        )
        print("[DEL] Unlearning complete")
        return mask

    def _get_target_layers(self) -> List[str]:
        """Attention and FFN parameter names (D4 fix: include MLP/FFN, not just attention)."""
        keywords = [
            "attention", "self_attn",
            "q_proj", "k_proj", "v_proj", "o_proj",
            "mlp", "dense", "fc1", "fc2",
            "feed_forward", "intermediate",
        ]
        return [
            name
            for name, _ in self.model.named_parameters()
            if any(kw in name.lower() for kw in keywords)
        ]
