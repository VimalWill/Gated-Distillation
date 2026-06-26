"""
DEL (Deletion by Example Localization) unlearning method.

Reference: "Improved Localized Machine Unlearning Through the Lens of Memorization"
by Torkzadehmahani et al.

Main contribution: Leverages memorization to identify critical parameters at channel level
using weighted gradients (θ * g), then performs Reset+Finetune unlearning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class DELUnlearning:
    """Deletion by Example Localization for machine unlearning."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize DEL unlearning.

        Args:
            model: The model to unlearn from
            device: Device to run computations on
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.criticality_scores = None

    def compute_criticality_scores(
        self,
        forget_loader,
        layer_names: Optional[List[str]] = None,
        top_h: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted gradient criticality scores for each parameter.

        Args:
            forget_loader: DataLoader with examples to forget
            layer_names: Specific layers to compute scores for (None = all attention layers)
            top_h: Number of top scores per channel to average

        Returns:
            Dictionary mapping parameter names to criticality scores
        """
        criticality = defaultdict(lambda: torch.zeros(1, device=self.device))

        # Identify target layers (attention layers by default)
        if layer_names is None:
            layer_names = self._get_attention_layers()

        self.model.eval()
        with torch.no_grad():
            for batch in forget_loader:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Compute loss for gradient
                outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss.backward()

        # Compute weighted gradient scores: |θ * ∇ℓ|
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is None or not any(ln in name for ln in layer_names):
                    continue

                # Weighted gradient criticality
                scores = torch.abs(param.data * param.grad)
                criticality[name] = scores

        self.criticality_scores = dict(criticality)
        return self.criticality_scores

    def generate_mask(
        self, criticality_scores: Dict[str, torch.Tensor], budget_alpha: float = 0.25
    ) -> Dict[str, torch.Tensor]:
        """
        Generate binary mask for parameters to unlearn based on criticality scores.

        Args:
            criticality_scores: Dict of criticality scores per parameter
            budget_alpha: Fraction of parameters to mark for unlearning (0.2-0.3 optimal)

        Returns:
            Dictionary of binary masks for each parameter
        """
        masks = {}
        total_params = sum(s.numel() for s in criticality_scores.values())
        budget = int(budget_alpha * total_params)

        # Flatten all scores to find global top-k
        all_scores = []
        score_to_param = {}
        for name, scores in criticality_scores.items():
            flat_scores = scores.flatten()
            for idx, score in enumerate(flat_scores):
                all_scores.append((score.item(), name, idx))
                score_to_param[(name, idx)] = score

        # Sort by score and select top budget items
        all_scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = set((name, idx) for _, name, idx in all_scores[:budget])

        # Create masks
        for name, scores in criticality_scores.items():
            mask = torch.zeros_like(scores, dtype=torch.bool, device=self.device)
            flat_mask = mask.flatten()
            for idx in range(scores.numel()):
                if (name, idx) in top_indices:
                    flat_mask[idx] = True
            masks[name] = mask

        return masks

    def reset_parameters(self, mask: Dict[str, torch.Tensor], noise_scale: float = 1.0):
        """
        Reset parameters marked by mask to random initialization.

        Args:
            mask: Binary mask indicating which parameters to reset
            noise_scale: Standard deviation of Gaussian noise for reset
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in mask:
                    masked_param = mask[name]
                    noise = torch.randn_like(param.data) * noise_scale
                    param.data[masked_param] = noise[masked_param]

    def finetune_masked_params(
        self,
        retain_loader,
        mask: Dict[str, torch.Tensor],
        learning_rate: float = 0.01,
        epochs: int = 1,
        optimizer_class=torch.optim.Adam,
    ):
        """
        Finetune only masked parameters on retain set.

        Args:
            retain_loader: DataLoader with examples to retain
            mask: Binary mask for parameters to finetune
            learning_rate: Learning rate for finetuning
            epochs: Number of epochs to finetune
            optimizer_class: Optimizer class to use
        """
        # Freeze non-masked parameters
        for name, param in self.model.named_parameters():
            if name not in mask:
                param.requires_grad = False
            else:
                param.requires_grad = mask[name].any()

        optimizer = optimizer_class(
            [p for name, p in self.model.named_parameters() if name in mask],
            lr=learning_rate,
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in retain_loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                optimizer.zero_grad()
                outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
                loss = (
                    outputs.loss if hasattr(outputs, "loss") else outputs[0]
                )

                loss.backward()

                # Zero gradients for non-masked parameters
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name not in mask:
                        param.grad.zero_()

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(retain_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Re-enable all parameters
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
        """
        Full DEL unlearning pipeline: localize → reset → finetune.

        Args:
            forget_loader: Data to forget
            retain_loader: Data to retain
            budget_alpha: Fraction of parameters to unlearn
            top_h: Top-h scores per channel to average
            finetune_lr: Learning rate for finetuning
            finetune_epochs: Number of finetuning epochs
        """
        print(f"[DEL] Computing criticality scores...")
        self.compute_criticality_scores(forget_loader, top_h=top_h)

        print(f"[DEL] Generating mask with budget α={budget_alpha}...")
        mask = self.generate_mask(self.criticality_scores, budget_alpha)

        print(f"[DEL] Resetting {sum(m.sum().item() for m in mask.values())} parameters...")
        self.reset_parameters(mask)

        print(f"[DEL] Finetuning masked parameters on retain set...")
        self.finetune_masked_params(
            retain_loader,
            mask,
            learning_rate=finetune_lr,
            epochs=finetune_epochs,
        )

        print(f"[DEL] Unlearning complete")
        return mask

    def _get_attention_layers(self) -> List[str]:
        """Identify attention layer names in the model."""
        attention_keywords = ["attention", "self_attn", "q_proj", "k_proj", "v_proj"]
        return [
            name
            for name, _ in self.model.named_parameters()
            if any(kw in name.lower() for kw in attention_keywords)
        ]
