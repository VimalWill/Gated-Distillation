"""
SPE-Unlearn (Structure-aware Parameter-Efficient) unlearning method.

Reference: "Structure-Aware Parameter-Efficient Machine Unlearning on Transformer Models"

Main contribution: Uses Fisher Information Matrix + forget gradients to identify
critical attention heads and FFN filters, enabling efficient unlearning via
sparse second-order updates.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class SPEUnlearning:
    """Structure-aware Parameter-Efficient unlearning for Transformers."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize SPE-Unlearn.

        Args:
            model: Transformer model to unlearn from
            device: Device to run computations on
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.fim_diag = None
        self.forget_grads = None
        self.importance_scores = None

    def accumulate_fisher_information(
        self, retain_loader, layer_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Accumulate diagonal Fisher Information Matrix on retain set.

        FIM = E[(∇ℓ)²] approximates Hessian curvature of loss landscape.

        Args:
            retain_loader: DataLoader for retain set
            layer_names: Specific layers to target (None = attention + FFN)

        Returns:
            Dictionary mapping parameter names to diagonal FIM values
        """
        fim = defaultdict(lambda: torch.zeros(1, device=self.device))

        if layer_names is None:
            layer_names = self._get_structural_layers()

        self.model.eval()
        num_batches = 0

        for batch in retain_loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            grads = torch.autograd.grad(
                loss,
                [p for name, p in self.model.named_parameters() if any(ln in name for ln in layer_names)],
                create_graph=False,
                retain_graph=False,
            )

            for (name, param), grad in zip(
                [(n, p) for n, p in self.model.named_parameters() if any(ln in n for ln in layer_names)],
                grads,
            ):
                if grad is not None:
                    # Accumulate squared gradients: FIM = (∇ℓ)²
                    fim[name] = fim[name] + (grad ** 2)

            num_batches += 1

        # Average over batches
        for name in fim:
            fim[name] = fim[name] / num_batches

        self.fim_diag = dict(fim)
        return self.fim_diag

    def accumulate_forget_gradients(
        self, forget_loader, layer_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Accumulate gradients on forget set for importance scoring.

        Args:
            forget_loader: DataLoader for data to forget
            layer_names: Specific layers to target

        Returns:
            Dictionary mapping parameter names to accumulated gradients
        """
        grads = defaultdict(lambda: torch.zeros(1, device=self.device))

        if layer_names is None:
            layer_names = self._get_structural_layers()

        self.model.eval()

        for batch in forget_loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            batch_grads = torch.autograd.grad(
                loss,
                [p for name, p in self.model.named_parameters() if any(ln in name for ln in layer_names)],
                create_graph=False,
                retain_graph=False,
            )

            for (name, param), grad in zip(
                [(n, p) for n, p in self.model.named_parameters() if any(ln in n for ln in layer_names)],
                batch_grads,
            ):
                if grad is not None:
                    grads[name] = grads[name] + grad.abs()

        self.forget_grads = dict(grads)
        return self.forget_grads

    def compute_importance_scores(
        self,
        retain_loader,
        forget_loader,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores: SC_i = (1/2)*FIM + forget_gradients

        Combines curvature (FIM) and gradient-based importance from forget set.

        Args:
            retain_loader: Data to retain
            forget_loader: Data to forget
            layer_names: Layers to target

        Returns:
            Dictionary of importance scores per parameter
        """
        print("[SPE] Accumulating Fisher Information Matrix...")
        self.accumulate_fisher_information(retain_loader, layer_names)

        print("[SPE] Accumulating forget set gradients...")
        self.accumulate_forget_gradients(forget_loader, layer_names)

        # SC_i = (1/2)*FIM + |∇ℓ_forget|
        importance_scores = {}
        for name in self.fim_diag:
            if name in self.forget_grads:
                sc = 0.5 * self.fim_diag[name] + self.forget_grads[name]
                importance_scores[name] = sc

        self.importance_scores = importance_scores
        return importance_scores

    def generate_structural_mask(
        self,
        importance_scores: Dict[str, torch.Tensor],
        sparsity: float = 0.9,
        target_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate mask selecting low-importance structures to update (freeze others).

        Args:
            importance_scores: Importance scores per parameter
            sparsity: Fraction of parameters to freeze (0.9 = update 10%)
            target_ratio: Target ratio of params to update (overrides sparsity if set)

        Returns:
            Binary masks for parameters (1=update, 0=freeze)
        """
        masks = {}
        total_params = sum(s.numel() for s in importance_scores.values())

        # Determine update budget
        if target_ratio is not None:
            update_budget = int(target_ratio * total_params)
        else:
            update_budget = int((1 - sparsity) * total_params)

        # Flatten and sort by importance
        all_scores = []
        for name, scores in importance_scores.items():
            flat_scores = scores.flatten()
            for idx, score in enumerate(flat_scores):
                all_scores.append((score.item(), name, idx))

        # Sort ascending: low importance scores = update
        all_scores.sort(key=lambda x: x[0])

        # Select indices to update
        update_indices = set((name, idx) for _, name, idx in all_scores[:update_budget])

        # Create masks (1 = update, 0 = freeze)
        for name, scores in importance_scores.items():
            mask = torch.zeros_like(scores, dtype=torch.float32, device=self.device)
            flat_mask = mask.flatten()
            for idx in range(scores.numel()):
                if (name, idx) in update_indices:
                    flat_mask[idx] = 1.0
            masks[name] = mask

        return masks

    def sparse_second_order_update(
        self,
        mask: Dict[str, torch.Tensor],
        learning_rate: float = 1e-6,
    ):
        """
        Apply sparse second-order update: θ += η * m ◦ (Î^{-1} * g_forget)

        Args:
            mask: Binary mask (1=update, 0=freeze)
            learning_rate: Update learning rate
        """
        epsilon = 1e-8  # Numerical stability

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in mask or name not in self.fim_diag:
                    continue

                # Compute diagonal FIM inverse with regularization
                fim_inv = 1.0 / (self.fim_diag[name] + epsilon)

                # Apply sparse second-order update
                # θ += η * m ◦ (Î^{-1} * g_forget)
                update = self.forget_grads[name] * fim_inv
                sparse_update = mask[name] * update
                param.data = param.data + learning_rate * sparse_update

    def unlearn(
        self,
        retain_loader,
        forget_loader,
        sparsity: float = 0.9,
        learning_rate: float = 1e-6,
        layer_names: Optional[List[str]] = None,
    ):
        """
        Full SPE-Unlearn pipeline: compute importance → mask → sparse SO update.

        Args:
            retain_loader: Data to retain
            forget_loader: Data to forget
            sparsity: Fraction to freeze (0.9 = update 10%)
            learning_rate: Update learning rate
            layer_names: Specific layers to target
        """
        print(f"[SPE] Computing importance scores (sparsity={sparsity})...")
        self.compute_importance_scores(retain_loader, forget_loader, layer_names)

        print(f"[SPE] Generating structural mask...")
        mask = self.generate_structural_mask(self.importance_scores, sparsity=sparsity)

        print(f"[SPE] Applying sparse second-order update (lr={learning_rate})...")
        self.sparse_second_order_update(mask, learning_rate=learning_rate)

        print(f"[SPE] Unlearning complete")
        return mask

    def _get_structural_layers(self) -> List[str]:
        """Identify attention and FFN layer names."""
        structural_keywords = ["attention", "self_attn", "q_proj", "k_proj", "v_proj",
                               "mlp", "dense", "fc", "feed_forward"]
        return [
            name
            for name, _ in self.model.named_parameters()
            if any(kw in name.lower() for kw in structural_keywords)
        ]
