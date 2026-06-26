"""
SPE-Unlearn (Structure-aware Parameter-Efficient) unlearning method.

Reference: "Structure-Aware Parameter-Efficient Machine Unlearning on Transformer Models"

Approximation note (S4/S8): The paper derives importance at the attention-head /
FFN-filter level via mask-level gradients (Eq. 8, Algorithm 1 line 12). This
implementation proxies structure importance by aggregating parameter-level FIM and
forget gradients to the output-neuron (row) level — a practical simplification when
head-boundary metadata isn't exposed by the model API.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class SPEUnlearning:
    """Structure-aware Parameter-Efficient unlearning for Transformers."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.fim_diag: Optional[Dict[str, torch.Tensor]] = None
        self.forget_grads: Optional[Dict[str, torch.Tensor]] = None
        self.importance_scores: Optional[Dict[str, torch.Tensor]] = None

    def accumulate_fisher_information(
        self, retain_loader, layer_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Accumulate diagonal FIM on retain set: Î = (1/|D_r|) Σ (∇ℓ)².
        """
        if layer_names is None:
            layer_names = self._get_structural_layers()

        target_params = [
            (name, param)
            for name, param in self.model.named_parameters()
            if any(ln in name for ln in layer_names)
        ]

        # Pre-allocate with correct shapes (avoids defaultdict broadcast footgun)
        fim: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in target_params
        }

        self.model.eval()
        num_batches = 0

        # S2 fix: enable_grad() so autograd.grad works even if caller is inside no_grad
        with torch.enable_grad():
            for batch in retain_loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

                grads = torch.autograd.grad(
                    loss,
                    [p for _, p in target_params],
                    create_graph=False,
                    allow_unused=True,
                )
                with torch.no_grad():
                    for (name, _), grad in zip(target_params, grads):
                        if grad is not None:
                            fim[name].add_(grad ** 2)

                num_batches += 1

        with torch.no_grad():
            for name in fim:
                fim[name].div_(max(num_batches, 1))

        self.fim_diag = fim
        return fim

    def accumulate_forget_gradients(
        self, forget_loader, layer_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Accumulate signed gradients on forget set: Σ ∇_θ ℓ(θ, x) for x in D_f.

        S3 fix: keep signed gradients (no .abs()) — the Newton step
        θ += η · Î^{-1} · g_θ needs the direction, not just magnitude.
        """
        if layer_names is None:
            layer_names = self._get_structural_layers()

        target_params = [
            (name, param)
            for name, param in self.model.named_parameters()
            if any(ln in name for ln in layer_names)
        ]

        # Pre-allocate with correct shapes
        grads: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in target_params
        }

        self.model.eval()
        with torch.enable_grad():  # S2 fix
            for batch in forget_loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

                batch_grads = torch.autograd.grad(
                    loss,
                    [p for _, p in target_params],
                    create_graph=False,
                    allow_unused=True,
                )
                with torch.no_grad():
                    for (name, _), grad in zip(target_params, batch_grads):
                        if grad is not None:
                            # S3 fix: accumulate signed gradients, not grad.abs()
                            grads[name].add_(grad)

        self.forget_grads = grads
        return grads

    def compute_importance_scores(
        self,
        retain_loader,
        forget_loader,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        SC_i = (1/2) · FIM_i + |∇ℓ_forget_i|

        forget_grads are kept signed (for Newton step); abs is applied here only
        for the importance scoring step.
        """
        print("[SPE] Accumulating Fisher Information Matrix...")
        self.accumulate_fisher_information(retain_loader, layer_names)

        print("[SPE] Accumulating forget set gradients...")
        self.accumulate_forget_gradients(forget_loader, layer_names)

        importance: Dict[str, torch.Tensor] = {}
        for name in self.fim_diag:
            if name in self.forget_grads:
                importance[name] = 0.5 * self.fim_diag[name] + self.forget_grads[name].abs()

        self.importance_scores = importance
        return importance

    def generate_structural_mask(
        self,
        importance_scores: Dict[str, torch.Tensor],
        sparsity: float = 0.9,
    ) -> Dict[str, torch.Tensor]:
        """
        Select the most-important output neurons to update; freeze the rest.

        sparsity = fraction of structural units to freeze.
        (1 - sparsity) fraction receives the Newton update.

        Direction (paper Algorithm 1, lines 12-16): unimportant structures get
        m=0 (frozen); the high-importance structures keep m=1 and receive the
        update θ += η · m ∘ Î⁻¹ gθ. Influence-critical structures are the ones
        modified, not the least-critical ones.

        S5 fix: granularity is output neurons (rows of weight matrices), not
        individual weights. Row-level importance = mean score across each row.
        This proxies head/filter structures when boundaries aren't directly accessible.
        """
        # Aggregate parameter-level scores to output-neuron level
        row_scores: Dict[str, torch.Tensor] = {}
        for name, scores in importance_scores.items():
            if scores.dim() >= 2:
                row_scores[name] = scores.mean(dim=tuple(range(1, scores.dim())))  # (out_dim,)
            else:
                row_scores[name] = scores  # bias: per-element

        total_neurons = sum(s.numel() for s in row_scores.values())
        update_budget = int((1 - sparsity) * total_neurons)

        # Rank neurons descending: highest importance → update first (paper lines 13-16)
        all_neurons = [
            (row_scores[name][idx].item(), name, idx)
            for name in row_scores
            for idx in range(row_scores[name].numel())
        ]
        all_neurons.sort(key=lambda x: x[0], reverse=True)
        update_set = {(name, idx) for _, name, idx in all_neurons[:update_budget]}

        # Build parameter-level masks: 1.0 = update this row, 0.0 = freeze
        param_dict = dict(self.model.named_parameters())
        masks: Dict[str, torch.Tensor] = {}
        for name in importance_scores:
            if name not in param_dict:
                continue
            param = param_dict[name]
            mask = torch.zeros_like(param, dtype=torch.float32, device=self.device)
            for row_idx in range(param.shape[0]):
                if (name, row_idx) in update_set:
                    mask[row_idx] = 1.0
            masks[name] = mask

        return masks

    def sparse_second_order_update(
        self,
        mask: Dict[str, torch.Tensor],
        learning_rate: float = 1e-6,
        epsilon: float = 1e-8,
    ):
        """
        Apply sparse Newton step: θ += η · m ⊙ (Î^{-1} · g_θ)

        S6 fix: use in-place .add_() instead of .data reassignment to avoid
        breaking optimizer state and forward hooks.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in mask or name not in self.fim_diag:
                    continue
                fim_inv = 1.0 / (self.fim_diag[name] + epsilon)
                update = self.forget_grads[name] * fim_inv
                # S6 fix: in-place mutation, not param.data = param.data + ...
                param.data.add_(learning_rate * mask[name] * update)

    def unlearn(
        self,
        retain_loader,
        forget_loader,
        sparsity: float = 0.9,
        learning_rate: float = 1e-6,
        layer_names: Optional[List[str]] = None,
    ):
        """Full SPE-Unlearn pipeline: importance → structural mask → sparse SO update."""
        print(f"[SPE] Computing importance scores (sparsity={sparsity})...")
        self.compute_importance_scores(retain_loader, forget_loader, layer_names)

        print("[SPE] Generating structural mask...")
        mask = self.generate_structural_mask(self.importance_scores, sparsity=sparsity)

        print(f"[SPE] Applying sparse second-order update (lr={learning_rate})...")
        self.sparse_second_order_update(mask, learning_rate=learning_rate)

        print("[SPE] Unlearning complete")
        return mask

    def _get_structural_layers(self) -> List[str]:
        """Attention and FFN parameter names."""
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
