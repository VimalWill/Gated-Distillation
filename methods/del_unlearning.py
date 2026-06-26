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
        Compute per-output-channel criticality: mean of top-h |Σ_B θ_j · g_j(B)| per channel.

        Paper Algorithm 1 (Appendix A.5):
            s_j = Σ_B θ_j · g_j(θ, B)        (signed sum across batches)
            s_j ← |s_j|                       (magnitude taken once)
            c_oi = (1/h) · Σ top-h s_i[j]    (per output channel)
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
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss.backward()

            # DEL-2 fix: accumulate signed theta·grad across batches; take magnitude once after.
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in score_accum and param.grad is not None:
                        score_accum[name].add_(param.data * param.grad)

        self.model.zero_grad()

        # DEL-2 fix: |s_j| applied once after the loop, per Algorithm 1 line 4.
        with torch.no_grad():
            for name in score_accum:
                score_accum[name].abs_()

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
        Greedy-fill mask in descending criticality order until the running param
        count reaches budget_alpha * total_params, marking whole output channels
        (rows of 2-D weights, individual bias elements for 1-D).

        budget_alpha is the parameter-fraction budget, matching paper Algorithm 1
        line 6: |m|_1 ≤ p · α. Each output channel contributes its full row of
        weights (or one bias element) to the running count.
        """
        # DEL-3 fix: budget is parameter-fraction, not channel-fraction.
        # Compute per-channel param count and total targeted params up front.
        param_dict = {n: p for n, p in self.model.named_parameters()}

        channel_info = []  # (score, name, channel_idx, param_count_for_this_channel)
        for name, scores in criticality_scores.items():
            param = param_dict[name]
            if param.dim() >= 2:
                row_size = param.numel() // param.shape[0]
            else:
                row_size = 1
            for ch in range(scores.numel()):
                channel_info.append((scores[ch].item(), name, ch, row_size))
        channel_info.sort(key=lambda x: x[0], reverse=True)

        total_params = sum(
            p.numel() for n, p in self.model.named_parameters() if n in criticality_scores
        )
        target_params = int(budget_alpha * total_params)

        selected: set = set()
        running = 0
        for _, name, ch, sz in channel_info:
            if running + sz > target_params:
                break
            selected.add((name, ch))
            running += sz

        # Build per-parameter masks: mark full rows for 2-D, single elements for 1-D.
        masks: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if name not in criticality_scores:
                continue
            mask = torch.zeros_like(param, dtype=torch.bool, device=self.device)
            for ch in range(param.shape[0]):
                if (name, ch) in selected:
                    mask[ch] = True
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
        Finetune only masked entries on the retain set, plus the classifier head.

        D5 fix: after backward, zero gradient at unmasked positions inside masked
        parameters so those entries are not updated by optimizer.step().

        DEL-CLS fix: classifier-layer params (matched by name) are finetuned
        fully — paper's RFT finetunes masked params AND the classifier head on
        the retain set; without this the comparison is asymmetric.
        """
        # Names containing 'classifier' bypass gradient-zeroing entirely.
        # Keep requires_grad=True on all params so the forward graph is intact
        # (params outside the mask would otherwise break backward when used in
        # the forward pass — e.g. classifier head on a real HF model).
        for param in self.model.parameters():
            param.requires_grad = True

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
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss.backward()

                # D5 fix: zero gradient at unmasked positions within masked parameters,
                # and zero gradients of parameters outside the mask entirely so
                # optimizer.step() is a no-op for them.
                # DEL-CLS exception: classifier params update fully (no gradient zero).
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        continue
                    if "classifier" in name.lower():
                        continue  # let classifier params update freely
                    if name in mask:
                        param.grad[~mask[name]] = 0.0
                    else:
                        param.grad.zero_()

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

        print(f"[DEL] Generating mask (parameter-budget α={budget_alpha})...")
        mask = self.generate_mask(self.criticality_scores, budget_alpha)
        n_params = int(sum(m.sum().item() for m in mask.values()))
        total_targeted = sum(p.numel() for n, p in self.model.named_parameters() if n in self.criticality_scores)
        print(f"[DEL] Resetting {n_params}/{total_targeted} targeted parameters ({n_params/max(total_targeted,1):.1%})...")
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
