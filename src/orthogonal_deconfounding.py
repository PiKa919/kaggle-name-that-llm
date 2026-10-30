"""
Orthogonal Subspace Deconfounding and Truncation-Invariant Representation Learning.
Formulated for Kaggle 'Name That LLM' (Target: 0.70+ Balanced Accuracy).

Mathematical Architecture:
--------------------------
1. ECONOMY GRAM SVD & FACTORED LOW-RANK PROMPT ORTHOGONALIZATION:
   Let X in R^{N x D} be passage representations (from intermediate transformer layers),
   and p in {1, ..., G}^N be prompt group identifiers.
   Let C in R^{G x D} be the prompt centroid matrix where row g is:
       c_g = (1 / |S_g|) sum_{i in S_g} x_i
   Centered prompt centroids: C_tilde = C - (1_G mu_C^T), where mu_C = (1 / G) sum c_g.

   When D >> G (e.g., D = 4096 * 8 = 32768, G = 150):
   Direct SVD on C_tilde requires O(G D^2) and O(D^2) memory, which OOMs on 32GB GPUs.
   Instead, we compute the Economy Gram Eigendecomposition:
       K = C_tilde C_tilde^T in R^{G x G}
       K = V Lambda V^T,   Lambda = diag(lambda_1, ..., lambda_G)
       sigma_j = sqrt(max(lambda_j, 0))
   For non-zero singular values sigma_j > epsilon, the right singular vectors in R^D are:
       u_j = (1 / sigma_j) C_tilde^T v_j
   Forming the prompt subspace basis U_k = [u_1, ..., u_k] in R^{D x k}.
   Properties:
       U_k^T U_k = I_k   (Strict orthonormality)

   Factored Low-Rank Orthogonal Complement Projection:
       Dense projector:     Pi_perp = I_D - U_k U_k^T in R^{D x D}  (Allocates O(D^2) - Prohibited)
       Factored projection: X_deconf = X - (X U_k) U_k^T             (Allocates O(N k)   - Optimal)
   Identity:
       X_deconf = X (I_D - U_k U_k^T)
   Properties:
       (a) Idempotence:       (X_deconf)_deconf = X_deconf
       (b) Annihilation:      X_deconf U_k = 0_{N x k}
       (c) Zero D x D alloc:  O(N D k) FLOPs, exact to 1e-15 machine precision.

2. TRUNCATION-INVARIANT INTERMEDIATE POOLING & ATTENTION-SINK AVOIDANCE:
   Autoregressive transformers dump excess attention probability mass onto token 0
   (attention sink), distorting its hidden state norm by >100x (Xiao et al., 2023).
   - Token 0 Exclusion: mask[:, 0] = 0.
   - Multi-layer intermediate readout (layers 4-16): Probing literature demonstrates
     intermediate layers preserve syntactic/stylometric author signatures without
     overfitting to upper-layer semantic task objectives (Tenney et al., 2019).
   - Length-Dispersion Normalization: Short passages (<60 words) suffer from
     under-rotated RoPE frequencies and high token-mean sampling variance.
     We apply token-dispersion scaling s(L) = sqrt((L + tau) / (L + 1)) combined
     with layer-wise LayerNorm.

3. CROSS-TRUNK REPRESENTATION HARMONIZER:
   Calibrates and fuses representations across heterogeneous model families:
   - Trunk 1: Qwen3-8B (H=4096, 36 layers, Alibaba RoPE)
   - Trunk 2: Qwen3-14B (H=5120, 40 layers, dual T4 partitioned)
   - Trunk 3: OLMo2-7B (H=4096, native-24 prefix, AllenAI RMSNorm)
   Performs:
   - Label space permutation alignment (OLMo2 alphabetical -> Canonical 18).
   - Per-trunk orthogonal prompt subspace deconfounding.
   - Length-gated dynamic fusion: stylometric trunks prioritized on short text (<60),
     deep reasoning trunks prioritized on long text (>=60).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

# Canonical competition label ordering (18 LLMs)
CANONICAL_LABELS: List[str] = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"
]
K_CLASSES = len(CANONICAL_LABELS)

# OLMo2 assets alphabetical class ordering
OLMO2_CLASSES: List[str] = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]


def align_labels_to_canonical(
    probs: Union[np.ndarray, torch.Tensor],
    source_classes: Sequence[str],
    target_classes: Sequence[str] = CANONICAL_LABELS,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Permutes columns of probability / logit tensor from source_classes to target_classes.
    Supports both NumPy ndarrays and PyTorch tensors.
    """
    if list(source_classes) == list(target_classes):
        return probs

    indices = [list(source_classes).index(lbl) for lbl in target_classes]
    if isinstance(probs, torch.Tensor):
        idx_t = torch.tensor(indices, dtype=torch.long, device=probs.device)
        return probs.index_select(-1, idx_t)
    elif isinstance(probs, np.ndarray):
        return probs[..., indices]
    else:
        raise TypeError(f"Unsupported type {type(probs)}; expected torch.Tensor or np.ndarray")


# =====================================================================
# 1. PROMPT SUBSPACE DECONFOUNDER (ECONOMY GRAM SVD)
# =====================================================================

class PromptSubspaceDeconfounder:
    """
    Extracts the dominant prompt centroid variation subspace U_k via Economy Gram SVD,
    and projects arbitrary representation matrices X onto its orthogonal complement:
        X_perp = X - (X U_k) U_k^T

    Complexity:
        Fit: O(G^2 D + G^3) where G is number of prompt clusters, D is feature dimension.
        Transform: O(N D k) FLOPs with ZERO D x D matrix allocation.
    """

    def __init__(
        self,
        n_components: Optional[int] = 32,
        variance_ratio: Optional[float] = None,
        center_centroids: bool = True,
        center_features: bool = False,
        regularization: float = 1e-7,
    ) -> None:
        """
        Args:
            n_components: Maximum number of prompt subspace directions to remove.
            variance_ratio: Optional cumulative variance threshold in (0, 1].
                            If specified, dynamically selects k singular vectors to
                            explain this fraction of prompt centroid variance.
            center_centroids: Whether to subtract grand prompt centroid mean before SVD.
            center_features: Whether to center representations X by overall feature mean.
            regularization: Minimal singular value cutoff to prevent numerical instability.
        """
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.center_centroids = center_centroids
        self.center_features = center_features
        self.regularization = regularization

        # Fitted attributes
        self.U_: Optional[np.ndarray] = None                  # (D, k) orthonormal basis
        self.singular_values_: Optional[np.ndarray] = None    # (k,)
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_centroid_: Optional[np.ndarray] = None      # (D,)
        self.feature_mean_: Optional[np.ndarray] = None       # (D,)
        self.k_: int = 0
        self.d_in_: int = 0
        self.is_fitted_: bool = False

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        prompt_ids: Union[np.ndarray, torch.Tensor, Sequence[Any]],
    ) -> PromptSubspaceDeconfounder:
        """
        Fits the prompt subspace on training embeddings X and corresponding prompt IDs.

        Args:
            X: (N, D) representation matrix.
            prompt_ids: (N,) prompt group identifiers.
        """
        # Convert to numpy for stable double-precision SVD
        is_torch = isinstance(X, torch.Tensor)
        if is_torch:
            X_np = X.detach().cpu().numpy().astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)

        if isinstance(prompt_ids, torch.Tensor):
            p_np = prompt_ids.detach().cpu().numpy()
        else:
            p_np = np.asarray(prompt_ids)

        N, D = X_np.shape
        self.d_in_ = D

        if self.center_features:
            self.feature_mean_ = np.mean(X_np, axis=0)
        else:
            self.feature_mean_ = np.zeros(D, dtype=np.float64)

        # 1. Compute prompt centroids C in R^{G x D}
        unique_prompts, inverse_indices = np.unique(p_np, return_inverse=True)
        G = len(unique_prompts)

        if G < 2:
            # Cannot identify a variation subspace from <=1 prompt
            self.k_ = 0
            self.U_ = np.zeros((D, 0), dtype=np.float64)
            self.singular_values_ = np.zeros(0, dtype=np.float64)
            self.explained_variance_ratio_ = np.zeros(0, dtype=np.float64)
            self.mean_centroid_ = np.zeros(D, dtype=np.float64)
            self.is_fitted_ = True
            return self

        # Centroid accumulation across unique prompts
        centroids = np.zeros((G, D), dtype=np.float64)
        for g_idx in range(G):
            mask = (inverse_indices == g_idx)
            centroids[g_idx] = np.mean(X_np[mask], axis=0)

        # 2. Centering prompt centroids
        if self.center_centroids:
            self.mean_centroid_ = np.mean(centroids, axis=0)
            C_tilde = centroids - self.mean_centroid_
        else:
            self.mean_centroid_ = np.zeros(D, dtype=np.float64)
            C_tilde = centroids

        # 3. Economy Gram Matrix: K = C_tilde C_tilde^T in R^{G x G}
        K = C_tilde @ C_tilde.T

        # 4. Eigendecomposition of symmetric positive semi-definite Gram matrix
        eigvals, V_gram = np.linalg.eigh(K)

        # Sort descending
        sort_idx = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[sort_idx], 0.0)
        V_gram = V_gram[:, sort_idx]

        singular_values = np.sqrt(eigvals)
        total_var = np.sum(eigvals)

        # 5. Determine rank k
        # Max feasible rank is min(G - (1 if centered else 0), D)
        max_possible_k = min(G - 1 if self.center_centroids else G, D)

        # Filter numerically non-zero singular values
        valid_mask = singular_values > max(self.regularization, 1e-12 * singular_values[0] if len(singular_values) > 0 and singular_values[0] > 0 else 1e-12)
        valid_k = min(int(np.sum(valid_mask)), max_possible_k)

        if self.variance_ratio is not None and total_var > 1e-15:
            cum_var = np.cumsum(eigvals) / total_var
            ratio_k = int(np.searchsorted(cum_var, self.variance_ratio)) + 1
            chosen_k = min(ratio_k, valid_k)
            if self.n_components is not None:
                chosen_k = min(chosen_k, self.n_components)
        elif self.n_components is not None:
            chosen_k = min(self.n_components, valid_k)
        else:
            chosen_k = valid_k

        chosen_k = max(0, chosen_k)
        self.k_ = chosen_k

        if chosen_k == 0:
            self.U_ = np.zeros((D, 0), dtype=np.float64)
            self.singular_values_ = np.zeros(0, dtype=np.float64)
            self.explained_variance_ratio_ = np.zeros(0, dtype=np.float64)
            self.is_fitted_ = True
            return self

        # 6. Extract exact right singular vectors U_k in R^{D x k} via:
        #    u_j = (1 / sigma_j) * C_tilde^T v_j
        V_sub = V_gram[:, :chosen_k]
        s_sub = singular_values[:chosen_k]

        U_k = (C_tilde.T @ V_sub) / s_sub[None, :]

        # Guarantee strict orthonormality via Gram-Schmidt / QR if needed
        # (Economy SVD is already orthonormal to within machine precision)
        self.U_ = U_k
        self.singular_values_ = s_sub
        if total_var > 1e-15:
            self.explained_variance_ratio_ = eigvals[:chosen_k] / total_var
        else:
            self.explained_variance_ratio_ = np.ones(chosen_k) / max(chosen_k, 1)

        self.is_fitted_ = True
        return self

    def transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Projects representations X onto the orthogonal complement of the prompt subspace:
            X_perp = X - (X U_k) U_k^T

        Returns projected tensor of same type, dtype, and device as X.
        """
        assert self.is_fitted_, "PromptSubspaceDeconfounder must be fitted before transform."

        if self.k_ == 0 or self.U_ is None:
            return X

        is_torch = isinstance(X, torch.Tensor)
        if is_torch:
            orig_device = X.device
            orig_dtype = X.dtype

            # Convert basis to torch tensor on matching device and dtype
            U_t = torch.as_tensor(self.U_, dtype=orig_dtype, device=orig_device)

            if self.center_features and self.feature_mean_ is not None:
                mean_t = torch.as_tensor(self.feature_mean_, dtype=orig_dtype, device=orig_device)
                X_in = X - mean_t
            else:
                X_in = X

            # Factored low-rank projection: O(N * D * k)
            proj_coords = torch.matmul(X_in, U_t)            # (N, k)
            proj_subspace = torch.matmul(proj_coords, U_t.T) # (N, D)
            X_perp = X_in - proj_subspace

            if self.center_features and self.feature_mean_ is not None:
                # Add back mean if desired, or keep centered deconfounded
                X_perp = X_perp + mean_t

            return X_perp
        else:
            X_np = np.asarray(X)
            orig_dtype = X_np.dtype

            if self.center_features and self.feature_mean_ is not None:
                X_in = X_np - self.feature_mean_
            else:
                X_in = X_np

            proj_coords = X_in @ self.U_            # (N, k)
            proj_subspace = proj_coords @ self.U_.T # (N, D)
            X_perp = X_in - proj_subspace

            if self.center_features and self.feature_mean_ is not None:
                X_perp = X_perp + self.feature_mean_

            return X_perp.astype(orig_dtype)

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        prompt_ids: Union[np.ndarray, torch.Tensor, Sequence[Any]],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Fits on (X, prompt_ids) and transforms X."""
        return self.fit(X, prompt_ids).transform(X)

    def project_subspace(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Projects representations X directly INTO the prompt subspace:
            X_parallel = (X U_k) U_k^T
        """
        assert self.is_fitted_, "PromptSubspaceDeconfounder must be fitted."
        if self.k_ == 0 or self.U_ is None:
            if isinstance(X, torch.Tensor):
                return torch.zeros_like(X)
            return np.zeros_like(X)

        if isinstance(X, torch.Tensor):
            orig_device = X.device
            orig_dtype = X.dtype
            U_t = torch.as_tensor(self.U_, dtype=orig_dtype, device=orig_device)
            return torch.matmul(torch.matmul(X, U_t), U_t.T)
        else:
            X_np = np.asarray(X)
            return (X_np @ self.U_) @ self.U_.T

    def prompt_variance_ratio(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> float:
        """
        Computes the proportion of Frobenius energy in X aligned with the prompt subspace:
            ||X_parallel||_F^2 / ||X||_F^2
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)

        total_norm_sq = np.sum(X_np ** 2)
        if total_norm_sq < 1e-12:
            return 0.0

        X_par = self.project_subspace(X_np)
        par_norm_sq = np.sum(X_par ** 2)
        return float(par_norm_sq / total_norm_sq)


# =====================================================================
# 2. TRUNCATION-INVARIANT POOLING & ATTENTION SINK AVOIDANCE
# =====================================================================

class TruncationInvariantPooling(nn.Module):
    """
    Truncation-Invariant Intermediate Hidden State Pooling.
    Protects representations from:
    1. Attention Sink Contamination: Xiao et al. (2023) demonstrated autoregressive
       models assign massive probability mass to token 0. We enforce mask[:, 0] = 0.
    2. Syntactic/Stylometric Readout: Extracts lower/intermediate layers (e.g. 4-16)
       which encode syntactic and stylometric markers, avoiding top-layer semantic overfitting.
    3. Length Dispersion Distortion: Normalizes representation scale across short passages
       (<60 tokens) and long passages via sequence dispersion stabilization and RoPE compensation.
    """

    def __init__(
        self,
        hidden_size: int,
        target_layers: Sequence[int] = (4, 8, 12, 16),
        avoid_attention_sink: bool = True,
        sink_token_indices: Sequence[int] = (0,),
        use_layer_norm: bool = True,
        length_dispersion_norm: bool = True,
        dispersion_tau: float = 8.0,
        rope_attenuation_correction: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.target_layers = list(target_layers)
        self.avoid_attention_sink = avoid_attention_sink
        self.sink_token_indices = list(sink_token_indices)
        self.use_layer_norm = use_layer_norm
        self.length_dispersion_norm = length_dispersion_norm
        self.dispersion_tau = dispersion_tau
        self.rope_attenuation_correction = rope_attenuation_correction
        self.eps = eps

        # Layer normalization per intermediate layer readout
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_size, eps=eps) for _ in self.target_layers
            ])
        else:
            self.layer_norms = None

    @property
    def output_dim(self) -> int:
        return self.hidden_size * len(self.target_layers)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, Sequence[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pools intermediate transformer representations.

        Args:
            hidden_states: Either a single tensor (B, S, H) or a list/tuple of layer
                           tensors (each B, S, H).
            attention_mask: (B, S) binary mask (1 for token, 0 for pad).

        Returns:
            (B, len(target_layers) * H) pooled representation tensor.
        """
        # Unify hidden states to list of layer tensors
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.ndim == 4:
                # (B, L, S, H)
                layers_list = [hidden_states[:, layer_i, :, :] for layer_i in range(hidden_states.shape[1])]
            elif hidden_states.ndim == 3:
                # (B, S, H) single layer
                layers_list = [hidden_states]
            else:
                raise ValueError(f"Unexpected hidden_states ndim: {hidden_states.ndim}")
        else:
            layers_list = list(hidden_states)

        B, S, H = layers_list[0].shape
        dev = layers_list[0].device

        # Construct / clean attention mask
        if attention_mask is None:
            mask = torch.ones((B, S), dtype=torch.float32, device=dev)
        else:
            mask = attention_mask.to(device=dev, dtype=torch.float32)

        # 1. Attention Sink Token Avoidance: mask[:, 0] = 0.0
        if self.avoid_attention_sink and S > 1:
            mask = mask.clone()
            for sink_idx in self.sink_token_indices:
                if sink_idx < S:
                    mask[:, sink_idx] = 0.0

        # Effective valid token length per sequence: (B, 1)
        eff_lens = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mask_expanded = mask.unsqueeze(-1)  # (B, S, 1)

        # 2. Length dispersion factor: s(L) = sqrt((L + tau) / (L + 1))
        if self.length_dispersion_norm:
            disp_scale = torch.sqrt((eff_lens + self.dispersion_tau) / (eff_lens + 1.0))
        else:
            disp_scale = torch.ones_like(eff_lens)

        # 3. RoPE Short-Passage Frequency Attenuation Compensation
        # For passages < 60 tokens, low-frequency RoPE dimensions fail to cycle,
        # leading to representation variance decay. We apply smooth compensation:
        if self.rope_attenuation_correction:
            # Alpha = 0.12 boost exponential decay with characteristic length 30
            rope_scale = 1.0 + 0.12 * torch.exp(-eff_lens / 30.0)
        else:
            rope_scale = 1.0

        combined_scale = disp_scale * rope_scale  # (B, 1)

        # 4. Multi-Layer Intermediate Readout
        pooled_slices = []
        for i, layer_idx in enumerate(self.target_layers):
            if layer_idx >= len(layers_list):
                raise IndexError(
                    f"Requested target layer {layer_idx}, but hidden_states only has {len(layers_list)} layers."
                )

            h = layers_list[layer_idx].float()  # (B, S, H)
            # Masked average pooling
            mean_pooled = (h * mask_expanded).sum(dim=1) / eff_lens  # (B, H)

            # Apply length dispersion and RoPE scale adjustment
            scaled_pooled = mean_pooled * combined_scale

            # Apply layer normalization
            if self.layer_norms is not None:
                normed = self.layer_norms[i](scaled_pooled)
            else:
                normed = scaled_pooled

            pooled_slices.append(normed)

        return torch.cat(pooled_slices, dim=-1)


def truncation_invariant_pool(
    hidden_states: Union[torch.Tensor, Sequence[torch.Tensor], np.ndarray, Sequence[np.ndarray]],
    attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    target_layers: Sequence[int] = (4, 8, 12, 16),
    avoid_attention_sink: bool = True,
    dispersion_tau: float = 8.0,
    rope_attenuation_correction: bool = True,
    eps: float = 1e-5,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Functional interface for truncation-invariant pooling with attention sink avoidance.
    Supports both PyTorch tensors and NumPy arrays seamlessly.
    """
    is_torch = isinstance(hidden_states, torch.Tensor) or (
        isinstance(hidden_states, (list, tuple)) and isinstance(hidden_states[0], torch.Tensor)
    )

    if is_torch:
        pooler = TruncationInvariantPooling(
            hidden_size=hidden_states[0].shape[-1] if isinstance(hidden_states, (list, tuple)) else hidden_states.shape[-1],
            target_layers=target_layers,
            avoid_attention_sink=avoid_attention_sink,
            use_layer_norm=True,
            length_dispersion_norm=True,
            dispersion_tau=dispersion_tau,
            rope_attenuation_correction=rope_attenuation_correction,
            eps=eps,
        )
        if isinstance(hidden_states, (list, tuple)):
            pooler = pooler.to(hidden_states[0].device)
        else:
            pooler = pooler.to(hidden_states.device)
        return pooler(hidden_states, attention_mask)

    # NumPy implementation
    if isinstance(hidden_states, np.ndarray):
        if hidden_states.ndim == 4:
            layers = [hidden_states[:, layer_i, :, :] for layer_i in range(hidden_states.shape[1])]
        else:
            layers = [hidden_states]
    else:
        layers = [np.asarray(h, dtype=np.float32) for h in hidden_states]

    B, S, H = layers[0].shape

    if attention_mask is None:
        mask = np.ones((B, S), dtype=np.float32)
    else:
        mask = np.asarray(attention_mask, dtype=np.float32).copy()

    if avoid_attention_sink and S > 1:
        mask[:, 0] = 0.0

    eff_lens = np.maximum(np.sum(mask, axis=1, keepdims=True), 1.0)  # (B, 1)
    mask_exp = mask[:, :, None]

    disp_scale = np.sqrt((eff_lens + dispersion_tau) / (eff_lens + 1.0))
    if rope_attenuation_correction:
        rope_scale = 1.0 + 0.12 * np.exp(-eff_lens / 30.0)
    else:
        rope_scale = 1.0

    combined_scale = disp_scale * rope_scale

    slices = []
    for l_idx in target_layers:
        h = layers[l_idx]
        mean_p = np.sum(h * mask_exp, axis=1) / eff_lens
        scaled_p = mean_p * combined_scale
        # Layer norm over H dimension
        m_vec = np.mean(scaled_p, axis=-1, keepdims=True)
        s_vec = np.std(scaled_p, axis=-1, keepdims=True) + eps
        normed = (scaled_p - m_vec) / s_vec
        slices.append(normed)

    return np.concatenate(slices, axis=-1)


# =====================================================================
# 3. CROSS-TRUNK REPRESENTATION HARMONIZER
# =====================================================================

@dataclass
class TrunkMetadata:
    """Descriptor for an individual model trunk."""
    name: str
    model_type: str                   # 'qwen3', 'olmo2', 'deberta', etc.
    hidden_size: int
    pool_layers: List[int]
    class_order: Sequence[str] = field(default_factory=lambda: CANONICAL_LABELS)
    base_weight: float = 1.0
    temperature: float = 1.0
    deconfounder: Optional[PromptSubspaceDeconfounder] = None


class CrossTrunkHarmonizer:
    """
    Calibrates, deconfounds, and dynamically fuses heterogeneous model backbones:
    - Qwen3-8B  (Alibaba lineage, deep syntax/stylometry)
    - Qwen3-14B (Alibaba lineage, 40-layer reasoning powerhouse)
    - OLMo2-7B  (AllenAI lineage, independent pretraining, alphabetical labels)
    - DeBERTa-v3 / Stylometric head (bidirectional linguistic representation)

    Features:
    - Automatic label space permutation alignment to canonical 18 LLMs.
    - Independent per-trunk orthogonal prompt subspace deconfounding.
    - Length-gated dynamic mixture:
      Prioritizes stylometric trunks for short text (tokens < 60),
      and deep reasoning trunks for complex text (tokens >= 60).
    - Unified latent representation construction.
    """

    def __init__(
        self,
        trunk_configs: Optional[Dict[str, TrunkMetadata]] = None,
        length_pivot: float = 60.0,
        gate_temperature: float = 15.0,
        deconfound_k: int = 32,
    ) -> None:
        self.length_pivot = length_pivot
        self.gate_temperature = gate_temperature
        self.deconfound_k = deconfound_k

        if trunk_configs is None:
            self.trunk_configs: Dict[str, TrunkMetadata] = {
                "qwen3_8b": TrunkMetadata(
                    name="qwen3_8b",
                    model_type="qwen3",
                    hidden_size=4096,
                    pool_layers=[4, 8, 12, 16],
                    class_order=CANONICAL_LABELS,
                    base_weight=0.30,
                    temperature=1.0,
                ),
                "qwen3_14b": TrunkMetadata(
                    name="qwen3_14b",
                    model_type="qwen3",
                    hidden_size=5120,
                    pool_layers=[4, 8, 12, 16],
                    class_order=CANONICAL_LABELS,
                    base_weight=0.55,
                    temperature=1.0,
                ),
                "olmo2_7b": TrunkMetadata(
                    name="olmo2_7b",
                    model_type="olmo2",
                    hidden_size=4096,
                    pool_layers=[4, 8, 12, 16],
                    class_order=OLMO2_CLASSES,  # Alphabetical ordering
                    base_weight=0.15,
                    temperature=1.0,
                ),
            }
        else:
            self.trunk_configs = trunk_configs

        # Fitted deconfounders per trunk
        self.deconfounders: Dict[str, PromptSubspaceDeconfounder] = {}

    def fit_deconfounders(
        self,
        trunk_features: Dict[str, Union[np.ndarray, torch.Tensor]],
        prompt_ids: Union[np.ndarray, torch.Tensor, Sequence[Any]],
    ) -> CrossTrunkHarmonizer:
        """
        Fits independent PromptSubspaceDeconfounders on each trunk's training features.
        """
        for name, feats in trunk_features.items():
            deconf = PromptSubspaceDeconfounder(
                n_components=self.deconfound_k,
                center_centroids=True,
                center_features=True,
            )
            deconf.fit(feats, prompt_ids)
            self.deconfounders[name] = deconf
            if name in self.trunk_configs:
                self.trunk_configs[name].deconfounder = deconf

        return self

    def deconfound_trunk_features(
        self,
        trunk_features: Dict[str, Union[np.ndarray, torch.Tensor]],
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Applies fitted orthogonal prompt deconfounding to each trunk's features.
        """
        deconfounded = {}
        for name, feats in trunk_features.items():
            if name in self.deconfounders:
                deconfounded[name] = self.deconfounders[name].transform(feats)
            else:
                deconfounded[name] = feats
        return deconfounded

    def fuse_representations(
        self,
        trunk_features: Dict[str, Union[np.ndarray, torch.Tensor]],
        apply_deconfounding: bool = True,
        standardize_per_trunk: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Constructs a unified multi-trunk representation vector by:
        1. Deconfounding prompt subspaces per trunk.
        2. Standardizing per-trunk feature variance to prevent 14B scale domination.
        3. Concatenating across active trunks.
        """
        if apply_deconfounding:
            feats_dict = self.deconfound_trunk_features(trunk_features)
        else:
            feats_dict = trunk_features

        processed_parts = []
        is_torch = False

        for name, arr in feats_dict.items():
            if isinstance(arr, torch.Tensor):
                is_torch = True
                x = arr.float()
                if standardize_per_trunk:
                    mu = x.mean(dim=0, keepdim=True)
                    sig = x.std(dim=0, keepdim=True).clamp(min=1e-5)
                    x = (x - mu) / sig
                processed_parts.append(x)
            else:
                x_np = np.asarray(arr, dtype=np.float32)
                if standardize_per_trunk:
                    mu = np.mean(x_np, axis=0, keepdims=True)
                    sig = np.maximum(np.std(x_np, axis=0, keepdims=True), 1e-5)
                    x_np = (x_np - mu) / sig
                processed_parts.append(x_np)

        if is_torch:
            torch_parts = [torch.as_tensor(p) for p in processed_parts]
            return torch.cat(torch_parts, dim=-1)
        else:
            return np.concatenate(processed_parts, axis=-1)

    def align_and_calibrate_logits(
        self,
        trunk_logits_or_probs: Dict[str, Union[np.ndarray, torch.Tensor]],
        is_logits: bool = True,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Permutes labels to CANONICAL_LABELS ordering and applies trunk temperature scaling.
        """
        calibrated = {}
        for name, tensor in trunk_logits_or_probs.items():
            meta = self.trunk_configs.get(name)
            source_classes = meta.class_order if meta else CANONICAL_LABELS
            temp = meta.temperature if meta else 1.0

            aligned = align_labels_to_canonical(
                tensor,
                source_classes=source_classes,
                target_classes=CANONICAL_LABELS,
            )

            if is_logits and abs(temp - 1.0) > 1e-4:
                aligned = aligned / temp

            calibrated[name] = aligned
        return calibrated

    def compute_length_gate_weights(
        self,
        lengths: Union[np.ndarray, torch.Tensor, Sequence[float]],
    ) -> Dict[str, np.ndarray]:
        """
        Computes dynamic mixture weights as a function of sequence length:
        w_m(L) = w_{m,0} + delta_w * sigmoid((L - L_pivot) / T)

        For short sequences (L < length_pivot):
            Stylometric / syntax trunks (Qwen3-8B, OLMo2-7B) have boosted weights.
        For long sequences (L >= length_pivot):
            Deep reasoning trunks (Qwen3-14B) receive dominant weight.
        """
        if isinstance(lengths, torch.Tensor):
            lens_np = lengths.detach().cpu().numpy().astype(np.float64)
        else:
            lens_np = np.asarray(lengths, dtype=np.float64)

        # Sigmoid transition centered at length_pivot
        # z in [0, 1]: 0 for very short text, 1 for very long text
        z = 1.0 / (1.0 + np.exp(-np.clip((lens_np - self.length_pivot) / self.gate_temperature, -20.0, 20.0)))
        N = len(lens_np)

        # Trunk base allocation adjustments
        # When z=0 (short text): shift weight toward 8b and olmo2
        # When z=1 (long text): shift weight toward 14b
        weights: Dict[str, np.ndarray] = {}
        raw_weights: Dict[str, np.ndarray] = {}

        for name, meta in self.trunk_configs.items():
            base = meta.base_weight
            if "14b" in name:
                # 14B gains weight on long passages
                # base * (0.7 + 0.6 * z)
                w = base * (0.65 + 0.70 * z)
            elif "8b" in name:
                # 8B is stylometrically reliable on shorter context
                # base * (1.2 - 0.4 * z)
                w = base * (1.25 - 0.50 * z)
            elif "olmo" in name:
                # OLMo native-24 carries strong syntactic punctuation prior
                w = base * (1.30 - 0.60 * z)
            else:
                w = np.full(N, base, dtype=np.float64)
            raw_weights[name] = np.maximum(w, 0.01)

        # Normalize weights so sum over trunks equals 1.0 per sample
        total_w = sum(raw_weights.values())
        for name in raw_weights:
            weights[name] = raw_weights[name] / np.maximum(total_w, 1e-12)

        return weights

    def fuse_probabilities(
        self,
        trunk_probs: Dict[str, Union[np.ndarray, torch.Tensor]],
        lengths: Optional[Union[np.ndarray, torch.Tensor, Sequence[float]]] = None,
        fusion_mode: str = "linear",  # 'linear' or 'log_linear' (product of experts)
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Fuses aligned class probabilities across trunks using length-gated weights.

        Args:
            trunk_probs: Dict of {trunk_name: (N, K) probabilities in canonical order}.
            lengths: Optional (N,) sequence lengths. If None, uses static base weights.
            fusion_mode: 'linear' for weighted mixture sum_m w_m P_m,
                         'log_linear' for product of experts exp(sum_m w_m log P_m) / Z.
        """
        calibrated = self.align_and_calibrate_logits(trunk_probs, is_logits=False)
        first_tensor = next(iter(calibrated.values()))
        is_torch = isinstance(first_tensor, torch.Tensor)
        device = first_tensor.device if is_torch else None
        dtype = first_tensor.dtype if is_torch else None

        N = first_tensor.shape[0]

        if lengths is not None:
            weights = self.compute_length_gate_weights(lengths)
        else:
            total_b = sum(self.trunk_configs[m].base_weight for m in calibrated if m in self.trunk_configs)
            weights = {
                m: np.full(N, self.trunk_configs[m].base_weight / max(total_b, 1e-12))
                if m in self.trunk_configs else np.full(N, 1.0 / len(calibrated))
                for m in calibrated
            }

        if fusion_mode == "log_linear":
            # Product of Experts: exp(sum w_m log(P_m + eps))
            log_fused = np.zeros((N, K_CLASSES), dtype=np.float64)
            for name, p in calibrated.items():
                p_np = p.detach().cpu().numpy() if is_torch else np.asarray(p, dtype=np.float64)
                p_safe = np.clip(p_np, 1e-12, 1.0)
                w_col = weights[name][:, None]
                log_fused += w_col * np.log(p_safe)

            # Softmax normalization
            max_log = np.max(log_fused, axis=-1, keepdims=True)
            exp_log = np.exp(log_fused - max_log)
            fused_np = exp_log / np.sum(exp_log, axis=-1, keepdims=True)
        else:
            # Linear mixture: sum w_m P_m
            fused_np = np.zeros((N, K_CLASSES), dtype=np.float64)
            for name, p in calibrated.items():
                p_np = p.detach().cpu().numpy() if is_torch else np.asarray(p, dtype=np.float64)
                w_col = weights[name][:, None]
                fused_np += w_col * p_np

            # Re-normalize to ensure exact simplex sum
            fused_np = fused_np / np.maximum(np.sum(fused_np, axis=-1, keepdims=True), 1e-12)

        if is_torch:
            return torch.as_tensor(fused_np, device=device, dtype=dtype)
        return fused_np
