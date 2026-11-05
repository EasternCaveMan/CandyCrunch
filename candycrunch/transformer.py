
from torch import flatten
import torch.nn.functional as F
import torch.nn as nn
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
import inspect
# print(torch.__version__)
# print(inspect.getsource(nn.TransformerEncoderLayer))

# ============================================================
# Normalization
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = (x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt())
        return x * rms * self.weight


def make_norm(norm_type, dim):
    if norm_type == "layer":
        return nn.LayerNorm(dim)

    if norm_type == "rms":
        return RMSNorm(dim)
    raise ValueError(f"Unknown norm_type={norm_type!r}. Use 'layer' or 'rms'.")
# ============================================================
# Activations
# ============================================================

def make_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unknown activation={name!r}.")

# ============================================================
# Normal dense Transformer FFN
# ============================================================

class DenseFeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout=0.2, activation="gelu"):
        super().__init__()
        self.net = nn.Sequential( nn.Linear(d_model, ff_dim),
                                  make_activation(activation),
                                  nn.Dropout(dropout),
                                  nn.Linear(ff_dim, d_model))
    def forward(self, x, padding_mask=None):
        return self.net(x)
    def get_aux_loss(self):
        return None


# ============================================================
# Token-level Transformer MoE
# ============================================================

class MoEFeedForward(nn.Module):
    """
    Token-level Mixture of Experts.

    For each token x:

        p = softmax(router(x))

        S = TopK(p)

        output =
            sum_{e in S} normalized_p_e * Expert_e(x)

    Note
    ----
    This implementation computes all experts first and then gathers the
    selected outputs. It is therefore logically sparse but not
    computationally sparse.
    """

    def __init__(self,d_model, ff_dim, num_experts=4, top_k=2, dropout=0.2, activation="gelu"):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.last_aux_loss = None
        self.last_gate_probs = None
        self.last_top_indices = None

        # Router:
        #
        # R^d_model -> R^num_experts
        #
        self.router = nn.Linear(d_model,num_experts)

        # Each expert is an independent FFN.
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, ff_dim),
                                                    make_activation(activation),
                                                    nn.Dropout(dropout),
                                                    nn.Linear(ff_dim, d_model)) for _ in range(num_experts)])

    def load_balancing_loss(self,gate_probs,top_indices,padding_mask=None,):
        """
        gate_probs:
            [B, T, E]
        top_indices:
            [B, T, K]
        Balance objective:
            L_balance =
                E * sum_e importance_e * load_e
        where
            importance_e =
                mean router probability assigned to expert e
            load_e =
                fraction of routing slots assigned to expert e
        """
        selected_experts = F.one_hot(top_indices,num_classes=self.num_experts).float()
        # Shape before sum:
        #
        # [B, T, K, E]
        #
        # Sum over K:
        #
        # [B, T, E]
        #
        selected_experts = selected_experts.sum(dim=-2)
        if padding_mask is not None:
            # [B, T, 1]
            valid_mask = ((~padding_mask).float().unsqueeze(-1))
            valid_count = (valid_mask.sum().clamp_min(1.0))
            importance = ((gate_probs * valid_mask).sum(dim=(0, 1)) / valid_count)
            load = ((selected_experts * valid_mask) .sum(dim=(0, 1))/ (valid_count * self.top_k))
        else:
            importance = gate_probs.mean(dim=(0, 1))
            load = (selected_experts.mean(dim=(0, 1)) / self.top_k)
        return self.num_experts * torch.sum(importance * load)

    def forward(self,x,padding_mask=None,):
        # ----------------------------------------------------
        # 1. Router logits
        # x:
        #   [B, T, D]
        # gate_logits:
        #   [B, T, E]
        # ----------------------------------------------------
        gate_logits = self.router(x)
        # ----------------------------------------------------
        # 2. Router probabilities
        # ----------------------------------------------------
        gate_probs = torch.softmax(gate_logits,dim=-1,)
        # ----------------------------------------------------
        # 3. Top-K routing
        # both:
        #   [B, T, K]
        # ----------------------------------------------------
        top_weights, top_indices = torch.topk(gate_probs,k=self.top_k,dim=-1)
        # ----------------------------------------------------
        # 4. Renormalize selected experts
        # Sum of selected weights becomes 1.
        # ----------------------------------------------------
        top_weights = (top_weights / top_weights.sum(dim=-1,keepdim=True).clamp_min(1e-9))
        # ----------------------------------------------------
        # 5. Auxiliary balancing loss
        # ----------------------------------------------------
        self.last_aux_loss = (self.load_balancing_loss(gate_probs=gate_probs,
                                                       top_indices=top_indices,
                                                       padding_mask=padding_mask))
        # Store routing information for analysis.
        self.last_gate_probs = gate_probs.detach()
        self.last_top_indices = top_indices.detach()
        # ----------------------------------------------------
        # 6. Compute expert outputs
        #
        # Every expert:
        #
        #   [B,T,D]
        #
        # stack ->
        #
        #   [B,T,E,D]
        # ----------------------------------------------------
        expert_outputs = torch.stack([expert(x) for expert in self.experts],dim=2)
        # ----------------------------------------------------
        # 7. Gather only Top-K expert outputs
        #
        # gather_index:
        #
        #   [B,T,K,D]
        # ----------------------------------------------------
        gather_index = (top_indices.unsqueeze(-1).expand(*top_indices.shape,x.size(-1)))
        selected_outputs = torch.gather(expert_outputs,dim=2,index=gather_index)
        # selected_outputs:
        #
        # [B,T,K,D]
        # ----------------------------------------------------
        # 8. Weighted mixture
        #
        # output:
        #
        # [B,T,D]
        # ----------------------------------------------------
        output = (selected_outputs * top_weights.unsqueeze(-1)).sum(dim=2)
        return output

    def get_aux_loss(self):
        return self.last_aux_loss


# ============================================================
# NEW:
# Spectrum-level classifier Mixture of Experts
# ============================================================

class MoEClassifier(nn.Module):
    """
    Mixture-of-Experts classifier.
    Unlike MoEFeedForward, routing happens once per spectrum/sample.
    Input:
        h : [B, D]
    Router:
        [B, D] -> [B, E]
    Each classifier expert:
        [B, D] -> [B, num_classes]
    Final logits:
        logits =
            sum_{e in TopK}
                gate_e * classifier_e(h)
    """
    def __init__(self,input_dim,num_classes,num_experts=4,top_k=2,hidden_dim=None,dropout=0.0,activation="gelu",):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k
        self.last_aux_loss = None
        self.last_gate_probs = None
        self.last_top_indices = None
        self.last_top_weights = None
        # ----------------------------------------------------
        # Spectrum-level router
        # ----------------------------------------------------
        self.router = nn.Linear(input_dim,num_experts)
        # ----------------------------------------------------
        # Classifier experts
        #
        # Can either be:
        #
        #   Linear:
        #       D -> C
        #
        # or shallow MLP:
        #       D -> H -> C
        # ----------------------------------------------------
        if hidden_dim is None:
            self.experts = nn.ModuleList([nn.Linear(input_dim,num_classes) for _ in range(num_experts)])
        else:
            self.experts = nn.ModuleList([nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                                        make_activation(activation),
                                                        nn.Dropout(dropout),
                                                        nn.Linear(hidden_dim,num_classes)) for _ in range(num_experts)])
    def load_balancing_loss(self, gate_probs, top_indices):
        """
        gate_probs:
            [B, E]

        top_indices:
            [B, K]
        """
        # [B,K,E]
        selected = F.one_hot(top_indices, num_classes=self.num_experts).float()
        # [B,E]
        selected = selected.sum(dim=1)
        # Average probability mass assigned to expert.
        importance = gate_probs.mean(dim=0)
        # Fraction of selected expert slots.
        load = (selected.mean(dim=0) / self.top_k)
        return self.num_experts * torch.sum(importance * load)

    def forward(self,x,return_routing=False,return_selected_logits=True):
        """
        x:
            [B, input_dim]

        Returns:
            logits [B, num_classes]

        Optionally:
            logits,
            routing_dictionary
        """

        # ----------------------------------------------------
        # 1. Router
        # ----------------------------------------------------
        gate_logits = self.router(x)
        # [B,E]
        gate_probs = torch.softmax(gate_logits,dim=-1)
        # ----------------------------------------------------
        # 2. Top-K classifier experts
        # ----------------------------------------------------
        top_weights, top_indices = torch.topk(gate_probs, k=self.top_k,dim=-1)
        # ----------------------------------------------------
        # 3. Renormalize selected router probabilities
        # ----------------------------------------------------
        top_weights = (top_weights / top_weights.sum(dim=-1,keepdim=True).clamp_min(1e-9))
        # ----------------------------------------------------
        # 4. Balancing loss
        # ----------------------------------------------------
        self.last_aux_loss = (self.load_balancing_loss(gate_probs,top_indices))
        self.last_gate_probs = (gate_probs.detach())
        self.last_top_indices = (top_indices.detach())
        self.last_top_weights = (top_weights.detach())
        # ----------------------------------------------------
        # 5. Dispatch only selected samples to each expert.
        #
        # This avoids materializing [B,E,C] logits when only
        # top_k experts contribute to each sample.
        # ----------------------------------------------------
        logits = x.new_zeros(x.size(0),self.num_classes)
        selected_logits = None
        if return_routing and return_selected_logits:
            selected_logits = x.new_zeros(x.size(0),self.top_k,self.num_classes)

        for expert_idx, expert in enumerate(self.experts):
            expert_mask = (top_indices == expert_idx)
            if not expert_mask.any():
                continue

            batch_indices, slot_indices = expert_mask.nonzero(as_tuple=True)
            expert_input = x.index_select(0,batch_indices)
            expert_output = expert(expert_input)
            weighted_output = (
                expert_output
                * top_weights[batch_indices,slot_indices].unsqueeze(-1)
            )
            logits.index_add_(0,batch_indices,weighted_output)

            if return_routing and return_selected_logits:
                selected_logits[batch_indices,slot_indices] = expert_output
        if return_routing:
            routing = {
                "gate_probs": gate_probs,
                "top_indices": top_indices,
                "top_weights": top_weights,
                "expert_batch_sizes": torch.bincount(
                    top_indices.reshape(-1),
                    minlength=self.num_experts,
                ),
            }
            if return_selected_logits:
                routing["selected_logits"] = selected_logits
            return logits, routing
        return logits

    def get_aux_loss(self):
        return self.last_aux_loss


# ============================================================
# Transformer Encoder Layer
# ============================================================

class CandyTransformerEncoderLayer(nn.Module):
    def __init__(self,d_model,nhead,ff_dim,use_transformer_ff=True,encoder_type="dense",norm_type="layer",
        dropout=0.2,activation="gelu",num_experts=4,moe_top_k=2,norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        self.use_transformer_ff = use_transformer_ff
        self.norm1 = make_norm(norm_type,d_model)
        self.norm2 = make_norm(norm_type,d_model)

        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=nhead,
                                          dropout=dropout,
                                          batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if not self.use_transformer_ff:
            self.ff = None
        elif encoder_type == "dense":
            self.ff = DenseFeedForward(d_model=d_model,
                                       ff_dim=ff_dim,
                                       dropout=dropout,
                                       activation=activation)
        elif encoder_type == "moe":
            self.ff = MoEFeedForward(d_model=d_model,
                                     ff_dim=ff_dim,
                                     num_experts=num_experts,
                                     top_k=moe_top_k,
                                     dropout=dropout,
                                     activation=activation)
        else:
            raise ValueError(f"Unknown encoder_type={encoder_type!r}. Use 'dense' or 'moe'.")

    def _mask_padding(self,x,padding_mask,):
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1),0.0)
        return x
    def forward(self,src,src_mask=None,src_key_padding_mask=None,is_causal=False):
        # ====================================================
        # PRE-NORM
        # ====================================================
        if self.norm_first:
            attn_input = self.norm1(src)
            attn_out, _ = self.attn(attn_input,
                                    attn_input,
                                    attn_input,
                                    attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask,
                                    need_weights=False,
                                    is_causal=is_causal)
            src = (src + self.dropout1(attn_out))
            src = self._mask_padding(src,src_key_padding_mask)
            if self.ff is not None:
                ff_out = self.ff(self.norm2(src), padding_mask=src_key_padding_mask)
                src = (src + self.dropout2(ff_out))
                src = self._mask_padding(src,src_key_padding_mask)
        # ====================================================
        # POST-NORM
        # ====================================================
        else:
            attn_out, _ = self.attn(src,
                                    src,
                                    src,
                                    attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask,
                                    need_weights=False,
                                    is_causal=is_causal)

            src = self.norm1(src + self.dropout1(attn_out))
            src = self._mask_padding(src,src_key_padding_mask)
            if self.ff is not None:
                ff_out = self.ff(src,padding_mask=src_key_padding_mask)
                src = self.norm2(src + self.dropout2(ff_out))
                src = self._mask_padding(src,src_key_padding_mask)
        return src

    def get_aux_loss(self):
        if self.ff is None:
            return None
        return self.ff.get_aux_loss()

# ============================================================
# Transformer Encoder
# ============================================================

class CandyTransformerEncoder(nn.Module):
    def __init__(self,encoder_layer,num_layers,norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm
        self.last_aux_loss = None

    def forward(self,src,mask=None,src_key_padding_mask=None,is_causal=False):
        output = src
        aux_losses = []
        for layer in self.layers:
            output = layer(output,src_mask=mask,src_key_padding_mask=src_key_padding_mask,is_causal=is_causal)
            aux_loss = layer.get_aux_loss()
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        if self.norm is not None:
            output = self.norm(output)
        if aux_losses:
            self.last_aux_loss = (torch.stack(aux_losses).mean())
        else:
            self.last_aux_loss = torch.tensor(0.0,device=output.device,dtype=output.dtype)
        return output
    def get_aux_loss(self):
        return self.last_aux_loss


# ============================================================
# CandyCrunch Transformer
# ============================================================

class CandyCrunch_Transformer(nn.Module):
    def __init__(self,num_classes=None, input_precursor_dim=None, heads=None, layers=None, ff_dim=None, peak_dim=2,
        metadata_dim=24, dropout=0.2, peak_encoder="linear", use_transformer_ff=True,
        encoder_type="dense", norm_type="layer", activation="leaky_relu", encoder_activation="gelu",
        peak_hidden_dim=None, lambda_min=10 ** -2.5, lambda_max=10 ** 3.3,
         num_experts=4, moe_top_k=2, classifier_moe=True, classifier_num_experts=4,
        classifier_top_k=2, classifier_expert_hidden_dim=None, norm_first=True,encoder_final_norm=True,
        input_dim=2048, min_mz=39.714, max_mz=3000.0, max_peaks=None, intensity_threshold=0.0, patch_size=4):
        super().__init__()

        if input_dim < 2 or max_mz <= min_mz:
            raise ValueError("input_dim must be at least 2 and max_mz must exceed min_mz.")
        if peak_dim != 2:
            raise ValueError("Binned input uses peak_dim=2: m/z and intensity.")
        if max_peaks is not None or intensity_threshold != 0.0:
            raise ValueError("Fixed-bin input keeps every bin; use max_peaks=None and intensity_threshold=0.0.")
        self.input_dim = input_dim
        self.min_mz = min_mz
        self.max_mz = max_mz
        # Same lower-edge grid used by CNN preprocessing; no learned positions needed.
        self.register_buffer("bin_mz", torch.linspace(min_mz, max_mz, input_dim), persistent=False)

        if heads is None or layers is None:
            raise ValueError("heads and layers must be provided.")

        if (use_transformer_ff and ff_dim is None):
            raise ValueError("ff_dim must be provided use_transformer_ff=True.")

        if peak_hidden_dim is None:
            if ff_dim is not None:
                peak_hidden_dim = ff_dim // 2
            else:
                peak_hidden_dim = heads * 64

        if peak_hidden_dim % heads != 0:
            raise ValueError(
                f"peak_hidden_dim must be divisible by heads. Got peak_hidden_dim= {peak_hidden_dim}, heads={heads}."
            )
        self.peak_encoder = peak_encoder
        self.use_transformer_ff = (use_transformer_ff)
        self.encoder_type = encoder_type
        self.norm_type = norm_type
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.classifier_moe = (classifier_moe)
        # ====================================================
        # Peak encoder
        # ====================================================
        if peak_encoder == "linear":
            self.peak_projection = nn.Sequential(nn.Linear(peak_dim, peak_hidden_dim),
                                                 make_norm(norm_type,peak_hidden_dim),
                                                 make_activation(activation))

        elif peak_encoder == "fourier":
            self.mz_encoding_dim = (peak_hidden_dim)
            self.peak_extra_dim = max(peak_dim - 1,0)
            if self.mz_encoding_dim % 2 != 0:
                raise ValueError("mz_encoding_dim must be even.")

            self.mz_mlp = nn.Sequential(nn.Linear(self.mz_encoding_dim,peak_hidden_dim),
                                        make_norm(norm_type,peak_hidden_dim),
                                        make_activation(activation),
                                        nn.Linear(peak_hidden_dim,peak_hidden_dim),
                                        make_norm(norm_type,peak_hidden_dim),
                                        make_activation(activation))

            peak_mlp_input_dim = (peak_hidden_dim + self.peak_extra_dim)
            self.peak_mlp = nn.Sequential(nn.Linear(peak_mlp_input_dim,peak_hidden_dim),
                                          make_norm(norm_type,peak_hidden_dim),
                                          make_activation(activation),
                                          nn.Dropout(dropout),
                                          nn.Linear(peak_hidden_dim,peak_hidden_dim),
                                          make_norm(norm_type,peak_hidden_dim),
                                          make_activation(activation))

        else:
            raise ValueError(f"Unknown peak_encode {peak_encoder!r}.Use 'linear' or 'fourier'.")
        ##
        # Token reduction: 2048 bin tokens -> 2048 / patch_size tokens
        if input_dim % patch_size != 0:
            raise ValueError(f"input_dim={input_dim} must be divisible by patch_size={patch_size}.")
        self.patch_size = patch_size
        if patch_size > 1:
            # merges patch_size neighbouring bin embeddings into one token (no overlap)
            self.token_reduce = nn.Conv1d(in_channels=peak_hidden_dim, out_channels=peak_hidden_dim,
                                          kernel_size=patch_size, stride=patch_size)
            self.token_reduce_norm = make_norm(norm_type, peak_hidden_dim)
        else:
            self.token_reduce = None
        ##
        self.cls_token = nn.Parameter(torch.zeros(1,1,peak_hidden_dim))

        encoder_layer = (CandyTransformerEncoderLayer(d_model=peak_hidden_dim,
                                                      nhead=heads,
                                                      ff_dim=ff_dim,
                                                      use_transformer_ff=use_transformer_ff,
                                                      encoder_type=encoder_type,
                                                      norm_type=norm_type,
                                                      dropout=dropout,
                                                      activation=encoder_activation,
                                                      num_experts=num_experts,
                                                      moe_top_k=moe_top_k,
                                                      norm_first=norm_first))

        if encoder_final_norm:
            final_norm = make_norm(norm_type,peak_hidden_dim)
        else:
            final_norm = None
        self.transformer = (CandyTransformerEncoder(encoder_layer,num_layers=layers,norm=final_norm))

        self.type_emb = nn.Embedding(5,metadata_dim)

        self.mode_emb = nn.Embedding( 3, metadata_dim)
        self.lc_emb = nn.Embedding( 4,metadata_dim)
        self.modification_emb = nn.Embedding(4,metadata_dim)
        self.trap_emb = nn.Embedding(5,metadata_dim)
        self.prec_block = nn.Sequential(nn.Linear(input_precursor_dim,metadata_dim),
                                        make_norm(norm_type,metadata_dim),
                                        make_activation(activation))

        self.rt_block = nn.Sequential(nn.Linear(1, metadata_dim),
                                      make_norm(norm_type,metadata_dim),
                                      make_activation(activation))

        combined_dim = (peak_hidden_dim + 7 * metadata_dim)
        self.classifier_rep = nn.Sequential(nn.Linear(combined_dim,1024),
                                            make_norm(norm_type,1024),
                                            make_activation(activation),
                                            nn.Dropout(dropout),
                                            nn.Linear(1024,512),
                                            make_norm(norm_type,512),
                                            make_activation(activation),
                                            nn.Dropout(dropout))
        if classifier_moe:
            self.classifier_out = MoEClassifier(input_dim=512,
                                                num_classes=num_classes,
                                                num_experts=classifier_num_experts,
                                                top_k=classifier_top_k,
                                                hidden_dim=classifier_expert_hidden_dim,
                                                dropout=dropout,
                                                activation=activation)
        else:
            self.classifier_out = nn.Linear(512,num_classes)

    def encode_mz(self,mz):
        half_dim = (self.mz_encoding_dim // 2)
        wavelengths = (self.lambda_min * (self.lambda_max / self.lambda_min) **
                       (torch.arange(half_dim,dtype=mz.dtype,device=mz.device,) /
                        max(half_dim - 1,1,)))
        angles = (2.0 * torch.pi * mz / wavelengths.view(1,1,half_dim))
        return torch.cat( [torch.sin(angles),torch.cos(angles)],dim=-1)

    def encode_binned_spectrum(self, spectrum):
        """Convert CNN [B, 2, bins] (intensity, remainder) to [B, bins, 2] (m/z, intensity)."""
        if spectrum.ndim != 3 or spectrum.shape[1:] != (2, self.input_dim):
            raise ValueError(f"Expected binned spectrum [batch, 2, {self.input_dim}].")
        mz = self.bin_mz.to(dtype=spectrum.dtype) + spectrum[:, 1, :]
        return torch.stack([mz, spectrum[:, 0, :]], dim=-1)

    def make_peak_embedding(self,peak_list):
        if self.peak_encoder == "linear":
            return self.peak_projection(peak_list)
        if self.peak_encoder == "fourier":
            mz = peak_list[..., 0:1]
            mz_encoded = (self.encode_mz(mz))
            mz_embedding = ( self.mz_mlp(mz_encoded))
            if self.peak_extra_dim > 0:
                peak_extra = peak_list[...,1:1 + self.peak_extra_dim]
                peak_input = torch.cat([mz_embedding,peak_extra],dim=-1)
                return self.peak_mlp(peak_input)
            return self.peak_mlp(mz_embedding)
        raise ValueError(f"Unknown peak_encoder={self.peak_encoder!r}.")

    def get_transformer_aux_loss(self):
        aux_loss = (self.transformer.get_aux_loss())
        if aux_loss is None:
            return torch.tensor( 0.0,device=self.cls_token.device,dtype=self.cls_token.dtype)
        return aux_loss
    def get_classifier_aux_loss(self):
        if not self.classifier_moe:
            return torch.tensor( 0.0,device=self.cls_token.device,dtype=self.cls_token.dtype)
        aux_loss = (self.classifier_out.get_aux_loss())
        if aux_loss is None:
            return torch.tensor(0.0,device=self.cls_token.device,dtype=self.cls_token.dtype)
        return aux_loss

    def get_aux_loss(self,transformer_weight=1.0,classifier_weight=1.0):
        """
        Combined MoE regularization.

        L_aux =
            transformer_weight * L_transformer_MoE
            +
            classifier_weight * L_classifier_MoE
        """
        transformer_aux = (self.get_transformer_aux_loss())
        classifier_aux = (self.get_classifier_aux_loss())
        return transformer_weight * transformer_aux + classifier_weight * classifier_aux

    def forward(self,mz_features,precursor,glycan_type,rt,mode,lc,modification,trap,
        rep=False,return_routing=False,return_aux_losses=False,return_selected_logits=True):
        """Use CNN binned inputs; optionally return classifier and transformer routes."""
        peak_list = self.encode_binned_spectrum(mz_features)
        peak_features = (self.make_peak_embedding(peak_list))
        ##
        if self.token_reduce is not None:
            # Conv1d expects [B, channels, length]
            peak_features = self.token_reduce(peak_features.transpose(1, 2))       # [B, D, 2048] -> [B, D, 512]
            peak_features = self.token_reduce_norm(peak_features.transpose(1, 2))  # -> [B, 512, D]
        cls_token = self.cls_token.expand(peak_features.size(0), -1, -1)
        ##
        peak_features = torch.cat([cls_token, peak_features], dim=1)
        peak_features = self.transformer(peak_features)
        spectrum_rep = peak_features[:, 0, :]
        glycan_type = (self.type_emb(glycan_type).squeeze(1))
        mode = (self.mode_emb(mode).squeeze(1))
        lc = (self.lc_emb(lc).squeeze(1))
        modification = (self.modification_emb(modification).squeeze(1))
        trap = (self.trap_emb(trap).squeeze(1))
        precursor = (self.prec_block(precursor))
        rt = self.rt_block(rt)
        comb = torch.cat([spectrum_rep,precursor,glycan_type,rt,mode,lc,modification,trap],dim=1)
        comb_rep = (self.classifier_rep(comb))
        if self.classifier_moe:
            if return_routing:
                logits, routing = self.classifier_out(
                    comb_rep,
                    return_routing=True,
                    return_selected_logits=return_selected_logits,
                )
            else:
                logits = (self.classifier_out(comb_rep))
        else:
            logits = (self.classifier_out(comb_rep))
            routing = None

        if return_aux_losses:
            transformer_aux = (self.get_transformer_aux_loss().reshape(1))
            classifier_aux = (self.get_classifier_aux_loss().reshape(1))

        if return_routing and self.use_transformer_ff and self.encoder_type == "moe":
            if routing is None:
                routing = {}
            transformer_routing = []
            for layer in self.transformer.layers:
                top_indices = layer.ff.last_top_indices
                top_weights = layer.ff.last_gate_probs.gather(-1, top_indices)
                top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                transformer_routing.append({"top_indices": top_indices, "top_weights": top_weights})
            routing["transformer_layers"] = transformer_routing

        if rep and return_routing and return_aux_losses:
            return logits,comb_rep,routing,transformer_aux,classifier_aux

        if rep and return_routing:
            return logits,comb_rep,routing

        if rep and return_aux_losses:
            return logits,comb_rep,transformer_aux,classifier_aux

        if rep:
            return logits,comb_rep

        if return_routing and return_aux_losses:
            return logits,routing,transformer_aux,classifier_aux

        if return_routing:
            return logits,routing

        if return_aux_losses:
            return logits,transformer_aux,classifier_aux

        return logits
