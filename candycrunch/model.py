
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

def remove_low_intensity_peaks(array, removal_threshold, removal_percentage):
  candidate_indices = np.where(np.logical_and(array > 0.0001, array <= removal_threshold))[0]
  indices_to_remove = np.random.choice(candidate_indices, round(removal_percentage*len(candidate_indices)))
  array_copy = np.copy(array)
  array_copy[indices_to_remove] = 0
  return array_copy


def peak_intensity_jitter(array, augment_intensity):
  return array * np.random.uniform(1 - augment_intensity, 1 + augment_intensity, len(array)).astype(np.float32)


def new_peak_addition(array, n_noise_peaks, max_noise_intensity):
  idx_noise_peaks = np.random.choice(np.where(array == 0)[0], n_noise_peaks)
  new_values = max_noise_intensity * np.random.random(len(idx_noise_peaks))
  noisy_array = np.copy(array)
  noisy_array[idx_noise_peaks] = new_values
  return noisy_array


def transform_mz(x):
  return new_peak_addition(peak_intensity_jitter(remove_low_intensity_peaks(x, removal_threshold = 0.008, removal_percentage = 0.1),
                                                 augment_intensity = 0.25), n_noise_peaks = 10, max_noise_intensity = 0.005)


def rt_jitter(RT):
  return max(0, RT + random.uniform(-0.1, 0.1))


transform_rt = rt_jitter


class MemmapSpectrumDataset(torch.utils.data.Dataset):
    """Read spectra from a training cache or in-memory inference arrays."""

    def __init__(self,cache_dir,targets,composition_vectors,model_type,max_peaks=None,transform_mz=None,transform_rt=None,
                 *, arrays=None, repeats=1, default_composition=None):
        if (cache_dir is None) == (arrays is None):
            raise ValueError("Provide either cache_dir or in-memory arrays.")
        if not isinstance(repeats, (int, np.integer)) or repeats < 1:
            raise ValueError("repeats must be a positive integer.")
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self._arrays = None
        self.manifest = None
        if arrays is None:
            with (Path(cache_dir) / "manifest.json").open() as file:
                self.manifest = json.load(file)
            sample_count = self.manifest["samples"]
        else:
            array_names = ["metadata", "sample_compositions"]
            array_names.extend(["binned_intensities", "mz_remainder"] if model_type == "CNN" else ["peak_list"])
            sample_count = len(arrays["metadata"])
            self._arrays = {name: arrays[name] for name in array_names}
            if any(len(values) != sample_count for values in self._arrays.values()):
                raise ValueError("Inference arrays must have the same number of samples.")
        self.targets = np.zeros(sample_count, dtype=np.int64) if targets is None else np.asarray(targets, dtype=np.int64)
        self.composition_vectors = None if composition_vectors is None else np.asarray(composition_vectors, dtype=np.float32)
        self.default_composition = None if default_composition is None else np.asarray(default_composition, dtype=np.float32)
        self.model_type = model_type
        self.max_peaks = max_peaks
        self.transform_mz = transform_mz
        self.transform_rt = transform_rt
        self.repeats = repeats
        if sample_count != len(self.targets):
            raise ValueError(f"Feature/target length mismatch: {sample_count} != {len(self.targets)}")

    @classmethod
    def for_inference(cls, arrays, model_type, repeats=1, transform_mz=None, transform_rt=None):
        """Use in-memory arrays and per-sample compositions without creating a cache."""
        return cls(None, None, None, model_type, arrays=arrays, repeats=repeats,
                   transform_mz=transform_mz, transform_rt=transform_rt)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.cache_dir is not None:
            state["_arrays"] = None
        return state

    def __len__(self):
        return len(self.targets) * self.repeats

    def _open_arrays(self):
        if self._arrays is not None:
            return self._arrays
        names = ["bin_offsets", "bin_index", "bin_intensity", "bin_remainder"] if self.model_type == "CNN" else ["peak_offsets", "peak_mz", "peak_intensity"]
        if (Path(self.cache_dir) / "sample_compositions.npy").exists():
            names.append("sample_compositions")
        self._arrays = {name: np.load(Path(self.cache_dir) / f"{name}.npy", mmap_mode="r") for name in ["metadata", *names]}
        return self._arrays

    def __getitem__(self, index):
        index = index // self.repeats
        arrays = self._open_arrays()
        metadata = arrays["metadata"][index]
        target = int(self.targets[index])
        if self.composition_vectors is None:
            if "sample_compositions" in arrays:
                composition = arrays["sample_compositions"][index]
            elif self.default_composition is not None:
                composition = self.default_composition
            else:
                raise ValueError("Unlabeled spectrum stores need sample_compositions.npy or default_composition.")
        else:
            composition = self.composition_vectors[target]
        precursor = torch.tensor(composition, dtype=torch.float32)
        retention_time = float(metadata[1])
        if self.transform_rt is not None:
            retention_time = self.transform_rt(retention_time)
        if self.model_type == "CNN":
            if self.manifest is None:
                mz = np.asarray(arrays["binned_intensities"][index])
                mz_remainder = arrays["mz_remainder"][index]
            else:
                # Training stores keep only occupied bins (float16 intensities, m/z remainders as uint16 fractions of a bin), densified here
                start, end = arrays["bin_offsets"][index], arrays["bin_offsets"][index + 1]
                occupied = arrays["bin_index"][start:end]
                mz = np.zeros(self.manifest["num_bins"], dtype=np.float32)
                mz[occupied] = arrays["bin_intensity"][start:end]
                mz_remainder = np.zeros(self.manifest["num_bins"], dtype=np.float32)
                mz_remainder[occupied] = arrays["bin_remainder"][start:end] * self.manifest["remainder_scale"]
            if self.transform_mz is not None:
                mz = self.transform_mz(np.copy(mz))

        shared = (precursor,
                  torch.tensor([int(metadata[0])], dtype=torch.long),
                  torch.tensor([retention_time], dtype=torch.float32),
                  torch.tensor([int(metadata[2])], dtype=torch.long),
                  torch.tensor([int(metadata[3])], dtype=torch.long),
                  torch.tensor([int(metadata[4])], dtype=torch.long),
                  torch.tensor([int(metadata[5])], dtype=torch.long),
                  torch.tensor([target], dtype=torch.long))
        if self.model_type == "CNN":
            return (torch.tensor(mz, dtype=torch.float32),
                    torch.empty(0, dtype=torch.float32),
                    torch.tensor(mz_remainder, dtype = torch.float32), *shared)
        if self.manifest is None:
            peak_list = arrays["peak_list"][index]
        else:
            # Training stores keep only the real m/z-sorted peaks, so the zero padding is restored here
            start, end = arrays["peak_offsets"][index], arrays["peak_offsets"][index + 1]
            peak_list = np.zeros((self.manifest["peak_list_length"], 2), dtype = np.float32)
            peak_list[:end - start, 0] = arrays["peak_mz"][start:end]
            peak_list[:end - start, 1] = arrays["peak_intensity"][start:end]
        if self.max_peaks is not None:
            # Peak lists are sorted by m/z, so keep the most intense peaks (in m/z order, padding last), as spectrum_to_peak_list does at inference
            peak_list = peak_list[np.sort(np.argsort(-peak_list[:, 1], kind="stable")[: self.max_peaks])]
        peak_list = torch.tensor(peak_list, dtype=torch.float32)
        peak_padding_mask = peak_list.abs().sum(dim=-1) == 0
        return peak_list, peak_padding_mask, *shared


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, transform_mz=None, transform_rt=None):
        self.x = x
        self.y = y
        self.transform_mz = transform_mz
        self.transform_rt = transform_rt

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        mz = self.x[index][0]
        peak_list = self.x[index][1]
        mz_r = self.x[index][2]
        prec = self.x[index][3]
        glycan_type = self.x[index][4]
        RT = self.x[index][5]
        mode = self.x[index][6]
        lc = self.x[index][7]
        modification = self.x[index][8]
        trap = self.x[index][9]
        out = self.y[index]

        if self.transform_mz:
            mz = self.transform_mz(mz)

        if self.transform_rt:
            RT = self.transform_rt(RT)

        return (
            torch.FloatTensor(mz),
            torch.FloatTensor(peak_list),
            torch.FloatTensor(mz_r),
            torch.FloatTensor(prec),
            torch.LongTensor([glycan_type]),
            torch.FloatTensor([RT]),
            torch.LongTensor([mode]),
            torch.LongTensor([lc]),
            torch.LongTensor([modification]),
            torch.LongTensor([trap]),
            torch.LongTensor([out]),
        )

class TransDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, transform_rt=None):
        self.x = x
        self.y = y
        self.transform_rt = transform_rt

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        peak_list = self.x[index][1]
        prec = self.x[index][3]
        glycan_type = self.x[index][4]
        RT = self.x[index][5]
        mode = self.x[index][6]
        lc = self.x[index][7]
        modification = self.x[index][8]
        trap = self.x[index][9]
        out = self.y[index]

        if self.transform_rt:
            RT = self.transform_rt(RT)

        peak_list = torch.FloatTensor(peak_list)

        # True means "ignore this token" for PyTorch Transformer padding masks.
        peak_padding_mask = peak_list.abs().sum(dim=-1) == 0

        return (
            peak_list,
            peak_padding_mask,
            torch.FloatTensor(prec),
            torch.LongTensor([glycan_type]),
            torch.FloatTensor([RT]),
            torch.LongTensor([mode]),
            torch.LongTensor([lc]),
            torch.LongTensor([modification]),
            torch.LongTensor([trap]),
            torch.LongTensor([out]),
        )


class PeakResBlock(nn.Module):
    def __init__(self, peak_hidden_dim, dropout=0.2, kernel_size=3, dilations=(1, 2, 4, 8), causal=False):
        super().__init__()
        self.resunits = nn.Sequential(*[ResUnit(in_channels=peak_hidden_dim, size=kernel_size, dilation=dilation,
                                                causal=causal, in_ln=True,) for dilation in dilations])
        self.dropout = nn.Dropout(dropout)
    def forward(self, peak_features, peak_padding_mask=None):
        if peak_padding_mask is not None:
            peak_features = peak_features.masked_fill( peak_padding_mask.unsqueeze(-1), 0.0)
        peak_features = peak_features.transpose(1, 2)
        peak_features = self.resunits(peak_features)
        peak_features = peak_features.transpose(1, 2)
        peak_features = self.dropout(peak_features)
        if peak_padding_mask is not None:
            peak_features = peak_features.masked_fill(peak_padding_mask.unsqueeze(-1),0.0)
        return peak_features

class ResUnit(nn.Module):
    """Daniel new Change"""
    def __init__(self, in_channels, size=3, dilation=1, causal=False, in_ln=True,
                 se=False, se_reduction=8):
        super(ResUnit, self).__init__()
        self.size = size
        self.dilation = dilation
        self.causal = causal
        self.in_ln = in_ln
        # 1. InstanceNorm1d
        self.se = se
        if self.in_ln:
            self.ln1 = nn.InstanceNorm1d(in_channels, affine=True)
            self.ln1.weight.data.fill_(1.0)
        # 2. Bottleneck 1×1 convolution, Reduces channels: C -> C/2
        self.conv_in = nn.Conv1d(in_channels, in_channels // 2, 1)
        # 3. InstanceNorm1d
        self.ln2 = nn.InstanceNorm1d(in_channels // 2, affine=True)
        self.ln2.weight.data.fill_(1.0)
        # 4. Dilated Conv1D
        # 5. Optional causal convolution, the padding part
        padding = dilation * (size - 1) if causal else dilation * (size - 1) // 2
        self.conv_dilated = nn.Conv1d( in_channels // 2, in_channels // 2, size, dilation=dilation, padding=padding)
        # 6. InstanceNorm1d
        self.ln3 = nn.InstanceNorm1d(in_channels // 2, affine=True)
        self.ln3.weight.data.fill_(1.0)
        # 7. Bottleneck 1×1 convolution, Restores channels: C/2 -> C
        self.conv_out = nn.Conv1d(in_channels // 2, in_channels, 1)
        if self.se:
            se_hidden = max(in_channels // se_reduction, 1)
            self.se_fc1 = nn.Conv1d(in_channels, se_hidden, 1)
            self.se_fc2 = nn.Conv1d(se_hidden, in_channels, 1)
    def forward(self, inp):
        x = inp
        if self.in_ln:
            x = self.ln1(x)
        x = nn.functional.leaky_relu(x)
        x = nn.functional.leaky_relu(self.ln2(self.conv_in(x)))
        x = self.conv_dilated(x)
        if self.causal and self.size > 1:
            x = x[:, :, :-self.dilation * (self.size - 1)]
        x = nn.functional.leaky_relu(self.ln3(x))
        x = self.conv_out(x)
        if self.se:
            s = x.mean(dim=2, keepdim=True) # (B, 64, L) -> (B, 64, 1)
            s = nn.functional.leaky_relu(self.se_fc1(s)) # (B, 64, 1) -> (B, 8, 1)
            s = torch.sigmoid(self.se_fc2(s)) # (B, 8, 1) -> (B, 64, 1)
            x = x * s # (B, 64, L) -> (B, 64, L)
        # 8. Residual connection
        out = x + inp
        return out


class CandyCrunch_CNN(nn.Module):
    """Daniel new Change"""
    def __init__( self, input_dim, num_classes=1, hidden_dim=512, input_precursor_dim=None, dropout=0.2,
                  se=True, se_reduction=8, classifier_moe=False, classifier_num_experts=4,
                  classifier_top_k=2, classifier_expert_hidden_dim=None):
        super(CandyCrunch_CNN, self).__init__()

        self.input_dim = input_dim
        self.classifier_moe = classifier_moe
        self.type_emb = nn.Embedding(5, 24)
        self.mode_emb = nn.Embedding(3, 24)
        self.lc_emb = nn.Embedding(4, 24)
        self.modification_emb = nn.Embedding(4, 24)
        self.trap_emb = nn.Embedding(5, 24)
        self.prec_block = nn.Sequential( nn.Linear(input_precursor_dim, 24), nn.LayerNorm(24), nn.LeakyReLU())
        self.rt_block = nn.Sequential( nn.Linear(1, 24), nn.LayerNorm(24), nn.LeakyReLU())

        self.res_block = nn.Sequential( nn.Conv1d(in_channels=2, out_channels=64, kernel_size=1),
                                        nn.LeakyReLU(),
                                        ResUnit(64, size=3, dilation=1, causal=False, se=se, se_reduction=se_reduction),
                                        ResUnit(64, size=3, dilation=2, causal=False, se=se, se_reduction=se_reduction),
                                        ResUnit(64, size=3, dilation=4, causal=False, se=se, se_reduction=se_reduction),
                                        ResUnit(64, size=3, dilation=8, causal=False, se=se, se_reduction=se_reduction),
                                        ResUnit(64, size=3, dilation=16, causal=False, se=se, se_reduction=se_reduction),
                                        ResUnit(64, size=3, dilation=32, causal=False, se=se, se_reduction=se_reduction),
                                        nn.AdaptiveMaxPool1d(102))
        self.fc_dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features=6528, out_features=1024)
        self.comb_block1 = nn.Sequential(nn.Linear( 2 * hidden_dim + 24 + 24 + 24 + 24 + 24 + 24 + 24, 2 * 512,),
                                         nn.LayerNorm(2 * 512),
                                         nn.LeakyReLU(),
                                         nn.Dropout(dropout))

        self.comb_lin1 = nn.Linear(2 * 512, 2 * 256)
        self.comb_block2 = nn.Sequential(nn.LayerNorm(2 * 256),
                                         nn.LeakyReLU(),
                                         nn.Dropout(dropout))

        if classifier_moe:
            self.classifier_out = MoEClassifier(input_dim=2 * 256,
                                                num_classes=num_classes,
                                                num_experts=classifier_num_experts,
                                                top_k=classifier_top_k,
                                                hidden_dim=classifier_expert_hidden_dim,
                                                dropout=dropout,
                                                activation="leaky_relu")
        else:
            self.comb_lin2 = nn.Linear(2 * 256, num_classes)

    def get_transformer_aux_loss(self):
        return torch.tensor(0.0,device=self.fc1.weight.device,dtype=self.fc1.weight.dtype)
    def get_classifier_aux_loss(self):
        if not self.classifier_moe:
            return torch.tensor(0.0,device=self.fc1.weight.device,dtype=self.fc1.weight.dtype)
        aux_loss = self.classifier_out.get_aux_loss()
        if aux_loss is None:
            return torch.tensor(0.0,device=self.fc1.weight.device,dtype=self.fc1.weight.dtype)

        return aux_loss

    def forward(self, mz_features, precursor, glycan_type, rt, mode, lc, modification, trap, rep=False,
                return_routing=False, return_aux_losses=False, return_selected_logits=True):
        glycan_type = self.type_emb(glycan_type).squeeze(1)
        mode = self.mode_emb(mode).squeeze(1)
        lc = self.lc_emb(lc).squeeze(1)
        modification = self.modification_emb(modification).squeeze(1)
        trap = self.trap_emb(trap).squeeze(1)
        precursor = self.prec_block(precursor)
        rt = self.rt_block(rt)
        mz = self.res_block(mz_features)
        mz = flatten(mz, start_dim=1)
        mz = self.fc_dropout(mz)
        mz = F.leaky_relu(self.fc1(mz))
        comb = torch.cat([mz,precursor,glycan_type,rt,mode,lc,modification,trap],dim=1)
        comb = self.comb_block1(comb)
        comb_rep = self.comb_lin1(comb)
        comb = self.comb_block2(comb_rep)

        if self.classifier_moe:
            if return_routing:
                logits, routing = self.classifier_out(
                    comb,
                    return_routing=True,
                    return_selected_logits=return_selected_logits,
                )
            else:
                logits = self.classifier_out(comb)
                routing = None
        else:
            logits = self.comb_lin2(comb)
            routing = None

        if return_aux_losses:
            transformer_aux = (self.get_transformer_aux_loss().reshape(1))
            classifier_aux = (self.get_classifier_aux_loss().reshape(1))

        if rep and return_routing and return_aux_losses:
            return logits, comb_rep, routing, transformer_aux, classifier_aux
        if rep and return_routing:
            return logits, comb_rep, routing
        if rep and return_aux_losses:
            return logits, comb_rep, transformer_aux, classifier_aux
        if rep:
            return logits, comb_rep
        if return_routing and return_aux_losses:
            return logits, routing, transformer_aux, classifier_aux
        if return_routing:
            return logits, routing
        if return_aux_losses:
            return logits, transformer_aux, classifier_aux
        return logits


# class CandyCrunch_CNN(nn.Module):
#     """Daniel new Change"""
#     def __init__( self, input_dim, num_classes=1, hidden_dim=512, input_precursor_dim=None, dropout=0.2,
#                   se=True, se_reduction=8):
#         super(CandyCrunch_CNN, self).__init__()
#
#         self.input_dim = input_dim
#         self.type_emb = nn.Embedding(5, 24)
#         self.mode_emb = nn.Embedding(3, 24)
#         self.lc_emb = nn.Embedding(4, 24)
#         self.modification_emb = nn.Embedding(4, 24)
#         self.trap_emb = nn.Embedding(5, 24)
#         self.prec_block = nn.Sequential( nn.Linear(input_precursor_dim, 24), nn.LayerNorm(24), nn.LeakyReLU())
#         self.rt_block = nn.Sequential( nn.Linear(1, 24), nn.LayerNorm(24), nn.LeakyReLU())
#
#         self.res_block = nn.Sequential( nn.Conv1d(in_channels=2, out_channels=64, kernel_size=1),
#                                         nn.LeakyReLU(),
#                                         ResUnit(64, size=3, dilation=1, causal=False, se=se, se_reduction=se_reduction),
#                                         ResUnit(64, size=3, dilation=2, causal=False, se=se, se_reduction=se_reduction),
#                                         ResUnit(64, size=3, dilation=4, causal=False, se=se, se_reduction=se_reduction),
#                                         ResUnit(64, size=3, dilation=8, causal=False, se=se, se_reduction=se_reduction),
#                                         ResUnit(64, size=3, dilation=16, causal=False, se=se, se_reduction=se_reduction),
#                                         ResUnit(64, size=3, dilation=32, causal=False, se=se, se_reduction=se_reduction),
#                                         nn.AdaptiveMaxPool1d(102))
#         self.fc_dropout = nn.Dropout(dropout)
#         self.fc1 = nn.Linear(in_features=6528, out_features=1024)
#         self.comb_block1 = nn.Sequential(nn.Linear( 2 * hidden_dim + 24 + 24 + 24 + 24 + 24 + 24 + 24, 2 * 512,),
#                                          nn.LayerNorm(2 * 512),
#                                          nn.LeakyReLU(),
#                                          nn.Dropout(dropout))
#
#         self.comb_lin1 = nn.Linear(2 * 512, 2 * 256)
#         self.comb_block2 = nn.Sequential(nn.LayerNorm(2 * 256),
#                                          nn.LeakyReLU(),
#                                          nn.Dropout(dropout))
#         self.comb_lin2 = nn.Linear(2 * 256, num_classes)
#
#     def forward(self, mz_features, precursor, glycan_type, rt, mode, lc, modification, trap, rep=False):
#         glycan_type = self.type_emb(glycan_type).squeeze(1)
#         mode = self.mode_emb(mode).squeeze(1)
#         lc = self.lc_emb(lc).squeeze(1)
#         modification = self.modification_emb(modification).squeeze(1)
#         trap = self.trap_emb(trap).squeeze(1)
#         precursor = self.prec_block(precursor)
#         rt = self.rt_block(rt)
#         mz = self.res_block(mz_features)
#         mz = flatten(mz, start_dim=1)
#         mz = self.fc_dropout(mz)
#         mz = F.leaky_relu(self.fc1(mz))
#         comb = torch.cat([mz,precursor,glycan_type,rt,mode,lc,modification,trap],dim=1)
#         comb = self.comb_block1(comb)
#         comb_rep = self.comb_lin1(comb)
#         comb = self.comb_block2(comb_rep)
#         comb = self.comb_lin2(comb)
#         if rep:
#             return comb, comb_rep
#         else:
#             return comb
#########################################################################################################
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
        peak_hidden_dim=None, lambda_min=10 ** -2.5, lambda_max=10 ** 3.3, use_resunits=False,
        res_kernel_size=3, res_dilations=(1, 2, 4, 8), res_causal=False, num_experts=4, moe_top_k=2,
        classifier_moe=True, classifier_num_experts=4, classifier_top_k=2, classifier_expert_hidden_dim=None,
        norm_first=True,encoder_final_norm=True):
        super().__init__()

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
        self.use_resunits = use_resunits
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
        if self.use_resunits:
            self.peak_res_block = PeakResBlock(peak_hidden_dim=peak_hidden_dim,
                                               dropout=dropout,
                                               kernel_size=res_kernel_size,
                                               dilations=res_dilations,
                                               causal=res_causal)
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

    def forward(self,peak_list,peak_padding_mask,precursor,glycan_type,rt,mode,lc,modification,trap,
        rep=False,return_routing=False,return_aux_losses=False,return_selected_logits=True):
        """Optionally return classifier routes and token routes under transformer_layers."""
        batch_size = peak_list.size(0)
        peak_features = (self.make_peak_embedding(peak_list))
        if self.use_resunits:
            peak_features = (self.peak_res_block(peak_features,peak_padding_mask=(peak_padding_mask)))
        # cls_token = (self.cls_token.expand(batch_size, -1, -1, ))
        # peak_features = torch.cat([cls_token, peak_features, ], dim=1)
        # # CLS is never padded.
        # cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=(peak_padding_mask.device))
        # transformer_padding_mask = (torch.cat([cls_padding, peak_padding_mask], dim=1))
        # peak_features = (self.transformer(peak_features, src_key_padding_mask=(transformer_padding_mask)))
        # spectrum_rep = (peak_features[:, 0, :])
        peak_features = (self.transformer(peak_features,src_key_padding_mask=(peak_padding_mask)))
        if peak_padding_mask is not None:
            valid_mask = (~peak_padding_mask).unsqueeze(-1)
            masked_peak_features = peak_features.masked_fill(peak_padding_mask.unsqueeze(-1),0.0)
            token_count = valid_mask.sum(dim=1).clamp_min(1).to(peak_features.dtype)
            spectrum_rep = (masked_peak_features.sum(dim=1) / token_count)
        else:
            spectrum_rep = peak_features.mean(dim=1)
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
