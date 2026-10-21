
from torch import flatten
import torch.nn.functional as F
import torch.nn as nn
import json
import random
from pathlib import Path
import numpy as np
import torch
from candycrunch.inference import transform_mz, transform_rt

class MemmapSpectrumDataset(torch.utils.data.Dataset):
    """Read spectra from a training cache or in-memory inference arrays."""

    def __init__(self,cache_dir,targets,composition_vectors,model_type,transform_mz=None,transform_rt=None,
                 *, arrays=None, repeats=1, default_composition=None):
        if model_type not in {"CNN", "CNN2"}:
            raise ValueError(f"Unknown model_type={model_type!r}. Expected 'CNN' or 'CNN2'.")
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
            array_names.extend(["binned_intensities", "mz_remainder"])
            sample_count = len(arrays["metadata"])
            self._arrays = {name: arrays[name] for name in array_names}
            if any(len(values) != sample_count for values in self._arrays.values()):
                raise ValueError("Inference arrays must have the same number of samples.")
        self.targets = np.zeros(sample_count, dtype=np.int64) if targets is None else np.asarray(targets, dtype=np.int64)
        self.composition_vectors = None if composition_vectors is None else np.asarray(composition_vectors, dtype=np.float32)
        self.default_composition = None if default_composition is None else np.asarray(default_composition, dtype=np.float32)
        self.model_type = model_type
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
        names = ["bin_offsets", "bin_index", "bin_intensity", "bin_remainder"]
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
        return (torch.tensor(mz, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
                torch.tensor(mz_remainder, dtype = torch.float32), *shared)


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
            torch.empty(0, dtype=torch.float32),
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


class CandyCrunch_CNN2(nn.Module):
    """Daniel new Change"""
    def __init__( self, input_dim, num_classes=1, hidden_dim=512, input_precursor_dim=None, dropout=0.2,
                  se=True, se_reduction=8, classifier_moe=False, classifier_num_experts=4,
                  classifier_top_k=2, classifier_expert_hidden_dim=None):
        super(CandyCrunch_CNN2, self).__init__()

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
            classifier_aux = (self.get_classifier_aux_loss().reshape(1))

        if rep and return_routing and return_aux_losses:
            return logits, comb_rep, routing, classifier_aux
        if rep and return_routing:
            return logits, comb_rep, routing
        if rep and return_aux_losses:
            return logits, comb_rep, classifier_aux
        if rep:
            return logits, comb_rep
        if return_routing and return_aux_losses:
            return logits, routing, classifier_aux
        if return_routing:
            return logits, routing
        if return_aux_losses:
            return logits, classifier_aux
        return logits
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
# NEW:
# Spectrum-level classifier Mixture of Experts
# ============================================================

class MoEClassifier(nn.Module):
    """
    Mixture-of-Experts classifier.
    Routing happens once per spectrum/sample.
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



class CandyCrunch_CNN(torch.nn.Module):

    def __init__(self, input_dim, num_classes = 1, hidden_dim = 512, input_precursor_dim = None):
        super(CandyCrunch_CNN, self).__init__()

        self.input_dim = input_dim

        self.mz_lin1 = torch.nn.Linear(input_dim, 2 * hidden_dim)  # not used
        self.prec_lin1 = torch.nn.Linear(input_precursor_dim, 24)
        self.rt_lin1 = torch.nn.Linear(1, 24)
        self.comb_lin1 = torch.nn.Linear(2 * hidden_dim + 24 + 24 + 24 + 24 + 24 + 24 + 24, 2 * 512)
        self.comb_lin2 = torch.nn.Linear(2 * 512, 2 * 256)
        self.comb_lin3 = torch.nn.Linear(2 * 256, num_classes)

        self.type_emb = torch.nn.Embedding(5, 24)
        self.mode_emb = torch.nn.Embedding(3, 24)
        self.lc_emb = torch.nn.Embedding(4, 24)
        self.modification_emb = torch.nn.Embedding(4, 24)
        self.trap_emb = torch.nn.Embedding(5, 24)

        self.conv1 = torch.nn.Conv1d(in_channels = 2, out_channels = 64, kernel_size = 1)
        self.res1 = ResUnit(64, size = 2, dilation = 1, causal = True)
        self.res2 = ResUnit(64, size = 2, dilation = 2, causal = True)
        self.res3 = ResUnit(64, size = 2, dilation = 4, causal = True)
        self.res4 = ResUnit(64, size = 2, dilation = 8, causal = True)
        self.res5 = ResUnit(64, size = 2, dilation = 16, causal = True)
        self.res6 = ResUnit(64, size = 2, dilation = 32, causal = True)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size = 20)
        self.fc1 = torch.nn.Linear(in_features = 6528, out_features = 1024)

        self.mz_bn1 = torch.nn.LayerNorm(2 * hidden_dim)  # not used
        self.prec_bn1 = torch.nn.LayerNorm(24)
        self.rt_bn1 = torch.nn.LayerNorm(24)
        self.comb_bn1 = torch.nn.LayerNorm(2 * 512)
        self.comb_bn2 = torch.nn.LayerNorm(2 * 256)
        self.mz_act1 = torch.nn.LeakyReLU()  # not used
        self.prec_act1 = torch.nn.LeakyReLU()
        self.rt_act1 = torch.nn.LeakyReLU()
        self.comb_act1 = torch.nn.LeakyReLU()
        self.comb_act2 = torch.nn.LeakyReLU()
        self.comb_dp1 = torch.nn.Dropout(0.2)
        self.comb_dp2 = torch.nn.Dropout(0.2)

    def forward(self, mz_list, precursor, glycan_type, rt, mode, lc, modification, trap, rep = False):
        glycan_type = self.type_emb(glycan_type).squeeze(1)
        mode = self.mode_emb(mode).squeeze(1)
        lc = self.lc_emb(lc).squeeze(1)
        modification = self.modification_emb(modification).squeeze(1)
        trap = self.trap_emb(trap).squeeze(1)
        precursor = self.prec_act1(self.prec_bn1(self.prec_lin1(precursor)))
        rt = self.rt_act1(self.rt_bn1(self.rt_lin1(rt)))
        # Convolutional trunk runs per 16 spectra (per-spectrum results are identical): full-batch activations are
        # >100 MB each and get freshly page-faulted on every layer, whereas chunk-sized ones are recycled by the allocator
        mz_chunks = []
        for chunk in mz_list.split(16):
            mz = F.leaky_relu(self.conv1(chunk), inplace = True)
            mz = self.res1(mz)
            mz = self.res2(mz)
            mz = self.res3(mz)
            mz = self.res4(mz)
            mz = self.res5(mz)
            mz = self.res6(mz)
            mz_chunks.append(self.maxpool1(mz))
        mz = F.leaky_relu(self.fc1(flatten(torch.cat(mz_chunks), start_dim = 1)))

        comb = torch.cat([mz, precursor, glycan_type, rt, mode, lc, modification, trap], dim = 1)
        comb = self.comb_dp1(self.comb_act1(self.comb_bn1(self.comb_lin1(comb))))
        comb_rep = self.comb_lin2(comb)
        comb = self.comb_dp2(self.comb_act2(self.comb_bn2(comb_rep)))
        comb = self.comb_lin3(comb)
        if rep:
            return comb, comb_rep
        else:
            return comb
