from torch import flatten
import torch.nn.functional as F
import torch.nn as nn
import torch
from candycrunch.inference import transform_mz, transform_rt


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, transform_mz = None, transform_rt = None):
        self.x = x
        self.y = y
        self.transform_mz = transform_mz
        self.transform_rt = transform_rt

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        mz = self.x[index][0]
        if self.transform_mz:
            mz = self.transform_mz(mz)
        mz_r = self.x[index][1]
        prec = self.x[index][2]
        glycan_type = self.x[index][3]
        RT = self.x[index][4]
        if self.transform_rt:
            RT = self.transform_rt(RT)
        mode = self.x[index][5]
        lc = self.x[index][6]
        modification = self.x[index][7]
        trap = self.x[index][8]
        out = self.y[index]
        return torch.FloatTensor(mz), torch.FloatTensor(mz_r), torch.FloatTensor(prec), torch.LongTensor(
            [glycan_type]), torch.FloatTensor([RT]), torch.LongTensor([mode]), torch.LongTensor([lc]), torch.LongTensor(
            [modification]), torch.LongTensor([trap]), torch.LongTensor([out])


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
# NEW:
# Spectrum-level classifier Mixture of Experts
# ============================================================

# Glycan class codes, as the models take them: O 0, N 1, free/lipid 2 (prediction.CLASS_CODES); pretraining also uses 3 for spectra whose candidate
# compositions span classes. Fixed routing by glycan class has one classifier expert per class code
GLYCAN_CLASS_EXPERTS = 3


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
            shared(h)                       (only with shared_expert)
            + sum_{e in TopK}
                gate_e * classifier_e(h)

    top_k = num_experts is dense gating: every expert, weighted by the router's softmax, so the routing has no top-k switches (the
    balancing loss is then constant at 1 and has no gradient).
    shared_expert adds an expert that every spectrum uses (DeepSeekMoE's shared experts): each class's weights there learn from all of
    its spectra, and the routed experts add to it.
    fixed_routing replaces the router by expert indices given to forward (expert_indices, e.g. the glycan class): one expert per spectrum
    and no balancing loss; a spectrum whose index names no expert gets the mean of all experts.
    """
    def __init__(self, input_dim, num_classes, num_experts = 4, top_k = 2, hidden_dim = None, dropout = 0.0, activation = "gelu",
                 shared_expert = False, fixed_routing = False):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")
        if fixed_routing and top_k != 1:
            raise ValueError("Fixed routing sends each spectrum to one expert; use top_k=1.")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k
        self.fixed_routing = fixed_routing
        self.last_aux_loss = None
        self.last_gate_probs = None
        self.last_top_indices = None
        self.last_top_weights = None
        # ----------------------------------------------------
        # Spectrum-level router (none with fixed routing)
        # ----------------------------------------------------
        self.router = None if fixed_routing else nn.Linear(input_dim, num_experts)

        # ----------------------------------------------------
        # Classifier experts
        #
        # Can either be:
        #
        #   Linear:
        #       D -> C
        #
        # or shallow MLP (activation is a name, as make_activation takes it):
        #       D -> H -> C
        # ----------------------------------------------------
        def make_expert():
            if hidden_dim is None:
                return nn.Linear(input_dim, num_classes)
            return nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 make_activation(activation),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, num_classes))
        self.experts = nn.ModuleList([make_expert() for _ in range(num_experts)])
        self.shared_expert = make_expert() if shared_expert else None
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

    def fixed_gates(self, x, expert_indices):
        """Fixed routing: gate probabilities [B, E], expert index [B, 1] and weight [B, 1] of each spectrum. A spectrum whose index names no
        expert gets index -1, weight 0 and uniform gate probabilities (forward gives it the mean of all experts)."""
        expert_indices = expert_indices.reshape(-1).to(device = x.device, dtype = torch.long)
        if expert_indices.numel() != x.size(0):
            raise ValueError("expert_indices needs one expert index per spectrum.")
        known = (expert_indices >= 0) & (expert_indices < self.num_experts)
        one_hot = F.one_hot(expert_indices.clamp(0, self.num_experts - 1), num_classes = self.num_experts).to(x.dtype)
        gate_probs = torch.where(known.unsqueeze(1), one_hot, torch.full_like(one_hot, 1.0 / self.num_experts))
        top_indices = torch.where(known, expert_indices, torch.full_like(expert_indices, -1)).unsqueeze(1)
        return gate_probs, top_indices, known.to(x.dtype).unsqueeze(1)

    def forward(self, x, return_routing = False, return_selected_logits = True, expert_indices = None):
        """
        x:
            [B, input_dim]
        expert_indices:
            [B] expert of each spectrum; required with fixed routing, ignored otherwise

        Returns:
            logits [B, num_classes]

        Optionally:
            logits,
            routing_dictionary
        """
        if self.fixed_routing:
            if expert_indices is None:
                raise ValueError("A classifier with fixed routing needs expert_indices.")
            gate_probs, top_indices, top_weights = self.fixed_gates(x, expert_indices)
            # No router, so nothing to balance
            self.last_aux_loss = x.new_zeros(())
        else:
            # ----------------------------------------------------
            # 1. Router
            # ----------------------------------------------------
            gate_logits = self.router(x)
            # [B,E]
            gate_probs = torch.softmax(gate_logits, dim = -1)
            # ----------------------------------------------------
            # 2. Top-K classifier experts
            # ----------------------------------------------------
            top_weights, top_indices = torch.topk(gate_probs, k = self.top_k, dim = -1)
            # ----------------------------------------------------
            # 3. Renormalize selected router probabilities
            # ----------------------------------------------------
            top_weights = (top_weights / top_weights.sum(dim = -1, keepdim = True).clamp_min(1e-9))
            # ----------------------------------------------------
            # 4. Balancing loss
            # ----------------------------------------------------
            self.last_aux_loss = (self.load_balancing_loss(gate_probs, top_indices))
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
        if self.fixed_routing:
            # Spectra whose index names no expert (e.g. the mixed glycan class of pretraining spectra) get the mean of all experts
            unrouted = (top_indices[:, 0] < 0).nonzero(as_tuple = True)[0]
            if unrouted.numel():
                expert_input = x.index_select(0, unrouted)
                mean_output = torch.stack([expert(expert_input) for expert in self.experts]).mean(dim = 0)
                logits.index_add_(0, unrouted, mean_output)
                if return_routing and return_selected_logits:
                    selected_logits[unrouted, 0] = mean_output
        # ----------------------------------------------------
        # 6. Shared expert, used by every spectrum
        # ----------------------------------------------------
        if self.shared_expert is not None:
            logits = logits + self.shared_expert(x)
        if return_routing:
            routing = {
                "gate_probs": gate_probs,
                "top_indices": top_indices,
                "top_weights": top_weights,
                "expert_batch_sizes": torch.bincount(
                    top_indices[top_indices >= 0].reshape(-1),
                    minlength=self.num_experts,
                ),
            }
            if return_selected_logits:
                routing["selected_logits"] = selected_logits
            return logits, routing
        return logits

    def get_aux_loss(self):
        return self.last_aux_loss

class ResUnit2(nn.Module):
    """Daniel new Change"""
    def __init__(self, in_channels, size=3, dilation=1, causal=False, in_ln=True,
                 se=False, se_reduction=8, encoder_activation="leaky_relu"):
        super(ResUnit2, self).__init__()
        self.size = size
        self.dilation = dilation
        self.causal = causal
        self.activation = make_activation(encoder_activation)
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
        x = self.activation(x)
        x = self.activation(self.ln2(self.conv_in(x)))
        x = self.conv_dilated(x)
        if self.causal and self.size > 1:
            x = x[:, :, :-self.dilation * (self.size - 1)]
        x = self.activation(self.ln3(x))
        x = self.conv_out(x)
        if self.se:
            s = x.mean(dim=2, keepdim=True) # (B, 64, L) -> (B, 64, 1)
            s = self.activation(self.se_fc1(s)) # (B, 64, 1) -> (B, 8, 1)
            s = torch.sigmoid(self.se_fc2(s)) # (B, 8, 1) -> (B, 64, 1)
            x = x * s # (B, 64, L) -> (B, 64, L)
        # 8. Residual connection
        out = x + inp
        return out

class CandyCrunch_CNN2(nn.Module):
    """Daniel new Change"""
    def __init__( self,input_dim, num_classes=1, hidden_dim=1024, input_precursor_dim=None, dropout=0.2,
                  se=None, se_reduction=None,norm_type=None,activation=None,encoder_activation=None, classifier_moe=False, classifier_num_experts=4,
                  classifier_top_k=2, classifier_expert_hidden_dim=None, classifier_shared_expert=False, classifier_fixed_routing=False):
        super(CandyCrunch_CNN2, self).__init__()
        if (classifier_shared_expert or classifier_fixed_routing) and not classifier_moe:
            raise ValueError("classifier_shared_expert and classifier_fixed_routing need classifier_moe=True.")
        if classifier_fixed_routing and classifier_num_experts != GLYCAN_CLASS_EXPERTS:
            # With more experts, the ones past the class codes would never be trained
            raise ValueError(f"Fixed routing sends each spectrum to the expert of its glycan class (O, N, free/lipid), so it needs "
                             f"classifier_num_experts={GLYCAN_CLASS_EXPERTS}.")
        self.input_dim = input_dim
        self.classifier_moe = classifier_moe
        self.classifier_fixed_routing = bool(classifier_fixed_routing)
        self.type_emb = nn.Embedding(5, 24)
        self.mode_emb = nn.Embedding(3, 24)
        self.lc_emb = nn.Embedding(4, 24)
        self.modification_emb = nn.Embedding(4, 24)
        self.trap_emb = nn.Embedding(5, 24)
        self.prec_block = nn.Sequential( nn.Linear(input_precursor_dim, 24), make_norm(norm_type,24),  make_activation(activation))
        self.rt_block = nn.Sequential( nn.Linear(1, 24), make_norm(norm_type,24), make_activation(activation))

        self.res_block = nn.Sequential( nn.Conv1d(in_channels=2, out_channels=64, kernel_size=1),
                                        nn.LeakyReLU(),
                                        ResUnit2(64, size=3, dilation=1, causal=False, se=se, se_reduction=se_reduction, encoder_activation=encoder_activation),
                                        ResUnit2(64, size=3, dilation=2, causal=False, se=se, se_reduction=se_reduction,encoder_activation=encoder_activation ),
                                        ResUnit2(64, size=3, dilation=4, causal=False, se=se, se_reduction=se_reduction, encoder_activation=encoder_activation),
                                        ResUnit2(64, size=3, dilation=8, causal=False, se=se, se_reduction=se_reduction, encoder_activation=encoder_activation),
                                        ResUnit2(64, size=3, dilation=16, causal=False, se=se, se_reduction=se_reduction, encoder_activation=encoder_activation),
                                        ResUnit2(64, size=3, dilation=32, causal=False, se=se, se_reduction=se_reduction, encoder_activation=encoder_activation),
                                        nn.AdaptiveMaxPool1d(102))#
        # self.channel_reduce = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)
        self.fc1 = nn.Sequential(nn.Linear(in_features=6528, out_features=hidden_dim),
                                 make_norm(norm_type, hidden_dim),
                                 make_activation(activation),
                                 nn.Dropout(dropout)) #6528
        self.comb_block1 = nn.Sequential(nn.Linear(hidden_dim + 24*7, hidden_dim//2),
                                         make_norm(norm_type, hidden_dim//2),
                                         make_activation(activation),
                                         nn.Dropout(dropout))
        if classifier_moe:
            self.classifier_out = MoEClassifier(input_dim=hidden_dim//2,
                                                num_classes=num_classes,
                                                num_experts=classifier_num_experts,
                                                top_k=classifier_top_k,
                                                hidden_dim=classifier_expert_hidden_dim,
                                                dropout=dropout,
                                                # The activation's name: MoEClassifier makes the module for each MLP expert
                                                activation=activation,
                                                shared_expert=classifier_shared_expert,
                                                fixed_routing=classifier_fixed_routing)
        else:
            self.comb_lin2 = nn.Linear(hidden_dim//2, num_classes)

    # def get_classifier_aux_loss(self):
    #     if not self.classifier_moe:
    #         return torch.tensor(0.0,device=self.fc1.weight.device,dtype=self.fc1.weight.dtype)
    #     aux_loss = self.classifier_out.get_aux_loss()
    #     if aux_loss is None:
    #         return torch.tensor(0.0,device=self.fc1.weight.device,dtype=self.fc1.weight.dtype)
    #
    #     return aux_loss

    def get_classifier_aux_loss(self):
        if not self.classifier_moe:
            return self.fc1[0].weight.new_zeros(())

        aux_loss = self.classifier_out.get_aux_loss()

        if aux_loss is None:
            return self.fc1[0].weight.new_zeros(())

        return aux_loss

    def forward(self, mz_features, precursor, glycan_type, rt, mode, lc, modification, trap, rep=False,
                return_routing=False, return_aux_losses=False, return_selected_logits=True):
        # Fixed routing sends each spectrum to the classifier expert of its glycan class code, taken before the code becomes its embedding
        expert_indices = glycan_type if self.classifier_fixed_routing else None
        glycan_type = self.type_emb(glycan_type).squeeze(1)
        mode = self.mode_emb(mode).squeeze(1)
        lc = self.lc_emb(lc).squeeze(1)
        modification = self.modification_emb(modification).squeeze(1)
        trap = self.trap_emb(trap).squeeze(1)
        precursor = self.prec_block(precursor)
        rt = self.rt_block(rt)
        mz = self.res_block(mz_features)
        mz = flatten(mz, start_dim=1)
        # mz = self.channel_reduce(mz)
        # mz = mz.squeeze(1)
        mz = self.fc1(mz)
        comb = torch.cat([mz,precursor,glycan_type,rt,mode,lc,modification,trap],dim=1)
        comb = self.comb_block1(comb)
        # comb_rep = self.comb_lin1(comb)

        if self.classifier_moe:
            if return_routing:
                logits, routing = self.classifier_out(
                    comb,
                    return_routing=True,
                    return_selected_logits=return_selected_logits,
                    expert_indices=expert_indices,
                )
            else:
                logits = self.classifier_out(comb, expert_indices=expert_indices)
                routing = None
        else:
            logits = self.comb_lin2(comb)
            routing = None

        if return_aux_losses:
            classifier_aux = (self.get_classifier_aux_loss().reshape(1))

        if rep and return_routing and return_aux_losses:
            return logits, comb, routing, classifier_aux
        if rep and return_routing:
            return logits, comb, routing
        if rep and return_aux_losses:
            return logits, comb, classifier_aux
        if rep:
            return logits, comb
        if return_routing and return_aux_losses:
            return logits, routing, classifier_aux
        if return_routing:
            return logits, routing
        if return_aux_losses:
            return logits, classifier_aux
        return logits


#######################################################################################################################
class ResUnit(nn.Module):

    def __init__(self, in_channels, size = 3, dilation = 1, causal = False, in_ln = True):
        super(ResUnit, self).__init__()
        self.size = size
        self.dilation = dilation
        self.causal = causal
        self.in_ln = in_ln
        if self.in_ln:
            self.ln1 = nn.InstanceNorm1d(in_channels, affine = True)
            self.ln1.weight.data.fill_(1.0)
        self.conv_in = nn.Conv1d(in_channels, in_channels // 2, 1)
        self.ln2 = nn.InstanceNorm1d(in_channels // 2, affine = True)
        self.ln2.weight.data.fill_(1.0)
        self.conv_dilated = nn.Conv1d(in_channels // 2, in_channels // 2, size, dilation = self.dilation,
                                      padding = ((dilation * (size - 1)) if causal else (dilation * (size - 1) // 2)))
        self.ln3 = nn.InstanceNorm1d(in_channels // 2, affine = True)
        self.ln3.weight.data.fill_(1.0)
        self.conv_out = nn.Conv1d(in_channels // 2, in_channels, 1)

    def forward(self, inp):
        # Instance norms run as group norms with one channel per group (the same normalization, ~4x faster on CPU) and the convolutions as
        # batched matrix products, the causal one as one product per tap on the shifted input (conv1d is slow for dilated kernel-2 convolutions)
        n = inp.shape[0]
        x = F.group_norm(inp, inp.shape[1], self.ln1.weight, self.ln1.bias, self.ln1.eps) if self.in_ln else inp
        x = F.leaky_relu(x, inplace = self.in_ln)
        x = torch.baddbmm(self.conv_in.bias[None, :, None], self.conv_in.weight[:, :, 0].expand(n, -1, -1), x)
        x = F.leaky_relu(F.group_norm(x, x.shape[1], self.ln2.weight, self.ln2.bias, self.ln2.eps), inplace = True)
        if self.causal:
            w = self.conv_dilated.weight
            y = torch.baddbmm(self.conv_dilated.bias[None, :, None], w[:, :, -1].expand(n, -1, -1), x)
            for tap in range(self.size - 1):
                shift = (self.size - 1 - tap) * self.dilation
                y[:, :, shift:].baddbmm_(w[:, :, tap].expand(n, -1, -1), x[:, :, :-shift])
        else:
            y = self.conv_dilated(x)
        x = F.leaky_relu(F.group_norm(y, y.shape[1], self.ln3.weight, self.ln3.bias, self.ln3.eps), inplace = True)
        return torch.baddbmm(self.conv_out.bias[None, :, None], self.conv_out.weight[:, :, 0].expand(n, -1, -1), x).add_(inp)

class CandyCrunch_CNN(torch.nn.Module):

    def __init__(self, input_dim, num_classes = 1, hidden_dim = 512, input_precursor_dim = None):
        super(CandyCrunch_CNN, self).__init__()

        self.input_dim = input_dim

        self.mz_lin1 = torch.nn.Linear(input_dim, 2 * hidden_dim)  # not used
        self.prec_lin1 = torch.nn.Linear(input_precursor_dim, 24)
        self.rt_lin1 = torch.nn.Linear(1, 24)
        self.comb_lin1 = torch.nn.Linear(2*hidden_dim + 24*7, 2 * 512)
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
