"""Classification losses, reusable composition constraints and the contrastive pretraining loss."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch_focalloss import MultiClassFocalLoss


def merge_candidates(logits, target, cand_mask):
    """Fold every class an (ambiguous) label allows into the target logit via logsumexp and drop the others, so any softmax loss on the target index scores their summed probability; a no-op for unambiguous labels."""
    allowed = cand_mask.index_select(0, target.reshape(-1))
    merged = torch.logsumexp(logits.masked_fill(~allowed, float("-inf")), dim=1, keepdim=True)
    return logits.masked_fill(allowed, float("-inf")).scatter(1, target.reshape(-1, 1), merged)


class CandidateSetLoss(torch.nn.Module):
    """Poly1 cross entropy with label smoothing on the summed probability of every class a label allows (cand_mask[target]).

    Identical to glycowork's Poly1CrossEntropyLoss for unambiguous labels, and to plain cross entropy with
    epsilon=0 and label_smoothing=0. An ambiguous label such as Gal(b1-3/4)GlcNAc is satisfied by probability
    on itself or on any more specific class (Gal(b1-4)GlcNAc), instead of penalizing those as wrong.
    """

    def __init__(self, cand_mask, epsilon=1.0, label_smoothing=0.1):
        super().__init__()
        self.register_buffer("cand_mask", cand_mask)
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing

    def forward(self, output, target):
        log_probs = F.log_softmax(output, dim=1)
        log_p_set = torch.logsumexp(log_probs.masked_fill(~self.cand_mask.index_select(0, target.reshape(-1)), float("-inf")), dim=1)
        loss = -(1 - self.label_smoothing) * log_p_set - self.label_smoothing * log_probs.mean(dim=1) + self.epsilon * (1 - log_p_set.exp())
        return loss.mean()


class FocalLoss(torch.nn.Module):
    """Adapt torch_focalloss to batch reduction and finite fractional-gamma gradients."""

    def __init__(self, gamma=2.0, reduction="mean", cand_mask=None):
        super().__init__()
        self.register_buffer("cand_mask", cand_mask)
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
        self.gamma = gamma
        self.reduction = reduction
        self.primary_loss = MultiClassFocalLoss(gamma=gamma, reduction="none")

    def forward(self, output, target):
        if output.ndim != 2 or target.ndim != 1:
            raise ValueError("Expected logits [batch, classes] and targets [batch].")
        if self.cand_mask is not None:
            output = merge_candidates(output, target, self.cand_mask)
        if 0 < self.gamma < 1:
            target_probabilities = output.softmax(dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
            saturated = target_probabilities == 1
            safe_output = output.masked_fill(saturated.unsqueeze(1), 0.0)
            loss = self.primary_loss(safe_output, target).masked_fill(saturated, 0.0)
        else:
            loss = self.primary_loss(output, target)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class PretrainingContrastiveLoss(torch.nn.Module):
    """Multi-positive InfoNCE (SupCon, Khosla et al. 2020, Eq. 2) for contrastive pretraining on batches of repeated-index views.

    Targets are rows of the pretraining pool, and the views of one row are always positives. A key per row says which spectra may be one
    molecule: composition_ids (supervised; -1 for none) or precursor_mz within mz_tolerance (unsupervised). Keys do not make positives,
    since one composition or precursor mass spans the isomers fine-tuning has to tell apart; instead, spectra of one key in one setting
    group (project, LC, ion mode, derivatization) eluting within same_peak_rt (RT2) of each other are positives, the same isomer in another
    run of the group included; spectra of one key in one run at least isomer_rt apart are negatives (mostly isomers); other pairs of one key
    are left out, as either could hold; and spectra of different keys are negatives. Without keys every other pair is a negative.
    """

    uses_embeddings = True

    def __init__(self, temperature = 0.1, composition_ids = None, precursor_mz = None, mz_tolerance = 0.5, run_ids = None, setting_ids = None,
                 retention_times = None, same_peak_rt = 0.005, isomer_rt = 0.03):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if not 0 <= same_peak_rt < isomer_rt:
            raise ValueError("same_peak_rt must be non-negative and below isomer_rt.")
        if composition_ids is not None and precursor_mz is not None:
            raise ValueError("Pair spectra by composition_ids or by precursor_mz, not both.")
        self.keyed = composition_ids is not None or precursor_mz is not None
        context = {"run_ids": run_ids, "setting_ids": setting_ids, "retention_times": retention_times}
        if self.keyed and any(value is None for value in context.values()):
            raise ValueError("Pairing spectra by composition or precursor m/z needs run_ids, setting_ids and retention_times.")
        self.temperature = temperature
        self.mz_tolerance = mz_tolerance
        self.same_peak_rt = same_peak_rt
        self.isomer_rt = isomer_rt
        buffers = dict(context, composition_ids = composition_ids, precursor_mz = precursor_mz)
        for name, value in buffers.items():
            dtype = torch.float32 if name in {"retention_times", "precursor_mz"} else torch.long
            self.register_buffer(name, None if value is None or not self.keyed else torch.tensor(np.asarray(value), dtype = dtype), persistent = False)

    def pair_types(self, targets):
        """Boolean [batch, batch] masks of each pair's role: same_row (views of one spectrum), same_peak (positives between spectra), isomer
        (negatives within one key) and ignored (pairs left out of the loss)."""
        targets = targets.reshape(-1)
        if self.keyed:
            targets = targets.to(self.run_ids.device)
        same_row = targets.unsqueeze(0) == targets.unsqueeze(1)
        nothing = torch.zeros_like(same_row)
        if not self.keyed:
            return {"same_row": same_row, "same_peak": nothing, "isomer": nothing, "ignored": nothing}
        if self.composition_ids is not None:
            composition = self.composition_ids[targets]
            known = composition >= 0
            same_key = (composition.unsqueeze(0) == composition.unsqueeze(1)) & known.unsqueeze(0) & known.unsqueeze(1)
        else:
            mz = self.precursor_mz[targets]
            same_key = (mz.unsqueeze(0) - mz.unsqueeze(1)).abs() <= self.mz_tolerance
        setting = self.setting_ids[targets]
        run = self.run_ids[targets]
        rt = self.retention_times[targets]
        gap = (rt.unsqueeze(0) - rt.unsqueeze(1)).abs()
        same_peak = same_key & (setting.unsqueeze(0) == setting.unsqueeze(1)) & (gap <= self.same_peak_rt) & ~same_row
        isomer = same_key & (run.unsqueeze(0) == run.unsqueeze(1)) & (gap >= self.isomer_rt) & ~same_row
        ignored = same_key & ~same_row & ~same_peak & ~isomer
        return {"same_row": same_row, "same_peak": same_peak, "isomer": isomer, "ignored": ignored}

    def pair_masks(self, targets):
        """(positive, valid) [batch, batch] masks: valid pairs enter the softmax denominator, positives also its numerator."""
        types = self.pair_types(targets)
        not_self = ~torch.eye(types["same_row"].size(0), dtype = torch.bool, device = types["same_row"].device)
        positive = (types["same_row"] | types["same_peak"]) & not_self
        return positive, not_self & ~types["ignored"]

    def forward(self, features, targets):
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, embedding_dim].")
        targets = targets.reshape(-1)
        if targets.numel() != features.size(0):
            raise ValueError("targets must match the feature batch size.")
        positive, valid = self.pair_masks(targets)
        counts = positive.sum(dim = 1)
        anchors = counts > 0
        if not anchors.any():
            return features.sum() * 0.0
        features = F.normalize(features, dim = 1)
        logits = (features[anchors] @ features.T) / self.temperature
        logits = logits - logits.max(dim = 1, keepdim = True).values.detach()
        log_denominator = torch.logsumexp(logits.masked_fill(~valid[anchors], float("-inf")), dim = 1, keepdim = True)
        log_prob = (logits - log_denominator).masked_fill(~positive[anchors], 0.0)
        return (-log_prob.sum(dim = 1) / counts[anchors]).mean()


class custom_loss(torch.nn.Module):
    def __init__(self, primary_loss, dist_sim, dist_comp, logit_norm=False, t=1.0):
        super(custom_loss, self).__init__()
        self.primary_loss = primary_loss
        self.dist_sim = dist_sim
        self.dist_comp = dist_comp
        self.logit_norm = logit_norm
        self.t = t

    def forward(self, output, target):
        if self.logit_norm:
            norms = torch.norm(output, p=2, dim=-1, keepdim=True) + 1e-7
            output = torch.div(output, norms) / self.t
        loss2 = self.primary_loss(output, target)
        output = torch.nn.functional.softmax(output, dim=1)
        target_sim = self.dist_sim[target]
        loss_sim = output * target_sim
        target_comp = self.dist_comp[target]
        loss_comp = output * target_comp
        loss = loss_comp.mean() + loss_sim.mean() + loss2
        return loss

class CompositionConstraint(torch.nn.Module):
    """Match samples to the full class vocabulary using exact composition counts."""

    def __init__(self, class_compositions):
        super().__init__()
        vectors = torch.as_tensor(class_compositions, dtype=torch.float32)
        if vectors.ndim != 2 or not all(vectors.shape):
            raise ValueError("class_compositions must have shape [classes, composition_features].")
        if not torch.isfinite(vectors).all():
            raise ValueError("Class compositions must contain finite counts.")
        compositions, class_ids = torch.unique(vectors, dim=0, return_inverse=True)
        self.register_buffer("compositions", compositions)
        self.register_buffer("class_ids", class_ids)

    def for_targets(self, targets):
        targets = targets.reshape(-1)
        sample_ids = self.class_ids.index_select(0, targets)
        return sample_ids.unsqueeze(1) == self.class_ids.unsqueeze(0)

    def for_compositions(self, sample_compositions):
        if sample_compositions.ndim != 2 or sample_compositions.size(1) != self.compositions.size(1):
            raise ValueError("Sample composition features must match the class composition vocabulary.")
        matches = (sample_compositions.unsqueeze(1) == self.compositions.unsqueeze(0)).all(dim=-1)
        return matches.index_select(1, self.class_ids)

    def probabilities(self, logits, sample_compositions):
        """Return conditional probabilities, or all zeros for unknown compositions."""
        allowed = self.for_compositions(sample_compositions)
        if allowed.shape != logits.shape:
            raise ValueError("Logits must match the sample count and class vocabulary.")
        masked_logits = logits.masked_fill(~allowed, float("-inf"))
        masked_logits = masked_logits.masked_fill(~allowed.any(dim=1, keepdim=True), 0.0)
        return F.softmax(masked_logits, dim=1).masked_fill(~allowed, 0.0)

class xyz_loss(torch.nn.Module):
    """Restrict standard cross entropy to classes matching the target composition."""

    def __init__(self, class_compositions, logit_norm=False, t=1.0, cand_mask=None):
        super().__init__()
        self.primary_loss = torch.nn.CrossEntropyLoss()
        self.register_buffer("cand_mask", cand_mask)
        self.logit_norm = logit_norm
        self.t = t
        self.composition_constraint = CompositionConstraint(class_compositions)

    def mask_logits(self, output, target):
        allowed = self.composition_constraint.for_targets(target)
        if output.shape != allowed.shape:
            raise ValueError("Logits must match the target count and class composition vocabulary.")
        return output.masked_fill(~allowed, float("-inf"))

    def prepare_logits(self, output, target):
        masked = self.mask_logits(output, target)
        if self.logit_norm:
            allowed = self.composition_constraint.for_targets(target)
            compatible = masked.masked_fill(~allowed, 0.0)
            norms = torch.norm(compatible, p=2, dim=-1, keepdim=True) + 1e-7
            normalized = torch.div(compatible, norms) / self.t
            masked = normalized.masked_fill(~allowed, float("-inf"))
        return masked if self.cand_mask is None else merge_candidates(masked, target, self.cand_mask)

    def forward(self, output, target):
        return self.primary_loss(self.prepare_logits(output, target), target)
