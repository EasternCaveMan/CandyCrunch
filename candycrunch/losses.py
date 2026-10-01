"""Classification losses and reusable composition constraints."""

import math
from collections import defaultdict

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


def supervised_contrastive_masks(labels):
    labels = labels.reshape(-1)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor.")
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(labels.numel(), device=labels.device, dtype=torch.bool)
    positive_mask = same_class & ~self_mask
    negative_mask = ~same_class
    return positive_mask, negative_mask


class SupConLoss(torch.nn.Module):
    """Supervised contrastive loss from Khosla et al., Eq. 2."""

    uses_embeddings = True

    def __init__(self, temperature=0.07, cand_mask=None):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        self.temperature = temperature
        self.register_buffer("cand_mask", cand_mask)

    def _gather_distributed(self, features, labels):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return features, labels, 0
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return features, labels, 0
        rank = torch.distributed.get_rank()
        gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(labels) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_features, features.contiguous())
        torch.distributed.all_gather(gathered_labels, labels.contiguous())
        gathered_features[rank] = features
        all_features = torch.cat(gathered_features, dim=0)
        all_labels = torch.cat(gathered_labels, dim=0)
        return all_features, all_labels, rank * features.size(0)

    def forward(self, features, labels):
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, embedding_dim].")
        labels = labels.reshape(-1)
        if labels.numel() != features.size(0):
            raise ValueError("labels must match the feature batch size.")

        features = F.normalize(features, dim=1)
        all_features, all_labels, local_offset = self._gather_distributed(features, labels)
        logits = torch.matmul(features, all_features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        local_rows = torch.arange(features.size(0), device=features.device)
        local_columns = local_rows + local_offset
        logits_mask = torch.ones_like(logits, dtype=torch.bool)
        logits_mask[local_rows, local_columns] = False
        positive_mask = labels.unsqueeze(1) == all_labels.unsqueeze(0)
        if self.cand_mask is not None:
            # A different label that one of the two labels allows (Gal(b1-4)GlcNAc for Gal(b1-3/4)GlcNAc) may be the same glycan, so it is neither a positive nor a negative
            maybe_same = self.cand_mask[labels.unsqueeze(1), all_labels.unsqueeze(0)] | self.cand_mask[all_labels.unsqueeze(0), labels.unsqueeze(1)]
            logits_mask = logits_mask & ~(maybe_same & ~positive_mask)
        positive_mask = positive_mask & logits_mask

        positive_counts = positive_mask.sum(dim=1)
        valid_anchors = positive_counts > 0
        if not valid_anchors.any():
            return features.sum() * 0.0

        log_denominator = torch.logsumexp(logits.masked_fill(~logits_mask, float("-inf")), dim=1)
        log_prob = logits - log_denominator.unsqueeze(1)
        anchor_losses = -(log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_counts.clamp_min(1))
        return anchor_losses[valid_anchors].mean()


class ClassAwareContrastiveBatchSampler(torch.utils.data.Sampler):
    """Yield repeated-index batches for two-view supervised contrastive learning."""

    def __init__(
        self,
        targets,
        classes_per_batch=64,
        examples_per_class=4,
        views_per_example=2,
        drop_last=True,
        seed=0,
    ):
        if classes_per_batch < 1:
            raise ValueError("classes_per_batch must be positive.")
        if examples_per_class < 2:
            raise ValueError("examples_per_class must be at least 2.")
        if views_per_example < 1:
            raise ValueError("views_per_example must be positive.")
        self.targets = torch.as_tensor(targets, dtype=torch.long).cpu().numpy()
        self.classes_per_batch = int(classes_per_batch)
        self.examples_per_class = int(examples_per_class)
        self.views_per_example = int(views_per_example)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        by_class = defaultdict(list)
        for index, target in enumerate(self.targets):
            by_class[int(target)].append(index)
        self.class_to_indices = {
            label: torch.as_tensor(indices, dtype=torch.long).numpy()
            for label, indices in by_class.items()
            if len(indices) >= self.examples_per_class
        }
        self.eligible_classes = sorted(self.class_to_indices)
        if not self.eligible_classes:
            raise ValueError("SupCon sampler needs at least one class with enough distinct examples.")
        if self.drop_last and len(self.eligible_classes) < self.classes_per_batch:
            raise ValueError("SupCon sampler has fewer eligible classes than classes_per_batch.")
        self.original_batch_size = self.classes_per_batch * self.examples_per_class
        self.batch_size = self.original_batch_size * self.views_per_example
        eligible_examples = sum(len(self.class_to_indices[label]) for label in self.eligible_classes)
        if self.drop_last:
            self._num_batches = max(1, eligible_examples // self.original_batch_size)
        else:
            self._num_batches = math.ceil(eligible_examples / self.original_batch_size)

    def __len__(self):
        return self._num_batches

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        class_count = len(self.eligible_classes)
        for _ in range(self._num_batches):
            permutation = torch.randperm(class_count, generator=generator).tolist()
            chosen_labels = [self.eligible_classes[index] for index in permutation[: self.classes_per_batch]]
            batch = []
            for label in chosen_labels:
                indices = self.class_to_indices[label]
                positions = torch.randperm(len(indices), generator=generator)[: self.examples_per_class].tolist()
                for position in positions:
                    batch.extend([int(indices[position])] * self.views_per_example)
            yield batch


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
