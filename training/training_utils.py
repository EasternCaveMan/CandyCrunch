#new code
import copy
import json
import numpy as np
import time
import torch
import wandb
import logging
import warnings
import matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.exceptions import UndefinedMetricWarning
from glycowork.ml.model_training import EarlyStopping, disable_running_stats, enable_running_stats
from torchmetrics.functional import accuracy
warnings.filterwarnings("ignore", category = UndefinedMetricWarning)
warnings.filterwarnings("ignore", category = UserWarning, module = "sklearn")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


def calculate_class_distribution(dataloader, num_classes):
    """Calculate class distribution in a dataloader"""
    targets = getattr(dataloader.dataset, "targets", None)
    if targets is not None:
        targets = np.asarray(targets, dtype=np.int64)
        valid_targets = targets[(targets >= 0) & (targets < num_classes)]
        return (
            np.bincount(valid_targets, minlength=num_classes).astype(np.float64),
            len(valid_targets),
        )

    class_counts = np.zeros(num_classes)
    total_samples = 0

    for data in dataloader:
        y = data[-1].squeeze()
        for label in y.cpu().numpy():
            if label < num_classes:
                class_counts[int(label)] += 1
                total_samples += 1

    return class_counts, total_samples


def _accumulate_classifier_routing(
    class_sample_counts,
    expert_selection_counts,
    expert_responsibility_sums,
    labels,
    routing,
):
    """Accumulate classifier-expert routing statistics."""
    stats_device = class_sample_counts.device
    labels = labels.detach().reshape(-1).to(device=stats_device, dtype=torch.long)
    top_indices = routing["top_indices"].detach().to(device=stats_device, dtype=torch.long)
    top_weights = routing["top_weights"].detach().to(
        device=stats_device,
        dtype=expert_responsibility_sums.dtype,
    )

    if top_indices.ndim != 2 or top_weights.shape != top_indices.shape:
        raise ValueError("Classifier routing tensors must both have shape [batch, top_k].")
    if top_indices.size(0) != labels.numel():
        raise ValueError("Classifier routing batch size does not match the label batch size.")

    num_classes = class_sample_counts.numel()
    num_experts = expert_selection_counts.size(1)
    valid_samples = (labels >= 0) & (labels < num_classes)

    labels = labels[valid_samples]
    top_indices = top_indices[valid_samples]
    top_weights = top_weights[valid_samples]

    if labels.numel() == 0:
        return

    expanded_labels = labels.unsqueeze(1).expand_as(top_indices)
    valid_routes = (top_indices >= 0) & (top_indices < num_experts)
    route_classes = expanded_labels[valid_routes]
    route_experts = top_indices[valid_routes]

    class_sample_counts.index_add_(
        0,
        labels,
        torch.ones_like(labels, dtype=class_sample_counts.dtype),
    )
    expert_selection_counts.index_put_(
        (route_classes, route_experts),
        torch.ones_like(route_classes, dtype=expert_selection_counts.dtype),
        accumulate=True,
    )
    expert_responsibility_sums.index_put_(
        (route_classes, route_experts),
        top_weights[valid_routes],
        accumulate=True,
    )


def _accumulate_transformer_routing(
    class_token_counts,
    expert_selection_counts,
    expert_responsibility_sums,
    labels,
    routing,
    padding_mask,
):
    """Aggregate token routing by true class, excluding padded peaks."""
    top_indices = routing["top_indices"]
    top_weights = routing["top_weights"]
    if top_indices.ndim != 3 or top_weights.shape != top_indices.shape:
        raise ValueError("Transformer routing tensors must have shape [batch, tokens, top_k].")
    labels = labels.detach().reshape(-1).to(top_indices.device)
    if labels.numel() != top_indices.size(0):
        raise ValueError("Transformer routing batch size does not match the label batch size.")
    valid_tokens = torch.ones_like(top_indices[..., 0], dtype=torch.bool)
    if padding_mask is not None:
        if padding_mask.shape != valid_tokens.shape:
            raise ValueError("Transformer padding mask must have shape [batch, tokens].")
        valid_tokens = ~padding_mask.to(device=top_indices.device, dtype=torch.bool)
    token_labels = labels.unsqueeze(1).expand_as(valid_tokens)
    _accumulate_classifier_routing(
        class_token_counts,
        expert_selection_counts,
        expert_responsibility_sums,
        token_labels[valid_tokens],
        {"top_indices": top_indices[valid_tokens], "top_weights": top_weights[valid_tokens]},
    )


def _classifier_routing_wandb_payload(
    phase,
    epoch,
    glycans,
    class_sample_counts,
    expert_selection_counts,
    expert_responsibility_sums,
    max_heatmap_classes=50,
    routing_name="classifier_routing",
    routing_title="classifier",
    count_label="Samples",
    glycan_compositions=None,
):
    """Build W&B tables, clustered heatmaps, and expert-load metrics for one epoch."""
    class_sample_counts = class_sample_counts.detach().cpu()
    expert_selection_counts = expert_selection_counts.detach().cpu()
    expert_responsibility_sums = expert_responsibility_sums.detach().cpu()
    present_classes = torch.nonzero(class_sample_counts > 0, as_tuple=False).flatten()
    if present_classes.numel() == 0:
        return {}

    denominators = class_sample_counts.clamp_min(1).unsqueeze(1).to(torch.float64)
    selection_rates = expert_selection_counts.to(torch.float64) / denominators
    responsibilities = expert_responsibility_sums.to(torch.float64) / denominators
    dominant_experts = responsibilities.argmax(dim=1)
    num_experts = expert_selection_counts.size(1)

    columns = [
        "Class Index",
        "Class",
        count_label,
        "Dominant Expert",
    ]
    columns.extend(
        f"Expert {expert_idx} Responsibility"
        for expert_idx in range(num_experts)
    )
    columns.extend(
        f"Expert {expert_idx} Selection Rate"
        for expert_idx in range(num_experts)
    )

    table_rows = []
    for class_idx_tensor in present_classes:
        class_idx = int(class_idx_tensor.item())
        class_name = (
            str(glycans[class_idx])
            if class_idx < len(glycans)
            else f"Class_{class_idx}"
        )
        table_rows.append([
            class_idx,
            class_name,
            int(class_sample_counts[class_idx].item()),
            int(dominant_experts[class_idx].item()),
            *responsibilities[class_idx].tolist(),
            *selection_rates[class_idx].tolist(),
        ])

    payload = {
        f"{phase}/{routing_name}/by_class": wandb.Table(
            data=table_rows,
            columns=columns,
        ),
    }

    total_samples = class_sample_counts.sum().clamp_min(1).to(torch.float64)
    expert_selection_rates = expert_selection_counts.sum(dim=0).to(torch.float64) / total_samples
    expert_responsibilities = (
        expert_responsibility_sums.sum(dim=0).to(torch.float64) / total_samples
    )
    for expert_idx in range(num_experts):
        payload[
            f"{phase}/{routing_name}/expert_{expert_idx}_selection_rate"
        ] = float(expert_selection_rates[expert_idx].item())
        payload[
            f"{phase}/{routing_name}/expert_{expert_idx}_responsibility"
        ] = float(expert_responsibilities[expert_idx].item())

    heatmap_class_indices = sorted(
        present_classes.tolist(),
        key=lambda class_idx: int(class_sample_counts[class_idx].item()),
        reverse=True,
    )[:max_heatmap_classes]
    if not heatmap_class_indices:
        return payload
    heatmap_values = responsibilities[heatmap_class_indices].numpy()
    heatmap_class_names = glycans if glycan_compositions is None else glycan_compositions
    heatmap_labels = []
    for class_idx in heatmap_class_indices:
        class_name = (
            str(heatmap_class_names[class_idx])
            if class_idx < len(heatmap_class_names)
            else f"Class_{class_idx}"
        )
        heatmap_labels.append(f"{class_idx}: {class_name[:40]}")

    figure_height = max(5.0, min(20.0, 0.32 * len(heatmap_class_indices) + 2.0))
    cluster_grid = sns.clustermap(
        heatmap_values,
        method="average",
        metric="euclidean",
        row_cluster=len(heatmap_class_indices) > 1,
        col_cluster=num_experts > 1,
        xticklabels=[f"Expert {expert_idx}" for expert_idx in range(num_experts)],
        yticklabels=heatmap_labels,
        figsize=(max(10.0, 1.25 * num_experts + 5.0), figure_height),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Mean normalized routing weight"},
    )
    cluster_grid.ax_heatmap.tick_params(axis="y", labelsize=7)
    cluster_grid.ax_heatmap.tick_params(axis="x", labelrotation=45)
    cluster_grid.ax_heatmap.set_xlabel(f"{routing_title.capitalize()} expert")
    cluster_grid.ax_heatmap.set_ylabel(
        "True class (clustered)" if glycan_compositions is None else "Glycan composition (clustered)"
    )
    cluster_grid.fig.subplots_adjust(top=0.92, bottom=max(0.08, 1.1 / figure_height))
    cluster_grid.ax_cbar.set_position(cluster_grid.cbar_pos)
    cluster_grid.fig.suptitle(
        f"{phase.title()} {routing_title} routing responsibility — epoch {epoch}",
        y=0.99,
    )
    try:
        payload[f"{phase}/{routing_name}/top_classes_heatmap"] = wandb.Image(cluster_grid.fig)
    finally:
        plt.close(cluster_grid.fig)

    return payload

def prepare_batch(data, model_type):

    if model_type == "CNN":
        mz_list,peak_list,mz_remainder,precursor,glycan_type,rt,mode_in,lc,modification,trap,y= data
        mz_features = torch.stack([mz_list, mz_remainder], dim=1).to(device)
        inputs = [mz_features,precursor,glycan_type,rt,mode_in,lc,modification,trap]
        inputs = [x.to(device) for x in inputs]
        y = y.squeeze().to(device)

    elif model_type == "Transformer":
        peak_list,peak_padding_mask,precursor,glycan_type,rt,mode_in,lc,modification,trap,y= data
        inputs = [peak_list,peak_padding_mask,precursor,glycan_type,rt,mode_in,lc,modification,trap]
        inputs = [x.to(device) for x in inputs]
        y = y.squeeze().to(device)
    else:
        raise ValueError(f"Unknown model_type={model_type!r}. Expected 'CNN' or 'Transformer'.")
    return inputs, y


def calculate_metrics_sklearn_from_classes(
    pred_classes, y, num_classes, batch_size_small=True, present_labels_only=False
):
    pred_classes_np = pred_classes.cpu().numpy()
    y_np = y.cpu().numpy()

    if present_labels_only:
        labels = np.unique(y_np)
    else:
        labels = range(num_classes)

    try:
        if batch_size_small or len(np.unique(y_np)) < 2:
            f1_micro = f1_score(y_np, pred_classes_np, average='micro', zero_division=0)
            f1_macro = f1_micro
            f1_weighted = f1_micro
        else:
            f1_macro = f1_score(
                y_np, pred_classes_np,
                average='macro',
                zero_division=0,
                labels=labels
            )
            f1_weighted = f1_score(
                y_np, pred_classes_np,
                average='weighted',
                zero_division=0,
                labels=labels
            )
            f1_micro = f1_score(y_np, pred_classes_np, average='micro', zero_division=0)
    except Exception as e:
        f1_macro = f1_weighted = f1_micro = 0.0

    # Calculate MCC (returns 0 if only one class present)
    try:
        if len(np.unique(y_np)) > 1:
            mcc = matthews_corrcoef(y_np, pred_classes_np)
        else:
            mcc = 0.0
    except Exception as e:
        mcc = 0.0

    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_micro': f1_micro,
        'mcc': mcc
    }


def calculate_metrics_sklearn(pred, y, num_classes, batch_size_small=True, present_labels_only=False):
    return calculate_metrics_sklearn_from_classes(
        torch.argmax(pred, dim=1),
        y,
        num_classes,
        batch_size_small=batch_size_small,
        present_labels_only=present_labels_only,
    )


def train_model(model,dataloaders,criterion,optimizer,scheduler,glycans,num_epochs=None,
                patience=None,log_to_wandb=True,num_classes=None,model_type=None,setting_name=None,
                transformer_moe_aux_loss_weight=0.0,classifier_moe_aux_loss_weight=0.0,checkpoint_metadata=None,
                memory_reporter=None,glycan_compositions=None,save_dir="./models",wandb_prefix=None,
                contrastive_criterion=None,contrastive_loss_weight=0.0,):
    """
    Train a CNN or Transformer with SAM.

    CNN models have no token-level MoE. Their objective is classification
    loss plus optional classifier MoE (mixture of classifiers) load balancing.
    Transformer models can additionally use token-level MoE load balancing
    when their feed-forward layers are enabled and encoder_type is "moe".

    Both models share the forward output (predictions, transformer_aux,
    classifier_aux) when return_aux_losses=True. The CNN's transformer_aux
    is a zero placeholder; disabled components do not enter the objective.
    Each active auxiliary loss is scaled by its corresponding weight.

    When contrastive_criterion is provided, fine-tuning optimizes the selected
    classification loss plus contrastive_loss_weight times supervised
    contrastive loss over the model embeddings.

    Both SAM passes use auxiliary losses from that same forward pass.
    Validation reports classification loss alone for early stopping.
    Routing statistics are optional WandB diagnostics and do not control
    regularization. Transformer routing is aggregated over unpadded tokens
    separately for each encoder layer.
    """
    since = time.time()
    early_stopping = EarlyStopping(patience=patience,verbose=True)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.0
    best_acc = 0.0
    val_losses = []
    val_acc = []
    train_losses = []
    train_acc = []
    contrastive_enabled = contrastive_criterion is not None and contrastive_loss_weight > 0
    # ============================================================
    # Metrics
    # ============================================================

    metric_names = ["loss","classification_loss","transformer_moe_aux_loss","classifier_moe_aux_loss","accuracy","mcc","f1_macro","f1_weighted","top5_accuracy","top10_accuracy"]
    if contrastive_enabled:
        metric_names.append("contrastive_loss")
    metrics_dict = {"train": {name: [] for name in metric_names},"val": {name: [] for name in metric_names},"time_seconds": [],}
    start = time.time_ns()
    if num_classes is None:
        num_classes = len(glycans)
    output_dir = Path(save_dir)

    def prefix_wandb_payload(payload):
        if not wandb_prefix:
            return payload
        return {f"{wandb_prefix}/{key}": value for key, value in payload.items()}
    # ============================================================
    # Helper: unwrap DataParallel
    # ============================================================

    def get_base_model(current_model):
        if isinstance(current_model,torch.nn.DataParallel):
            return current_model.module
        return current_model

    base_model = get_base_model(model)
    mask_predictions = getattr(criterion, "mask_logits", None)
    uses_embedding_loss = getattr(criterion, "uses_embeddings", False)
    supports_aux_losses = model_type in {"CNN", "Transformer"}
    transformer_moe_enabled = (model_type == "Transformer" and getattr(base_model, "use_transformer_ff", True) and getattr(base_model, "encoder_type", "dense") == "moe")
    classifier_moe_enabled = supports_aux_losses and getattr(base_model, "classifier_moe", False)
    classifier_routing_enabled = log_to_wandb and classifier_moe_enabled
    transformer_routing_enabled = log_to_wandb and transformer_moe_enabled
    classifier_num_experts = (base_model.classifier_out.num_experts if classifier_routing_enabled else 0)
    classifier_routing_device = (next(base_model.parameters()).device if classifier_routing_enabled or transformer_routing_enabled else None)
    # ============================================================
    # Helper: reduce auxiliary losses from forward outputs
    # ============================================================
    def reduce_aux_loss(aux_loss, reference_tensor):
        if aux_loss is None:
            return torch.zeros((), device=reference_tensor.device, dtype=reference_tensor.dtype)
        if not torch.is_tensor(aux_loss):
            aux_loss = torch.tensor(aux_loss,device=reference_tensor.device,dtype=reference_tensor.dtype)
        else:
            aux_loss = aux_loss.to(device=reference_tensor.device,dtype=reference_tensor.dtype)
        if aux_loss.ndim > 0:
            aux_loss = aux_loss.mean()
        return aux_loss
    # ============================================================
    # Class distributions
    # ============================================================
    val_class_counts, val_total = (calculate_class_distribution(dataloaders["val"],num_classes))
    classes_in_val = np.sum(val_class_counts > 0)
    print(f"Validation set: {classes_in_val}/{num_classes} classes present ({val_total} total samples)")
    train_distribution = (calculate_class_distribution(dataloaders["train"],num_classes))
    print(f"Training set: {np.sum(train_distribution[0] > 0)}/{num_classes} classes present")
    # ============================================================
    # Log dataset statistics
    # ============================================================
    if log_to_wandb:
        train_class_counts, train_total = (calculate_class_distribution(dataloaders["train"],num_classes))
        wandb.log(prefix_wandb_payload({"dataset_stats/train_samples":train_total,"dataset_stats/val_samples":val_total,
                   "dataset_stats/train_classes_present":np.sum(train_class_counts > 0),
                   "dataset_stats/val_classes_present":classes_in_val,
                   "dataset_stats/total_classes":num_classes}))
        # --------------------------------------------------------
        # Top 20 most frequent classes
        # --------------------------------------------------------
        top_classes_idx = np.argsort(train_class_counts)[-20:][::-1]
        top_classes_data = []
        for idx in top_classes_idx:
            if train_class_counts[idx] > 0:
                class_name = (glycans[idx] if idx < len(glycans) else f"Class_{idx}")
                class_name = class_name[:40]
                top_classes_data.append([class_name, int(train_class_counts[idx])])
        if top_classes_data:
            class_table = wandb.Table(data=top_classes_data,columns=["Class","Train Count"])
            wandb.log(prefix_wandb_payload({"top_20_classes":class_table}))
    # ============================================================
    # Print MoE configuration
    # ============================================================
    if model_type == "Transformer":
        transformer_ff_enabled = getattr(base_model,"use_transformer_ff",True)
        print("Transformer feed-forward enabled:",transformer_ff_enabled)
    if transformer_moe_enabled:
        print("Transformer MoE auxiliary-loss weight:",transformer_moe_aux_loss_weight)
    if classifier_moe_enabled:
        print("Classifier MoE auxiliary-loss weight:",classifier_moe_aux_loss_weight)
    if uses_embedding_loss:
        print("Embedding loss enabled:", criterion.__class__.__name__)
    if contrastive_enabled:
        print("Fine-tuning contrastive loss enabled:", contrastive_criterion.__class__.__name__)
        print("Fine-tuning contrastive-loss weight:", contrastive_loss_weight)
    if memory_reporter is not None:
        memory_reporter("before first epoch")
    # ============================================================
    # Training loop
    # ============================================================
    for epoch in range(num_epochs):
        train_batch_sampler = getattr(dataloaders["train"], "batch_sampler", None)
        if hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(epoch)
        print("Epoch {}/{}".format(epoch,num_epochs - 1))
        print("-" * 10)
        for phase in ["train","val",]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            # ====================================================
            # Running metrics
            # ====================================================
            running_loss = []
            running_classification_loss = []
            running_transformer_aux = []
            running_classifier_aux = []
            running_contrastive_loss = []
            running_acc = []
            running_mcc = []
            running_f1_macro = []
            running_f1_weighted = []
            running_topk = []
            running_topk10 = []
            # Accumulate complete validation predictions.
            all_preds_epoch = []
            all_labels_epoch = []
            if classifier_routing_enabled:
                routing_class_sample_counts = torch.zeros(num_classes,dtype=torch.long,device=classifier_routing_device)
                routing_expert_selection_counts = torch.zeros(num_classes,classifier_num_experts,dtype=torch.long,device=classifier_routing_device)
                routing_expert_responsibility_sums = torch.zeros(num_classes,classifier_num_experts,dtype=torch.float32,device=classifier_routing_device)
            if transformer_routing_enabled:
                transformer_routing_stats = [
                    (
                        torch.zeros(num_classes, dtype=torch.long, device=classifier_routing_device),
                        torch.zeros(num_classes, layer.ff.num_experts, dtype=torch.long, device=classifier_routing_device),
                        torch.zeros(num_classes, layer.ff.num_experts, dtype=torch.float32, device=classifier_routing_device),
                    )
                    for layer in base_model.transformer.layers
                ]
            # ====================================================
            # Batches
            # ====================================================
            for data in dataloaders[phase]:
                inputs, y = prepare_batch(data,model_type)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    needs_embeddings = uses_embedding_loss or (contrastive_enabled and phase == "train")
                    # =================================================
                    # FIRST FORWARD PASS
                    # =================================================
                    enable_running_stats(model)
                    if classifier_routing_enabled or transformer_routing_enabled:
                        if needs_embeddings:
                            pred,embeddings,classifier_routing,transformer_aux_raw,classifier_aux_raw = model(*inputs,rep=True,return_routing=True,return_aux_losses=True,return_selected_logits=False)
                        else:
                            pred,classifier_routing,transformer_aux_raw,classifier_aux_raw = model(*inputs,return_routing=True,return_aux_losses=True,return_selected_logits=False)
                        if classifier_routing_enabled:
                            _accumulate_classifier_routing(routing_class_sample_counts,routing_expert_selection_counts,routing_expert_responsibility_sums,y,classifier_routing)
                        if transformer_routing_enabled:
                            for layer_stats, layer_routing in zip(transformer_routing_stats, classifier_routing["transformer_layers"]):
                                _accumulate_transformer_routing(*layer_stats, y, layer_routing, inputs[1])
                        del classifier_routing
                    elif supports_aux_losses:
                        if needs_embeddings:
                            pred,embeddings,transformer_aux_raw,classifier_aux_raw= model(*inputs,rep=True,return_aux_losses=True)
                        else:
                            pred,transformer_aux_raw,classifier_aux_raw= model(*inputs,return_aux_losses=True)
                    else:
                        if needs_embeddings:
                            pred,embeddings = model(*inputs,rep=True)
                        else:
                            pred = model(*inputs)
                    classification_loss = criterion(embeddings if uses_embedding_loss else pred,y)
                    contrastive_loss = reduce_aux_loss(
                        contrastive_criterion(embeddings, y) if contrastive_enabled and phase == "train" else None,
                        classification_loss,
                    )
                    # -------------------------------------------------
                    # Reduce auxiliary losses for enabled components
                    # -------------------------------------------------
                    transformer_aux = reduce_aux_loss(transformer_aux_raw if transformer_moe_enabled else None, classification_loss)
                    classifier_aux = reduce_aux_loss(classifier_aux_raw if classifier_moe_enabled else None, classification_loss)
                    # -------------------------------------------------
                    # Full training objective
                    # -------------------------------------------------
                    total_loss = classification_loss
                    if transformer_moe_enabled:
                        total_loss = total_loss + transformer_moe_aux_loss_weight * transformer_aux
                    if classifier_moe_enabled:
                        total_loss = total_loss + classifier_moe_aux_loss_weight * classifier_aux
                    if contrastive_enabled and phase == "train":
                        total_loss = total_loss + contrastive_loss_weight * contrastive_loss
                    if phase == "train":
                        loss = total_loss
                    else:
                        loss = classification_loss
                    # =================================================
                    # SAM FIRST STEP
                    # =================================================
                    if phase == "train":
                        total_loss.backward()
                        optimizer.first_step(zero_grad=True)
                        # =================================================
                        # SAM SECOND FORWARD PASS
                        # =================================================
                        disable_running_stats(model)
                        if supports_aux_losses:
                            if uses_embedding_loss or contrastive_enabled:
                                pred_second,embeddings_second,transformer_aux_second_raw,classifier_aux_second_raw= model(*inputs,rep=True,return_aux_losses=True)
                            else:
                                pred_second,transformer_aux_second_raw,classifier_aux_second_raw= model(*inputs,return_aux_losses=True)
                        else:
                            if uses_embedding_loss or contrastive_enabled:
                                pred_second,embeddings_second = model(*inputs,rep=True)
                            else:
                                pred_second = model(*inputs)
                        classification_loss_second = (criterion(embeddings_second if uses_embedding_loss else pred_second,y))
                        contrastive_loss_second = reduce_aux_loss(
                            contrastive_criterion(embeddings_second, y) if contrastive_enabled else None,
                            classification_loss_second,
                        )
                        # ---------------------------------------------
                        # IMPORTANT:
                        #
                        # The second forward pass changes the router
                        # outputs, so the auxiliary losses must come
                        # from that same forward output.
                        # ---------------------------------------------
                        transformer_aux_second = reduce_aux_loss(transformer_aux_second_raw if transformer_moe_enabled else None, classification_loss_second)
                        classifier_aux_second = reduce_aux_loss(classifier_aux_second_raw if classifier_moe_enabled else None, classification_loss_second)
                        # ---------------------------------------------
                        # Second SAM objective
                        # ---------------------------------------------
                        loss_second = classification_loss_second
                        if transformer_moe_enabled:
                            loss_second = loss_second + transformer_moe_aux_loss_weight * transformer_aux_second
                        if classifier_moe_enabled:
                            loss_second = loss_second + classifier_moe_aux_loss_weight * classifier_aux_second
                        if contrastive_enabled:
                            loss_second = loss_second + contrastive_loss_weight * contrastive_loss_second
                        loss_second.backward()
                        optimizer.second_step(zero_grad=True)
                if mask_predictions is not None:
                    pred = mask_predictions(pred.detach(), y)
                # ====================================================
                # Validation prediction accumulation
                # ====================================================
                if phase == "val":
                    all_preds_epoch.append(pred.detach().argmax(dim=1).cpu())
                    all_labels_epoch.append(y.detach().cpu())
                # ====================================================
                # Batch-level losses
                # ====================================================
                running_loss.append(loss.detach().item())
                running_classification_loss.append(classification_loss.detach().item())
                running_transformer_aux.append(transformer_aux.detach().item())
                running_classifier_aux.append(classifier_aux.detach().item())
                if contrastive_enabled:
                    running_contrastive_loss.append(contrastive_loss.detach().item())
                # ====================================================
                # Batch metrics
                # ====================================================
                running_acc.append(accuracy(pred,y,task="multiclass",num_classes=num_classes).detach().item())
                running_topk.append(accuracy(pred, y, task="multiclass", num_classes=num_classes, top_k=5).detach().item())
                running_topk10.append(accuracy(pred, y, task="multiclass", num_classes=num_classes, top_k=10).detach().item())
                sklearn_metrics = (calculate_metrics_sklearn(pred, y, num_classes, batch_size_small=True))
                running_mcc.append(sklearn_metrics["mcc"])
                running_f1_macro.append(sklearn_metrics["f1_macro"])
                running_f1_weighted.append(sklearn_metrics["f1_weighted"])
            # ========================================================
            # Epoch averages
            # ========================================================
            epoch_loss = np.mean(running_loss)
            epoch_classification_loss = (np.mean(running_classification_loss))
            epoch_transformer_aux = (np.mean(running_transformer_aux) if running_transformer_aux else 0.0)
            epoch_classifier_aux = (np.mean(running_classifier_aux) if running_classifier_aux else 0.0)
            epoch_contrastive_loss = (np.mean(running_contrastive_loss) if running_contrastive_loss else 0.0)
            epoch_acc = float(np.mean(running_acc))
            epoch_topk = float(np.mean(running_topk))
            epoch_topk10 = float(np.mean(running_topk10))
            epoch_mcc = np.mean(running_mcc)
            epoch_f1_macro = np.mean(running_f1_macro)
            epoch_f1_weighted = np.mean(running_f1_weighted)
            # ========================================================
            # Full-validation metrics
            # ========================================================
            if (phase == "val" and len(all_preds_epoch) > 0):
                all_preds_cat = torch.cat(all_preds_epoch, dim=0)
                all_labels_cat = torch.cat(all_labels_epoch, dim=0)
                final_metrics = (calculate_metrics_sklearn_from_classes(all_preds_cat, all_labels_cat, num_classes, batch_size_small=False, present_labels_only=True))
                epoch_f1_macro_final = (final_metrics["f1_macro"])
                epoch_f1_weighted_final = (final_metrics["f1_weighted"])
                epoch_mcc_final = (final_metrics["mcc"])
                del all_preds_cat, all_labels_cat
            else:
                epoch_f1_macro_final = (epoch_f1_macro)
                epoch_f1_weighted_final = (epoch_f1_weighted)
                epoch_mcc_final = epoch_mcc
            # ========================================================
            # Console/logging
            # ========================================================
            message = ("{} " "Loss: {:.4f} " "Cls: {:.4f} " "T-MoE: {:.4f} " "C-MoE: {:.4f} " "Acc: {:.4f} " "MCC: {:.4f} " "F1-macro: {:.4f} " "F1-weighted: {:.4f} "
                "Top-5: {:.4f} " "Top-10: {:.4f}").format(phase, epoch_loss, epoch_classification_loss, epoch_transformer_aux, epoch_classifier_aux, epoch_acc, epoch_mcc_final,
                epoch_f1_macro_final, epoch_f1_weighted_final, epoch_topk, epoch_topk10)
            if contrastive_enabled:
                message += f" SupCon: {epoch_contrastive_loss:.4f}"
            logging.info(message)
            print(message)
            # ========================================================
            # WandB
            # ========================================================
            if log_to_wandb:
                wandb_payload = {f"{phase}/loss": epoch_loss, f"{phase}/classification_loss": epoch_classification_loss,
                    f"{phase}/transformer_moe_aux_loss": epoch_transformer_aux, f"{phase}/classifier_moe_aux_loss": epoch_classifier_aux,
                    f"{phase}/accuracy": epoch_acc, f"{phase}/mcc": epoch_mcc_final, f"{phase}/f1_score_macro": epoch_f1_macro_final,
                    f"{phase}/f1_score_weighted": epoch_f1_weighted_final, f"{phase}/top5_accuracy": epoch_topk, f"{phase}/top10_accuracy": epoch_topk10, "epoch": epoch,}
                if contrastive_enabled:
                    wandb_payload[f"{phase}/contrastive_loss"] = float(epoch_contrastive_loss)
                if classifier_routing_enabled:
                    wandb_payload.update(_classifier_routing_wandb_payload(phase, epoch, glycans, routing_class_sample_counts, routing_expert_selection_counts,
                        routing_expert_responsibility_sums,glycan_compositions=glycan_compositions))
                if transformer_routing_enabled:
                    for layer_idx, layer_stats in enumerate(transformer_routing_stats, start=1):
                        wandb_payload.update(_classifier_routing_wandb_payload(
                            phase, epoch, glycans, *layer_stats,
                            routing_name=f"transformer_routing/layer_{layer_idx}",
                            routing_title=f"Transformer MoE layer {layer_idx}",
                            count_label="Tokens",
                            glycan_compositions=glycan_compositions,
                        ))
                wandb.log(prefix_wandb_payload(wandb_payload))
            # ========================================================
            # Save metrics
            # ========================================================
            metrics_dict[phase]["loss"].append(float(epoch_loss))
            metrics_dict[phase]["classification_loss"].append(float(epoch_classification_loss))
            metrics_dict[phase]["transformer_moe_aux_loss"].append(float(epoch_transformer_aux))
            metrics_dict[phase]["classifier_moe_aux_loss"].append(float(epoch_classifier_aux))
            if contrastive_enabled:
                metrics_dict[phase]["contrastive_loss"].append(float(epoch_contrastive_loss))
            metrics_dict[phase]["accuracy"].append(float(epoch_acc.item() if hasattr(epoch_acc, "item") else epoch_acc))
            metrics_dict[phase]["mcc"].append(float(epoch_mcc_final))
            metrics_dict[phase]["f1_macro"].append(float(epoch_f1_macro_final))
            metrics_dict[phase]["f1_weighted"].append(float(epoch_f1_weighted_final))
            metrics_dict[phase]["top5_accuracy"].append(float(epoch_topk.item() if hasattr(epoch_topk, "item") else epoch_topk))
            metrics_dict[phase]["top10_accuracy"].append(float(epoch_topk10.item() if hasattr(epoch_topk10, "item") else epoch_topk10))
            # ========================================================
            # Best model / early stopping
            # ========================================================
            if (phase == "val" and epoch_loss <= best_loss):
                best_loss = epoch_loss
                best_model_wts = (copy.deepcopy(model.state_dict()))
            if (phase == "val" and epoch_acc > best_acc):
                best_acc = epoch_acc
            if phase == "val":
                val_losses.append(epoch_loss)
                val_acc.append(epoch_acc)
                early_stopping(epoch_loss, model)
                scheduler.step(epoch_loss)
            if phase == "train":
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc)
            torch.cuda.empty_cache()
            if memory_reporter is not None:
                memory_reporter(f"epoch {epoch + 1} {phase}")
        # ============================================================
        # Early stopping
        # ============================================================
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print()
        print(f"Time since start: " f"{(time.time_ns() - start) / 1e9:.2f} " f"seconds")
        metrics_dict["time_seconds"].append(float((time.time_ns() - start) / 1e9))
    # ================================================================
    # Training finished
    # ================================================================
    time_elapsed = (time.time() - since)
    print("Training complete in " "{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val loss: {:4f}, " "best Accuracy score: {:.4f}".format(best_loss, best_acc))
    # ================================================================
    # Save metrics
    # ================================================================
    os.makedirs(output_dir / "metrics", exist_ok=True)
    metrics_path = output_dir / "metrics" / f"CandyCrunch_metrics_{setting_name}.json"
    plot_path = output_dir / "metrics" / f"CandyCrunch_metric_{setting_name}.png"
    metrics_dict["best"] = {"val_loss": float(best_loss), "val_accuracy": float(best_acc.item() if hasattr(best_acc, "item") else best_acc),}
    metrics_dict["completed_epochs"] = len(metrics_dict["val"]["loss"])
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    # ================================================================
    # Plot loss / accuracy
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(range(len(val_losses)), val_losses, label="Validation")
    ax1.plot(range(len(train_losses)), train_losses, label="Training")
    ax1.set_ylabel("Loss")
    ax1.set_title("Model Training - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(range(len(val_acc)), val_acc, label="Validation")
    ax2.plot(range(len(train_acc)), train_acc, label="Training")
    ax2.set_xlabel("Number of Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Model Training - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    # ================================================================
    # Save best model
    # ================================================================
    best_model_path = output_dir / f"CandyCrunch_{setting_name}.pt"
    checkpoint = dict(checkpoint_metadata or {})
    checkpoint["state_dict"] = best_model_wts
    checkpoint["best_val_loss"] = float(best_loss)
    checkpoint["best_val_accuracy"] = (float(best_acc.item()) if hasattr(best_acc, "item") else float(best_acc))
    # Save auxiliary-loss weights explicitly too.
    checkpoint["transformer_moe_aux_loss_weight"] = float(transformer_moe_aux_loss_weight)
    checkpoint["classifier_moe_aux_loss_weight"] = float(classifier_moe_aux_loss_weight)
    if model_type == "Transformer":
        checkpoint["use_transformer_ff"] = bool(getattr(get_base_model(model), "use_transformer_ff", True))
    torch.save(checkpoint, best_model_path)
    # ================================================================
    # Final WandB logging
    # ================================================================
    if log_to_wandb:
        wandb.log(prefix_wandb_payload({"training_plots/loss_curves": wandb.Image(str(plot_path)), "best_metrics/best_val_loss": best_loss,
            "best_metrics/best_val_accuracy": (best_acc.item() if hasattr(best_acc, "item") else best_acc),}))
        wandb.save(str(best_model_path))
        wandb.save(str(metrics_path))
    # ================================================================
    # Restore best weights
    # ================================================================
    model.load_state_dict(best_model_wts)
    return model
