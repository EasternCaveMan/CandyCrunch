import copy
import json
import numpy as np
import time
import torch
import wandb
import logging
import warnings
import matplotlib.pyplot as plt
import os

from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.exceptions import UndefinedMetricWarning
from candycrunch.analysis import glycan_to_graph_monos, mono_graph_to_nx, enumerate_k_graphs, mono_frag_to_string
from glycowork.ml.model_training import EarlyStopping, disable_running_stats, enable_running_stats, training_setup, \
    Poly1CrossEntropyLoss
from glycowork.ml.models import init_weights

from torchmetrics.functional import accuracy

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category = UndefinedMetricWarning)
warnings.filterwarnings("ignore", category = UserWarning, module = "sklearn")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


def calculate_class_distribution(dataloader, num_classes):
    """Calculate class distribution in a dataloader"""
    class_counts = np.zeros(num_classes)
    total_samples = 0

    for data in dataloader:
        y = data[-1].squeeze()
        for label in y.cpu().numpy():
            if label < num_classes:
                class_counts[int(label)] += 1
                total_samples += 1

    return class_counts, total_samples


# def prepare_batch(data, model_type):
#     (
#         mz_list,
#         peak_list,
#         mz_remainder,
#         precursor,
#         glycan_type,
#         rt,
#         mode_in,
#         lc,
#         modification,
#         trap,
#         y,
#     ) = data
#
#     precursor = precursor.to(device)
#     glycan_type = glycan_type.to(device)
#     rt = rt.to(device)
#     mode_in = mode_in.to(device)
#     lc = lc.to(device)
#     modification = modification.to(device)
#     trap = trap.to(device)
#     y = y.squeeze().to(device)
#
#     if model_type == "CNN":
#         mz_features = torch.stack([mz_list, mz_remainder], dim=1).to(device)
#
#         inputs = [
#             mz_features,
#             precursor,
#             glycan_type,
#             rt,
#             mode_in,
#             lc,
#             modification,
#             trap,
#         ]
#
#     elif model_type == "Transformer":
#         peak_list = peak_list.to(device)
#
#         # Padding rows are expected to be [0.0, 0.0].
#         # True means "ignore this peak" for Transformer masks.
#         # For case [0.0, 0.0]
#         peak_padding_mask = peak_list.abs().sum(dim=-1) == 0
#         # For case [0.0]
#         # peak_padding_mask = peak_list.squeeze(-1) < 0
#
#         inputs = [
#             peak_list,
#             peak_padding_mask,
#             precursor,
#             glycan_type,
#             rt,
#             mode_in,
#             lc,
#             modification,
#             trap,
#         ]
#
#     else:
#         raise ValueError(
#             f"Unknown model_type={model_type!r}. Expected 'cnn' or 'transformer'."
#         )
#
#     return inputs, y


def prepare_batch(data, model_type):

    if model_type == "CNN":
        (
            mz_list,
            peak_list,
            mz_remainder,
            precursor,
            glycan_type,
            rt,
            mode_in,
            lc,
            modification,
            trap,
            y,
        ) = data

        mz_features = torch.stack([mz_list, mz_remainder], dim=1).to(device)

        inputs = [
            mz_features,
            precursor.to(device),
            glycan_type.to(device),
            rt.to(device),
            mode_in.to(device),
            lc.to(device),
            modification.to(device),
            trap.to(device),
        ]

        y = y.squeeze().to(device)

    elif model_type == "Transformer":
        (
            peak_list,
            peak_padding_mask,
            precursor,
            glycan_type,
            rt,
            mode_in,
            lc,
            modification,
            trap,
            y,
        ) = data

        inputs = [
            peak_list.to(device),
            peak_padding_mask.to(device),
            precursor.to(device),
            glycan_type.to(device),
            rt.to(device),
            mode_in.to(device),
            lc.to(device),
            modification.to(device),
            trap.to(device),
        ]

        y = y.squeeze().to(device)

    else:
        raise ValueError(
            f"Unknown model_type={model_type!r}. Expected 'CNN' or 'Transformer'."
        )

    return inputs, y


def calculate_metrics_sklearn(pred, y, num_classes, batch_size_small=True, present_labels_only=False):
    pred_classes = torch.argmax(pred, dim=1)

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


class custom_loss(torch.nn.Module):
    def __init__(self, primary_loss, dist_sim, dist_comp, logit_norm = False, t = 1.0):
        super(custom_loss, self).__init__()
        self.primary_loss = primary_loss
        self.dist_sim = dist_sim
        self.dist_comp = dist_comp
        self.logit_norm = logit_norm
        self.t = t

    def forward(self, output, target):
        if self.logit_norm:
            norms = torch.norm(output, p = 2, dim = -1, keepdim = True) + 1e-7
            output = torch.div(output, norms) / self.t
        loss2 = self.primary_loss(output, target)
        output = torch.nn.functional.softmax(output, dim = 1)
        target_sim = self.dist_sim[target]
        loss_sim = output * target_sim
        target_comp = self.dist_comp[target]
        loss_comp = output * target_comp
        loss = loss_comp.mean() + loss_sim.mean() + loss2
        return loss


def train_model(model, dataloaders, criterion, optimizer,
                scheduler, glycans, num_epochs = None, patience = None, log_to_wandb = True, num_classes = None,
                model_type = None, setting_name = None, moe_aux_loss_weight=None,checkpoint_metadata=None):
    """trains a deep learning model on predicting glycan properties

    Arguments:
    :-
    model (PyTorch object): graph neural network (such as SweetNet) for analyzing glycans
    dataloaders (PyTorch object): dictionary of dataloader objects with keys 'train' and 'val'
    criterion (PyTorch object): PyTorch loss function
    optimizer (PyTorch object): PyTorch optimizer
    scheduler (PyTorch object): PyTorch learning rate decay
    num_epochs (int): number of epochs for training; default:25
    patience (int): number of epochs without improvement until early stop; default:50
    log_to_wandb (bool): whether to log metrics to wandb; default:True
    num_classes (int): number of classes; default:None (uses len(glycans))

    Returns:
    :-
    Returns the best model seen during training
    """
    since = time.time()
    early_stopping = EarlyStopping(patience = patience, verbose = True)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.0
    best_acc = 0.0
    val_losses = []
    val_acc = []
    train_losses = []
    train_acc = []

    metric_names = [
        "loss",
        "accuracy",
        "mcc",
        "f1_macro",
        "f1_weighted",
        "top5_accuracy",
        "top10_accuracy",
    ]

    metrics_dict = {
        "train": {name: [] for name in metric_names},
        "val": {name: [] for name in metric_names},
        "time_seconds": [],
    }

    start = time.time_ns()

    if num_classes is None:
        num_classes = len(glycans)

    # Check class distribution in validation set
    val_class_counts, val_total = calculate_class_distribution(dataloaders['val'], num_classes)
    classes_in_val = np.sum(val_class_counts > 0)

    print(f"Validation set: {classes_in_val}/{num_classes} classes present ({val_total} total samples)")
    print(
        f"Training set: {np.sum(calculate_class_distribution(dataloaders['train'], num_classes)[0] > 0)}/{num_classes} classes present")

    # Log class distribution summary to wandb
    if log_to_wandb:
        train_class_counts, train_total = calculate_class_distribution(dataloaders['train'], num_classes)

        wandb.log({
            "dataset_stats/train_samples": train_total,
            "dataset_stats/val_samples": val_total,
            "dataset_stats/train_classes_present": np.sum(train_class_counts > 0),
            "dataset_stats/val_classes_present": classes_in_val,
            "dataset_stats/total_classes": num_classes,
        })

        # Log top 20 most frequent classes in training
        top_classes_idx = np.argsort(train_class_counts)[-20:][::-1]
        top_classes_data = []
        for idx in top_classes_idx:
            if train_class_counts[idx] > 0:
                class_name = glycans[idx] if idx < len(glycans) else f'Class_{idx}'
                class_name = class_name[:40]  # Truncate
                top_classes_data.append([class_name, int(train_class_counts[idx])])

        if top_classes_data:
            class_table = wandb.Table(data = top_classes_data, columns = ["Class", "Train Count"])
            wandb.log({"top_20_classes": class_table})

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = []
            running_acc = []
            running_mcc = []
            running_f1_macro = []
            running_f1_weighted = []
            running_topk = []
            running_topk10 = []

            # For accumulating predictions for end-of-epoch metrics
            all_preds_epoch = []
            all_labels_epoch = []

            for data in dataloaders[phase]:
                inputs, y = prepare_batch(data, model_type)
                optimizer.zero_grad(set_to_none = True)

                with torch.set_grad_enabled(phase == 'train'):
                    # first forward pass
                    enable_running_stats(model)

                    if moe_aux_loss_weight == None:
                        pred = model(*inputs)
                        loss = criterion(pred, y)
                        if phase == 'train':
                            loss.backward()
                            optimizer.first_step(zero_grad = True)
                            # second forward pass
                            disable_running_stats(model)
                            criterion(model(*inputs),y).backward()
                            optimizer.second_step(zero_grad = True)

                    elif moe_aux_loss_weight != None:
                        pred = model(*inputs)
                        loss = criterion(pred, y)

                        model_for_aux = model.module if hasattr(model, "module") else model

                        if phase == "train" and hasattr(model_for_aux, "get_aux_loss"):
                            aux_loss = model_for_aux.get_aux_loss()
                            if aux_loss is not None:
                                loss = loss + moe_aux_loss_weight * aux_loss

                        if phase == 'train':
                            loss.backward()
                            optimizer.first_step(zero_grad = True)

                            # second forward pass for SAM
                            disable_running_stats(model)

                            pred_second = model(*inputs)
                            loss_second = criterion(pred_second, y)

                            if hasattr(model_for_aux, "get_aux_loss"):
                                aux_loss_second = model_for_aux.get_aux_loss()
                                if aux_loss_second is not None:
                                    loss_second = loss_second + moe_aux_loss_weight * aux_loss_second

                            loss_second.backward()
                            optimizer.second_step(zero_grad = True)

                # Collect predictions for end-of-epoch metrics (for validation)
                if phase == 'val':
                    all_preds_epoch.append(pred.detach().cpu())
                    all_labels_epoch.append(y.detach().cpu())

                # Collect batch metrics
                running_loss.append(loss.item())
                running_acc.append(accuracy(pred, y, task = "multiclass", num_classes = num_classes))
                running_topk.append(accuracy(pred, y, task = "multiclass", num_classes = num_classes, top_k = 5))
                running_topk10.append(accuracy(pred, y, task = "multiclass", num_classes = num_classes, top_k = 10))

                # Calculate sklearn metrics for this batch
                sklearn_metrics = calculate_metrics_sklearn(pred, y, num_classes, batch_size_small = True)
                running_mcc.append(sklearn_metrics['mcc'])
                running_f1_macro.append(sklearn_metrics['f1_macro'])
                running_f1_weighted.append(sklearn_metrics['f1_weighted'])

            # Average metrics at end of epoch
            epoch_loss = np.mean(running_loss)
            epoch_acc = torch.mean(torch.stack(running_acc))
            epoch_topk = torch.mean(torch.stack(running_topk))
            epoch_topk10 = torch.mean(torch.stack(running_topk10))
            epoch_mcc = np.mean(running_mcc)
            epoch_f1_macro = np.mean(running_f1_macro)
            epoch_f1_weighted = np.mean(running_f1_weighted)

            # For validation, compute more accurate metrics on all predictions
            if phase == 'val' and len(all_preds_epoch) > 0:
                all_preds_cat = torch.cat(all_preds_epoch, dim = 0)
                all_labels_cat = torch.cat(all_labels_epoch, dim = 0)
                final_metrics = calculate_metrics_sklearn(
                    all_preds_cat,
                    all_labels_cat,
                    num_classes,
                    batch_size_small = False,
                    present_labels_only = True
                )
                epoch_f1_macro_final = final_metrics['f1_macro']
                epoch_f1_weighted_final = final_metrics['f1_weighted']
                epoch_mcc_final = final_metrics['mcc']
            else:
                epoch_f1_macro_final = epoch_f1_macro
                epoch_f1_weighted_final = epoch_f1_weighted
                epoch_mcc_final = epoch_mcc

            logging.info(
                '{} Loss: {:.4f} Acc: {:.4f} MCC: {:.4f} F1-macro: {:.4f} F1-weighted: {:.4f} Top-5: {:.4f} Top-10: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_mcc_final, epoch_f1_macro_final, epoch_f1_weighted_final,
                    epoch_topk, epoch_topk10))
            print(
                '{} Loss: {:.4f} Acc: {:.4f} MCC: {:.4f} F1-macro: {:.4f} F1-weighted: {:.4f} Top-5: {:.4f} Top-10: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_mcc_final, epoch_f1_macro_final, epoch_f1_weighted_final,
                    epoch_topk, epoch_topk10))

            # Log to wandb
            if log_to_wandb:
                wandb.log({
                    f'{phase}/loss': epoch_loss,
                    f'{phase}/accuracy': epoch_acc,
                    f'{phase}/mcc': epoch_mcc_final,
                    f'{phase}/f1_score_macro': epoch_f1_macro_final,
                    f'{phase}/f1_score_weighted': epoch_f1_weighted_final,
                    f'{phase}/top5_accuracy': epoch_topk,
                    f'{phase}/top10_accuracy': epoch_topk10,
                    'epoch': epoch
                })

            metrics_dict[phase]["loss"].append(float(epoch_loss))
            metrics_dict[phase]["accuracy"].append(float(epoch_acc.item() if hasattr(epoch_acc, "item") else epoch_acc))
            metrics_dict[phase]["mcc"].append(float(epoch_mcc_final))
            metrics_dict[phase]["f1_macro"].append(float(epoch_f1_macro_final))
            metrics_dict[phase]["f1_weighted"].append(float(epoch_f1_weighted_final))
            metrics_dict[phase]["top5_accuracy"].append(
                float(epoch_topk.item() if hasattr(epoch_topk, "item") else epoch_topk))
            metrics_dict[phase]["top10_accuracy"].append(
                float(epoch_topk10.item() if hasattr(epoch_topk10, "item") else epoch_topk10))

            # keep best model state_dict
            if phase == 'val' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
            if phase == 'val':
                val_losses.append(epoch_loss)
                val_acc.append(epoch_acc.item())
                # check Early Stopping & adjust learning rate if needed
                early_stopping(epoch_loss, model)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc.item())

            torch.cuda.empty_cache()

        if early_stopping.early_stop:
            print("Early stopping")
            break

        print()
        print(f"Time since start: {(time.time_ns() - start) / 1e9:.2f} seconds")
        metrics_dict["time_seconds"].append(float((time.time_ns() - start) / 1e9))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}, best Accuracy score: {:.4f}'.format(best_loss, best_acc))

    os.makedirs("./models", exist_ok = True)

    metrics_path = f'./models/CandyCrunch_metrics_{setting_name}.json'
    plot_path = f'./models/CandyCrunch_metric_{setting_name}.png'
    metrics_dict["best"] = {
        "val_loss": float(best_loss),
        "val_accuracy": float(best_acc.item() if hasattr(best_acc, "item") else best_acc),
    }

    metrics_dict["completed_epochs"] = len(metrics_dict["val"]["loss"])

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent = 2)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}, best Accuracy score: {:.4f}'.format(best_loss, best_acc))

    # Plot loss & score over the course of training
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 8))

    ax1.plot(range(len(val_losses)), val_losses, label = 'Validation')
    ax1.plot(range(len(train_losses)), train_losses, label = 'Training')
    ax1.set_ylabel('Loss')
    ax1.set_title('Model Training - Loss')
    ax1.legend()
    ax1.grid(True, alpha = 0.3)

    ax2.plot(range(len(val_acc)), val_acc, label = 'Validation')
    ax2.plot(range(len(train_acc)), train_acc, label = 'Training')
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Training - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

    # Save best model weights
    best_model_path = f'./models/CandyCrunch_{setting_name}.pt'

    checkpoint = dict(checkpoint_metadata or {})
    checkpoint["state_dict"] = best_model_wts
    checkpoint["best_val_loss"] = float(best_loss)
    checkpoint["best_val_accuracy"] = (float(best_acc.item()) if hasattr(best_acc, "item") else float(best_acc))
    torch.save(checkpoint, best_model_path)

    # Log final plots/model to wandb
    if log_to_wandb:
        wandb.log({
            'training_plots/loss_curves': wandb.Image(plot_path),
            'best_metrics/best_val_loss': best_loss,
            'best_metrics/best_val_accuracy': best_acc.item() if hasattr(best_acc, 'item') else best_acc,
        })
        wandb.save(best_model_path)
        wandb.save(metrics_path)

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)

    return model