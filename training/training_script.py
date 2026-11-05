import pickle
import warnings
import argparse
import random
import hashlib
import re
from pathlib import Path
import numpy as np
import torch
from candycrunch.losses import (
    CandidateSetLoss,
    ClassAwareContrastiveBatchSampler,
    FocalLoss,
    SupConLoss,
    UnlabeledContrastiveBatchSampler,
    UnsupervisedContrastiveLoss,
    custom_loss,
    xyz_loss,
)
from glycowork.ml.models import init_weights
from glycowork.ml.model_training import training_setup
from glycowork.motif.graph import compare_glycans
import pandas as pd
import wandb
from candycrunch.model import CandyCrunch_CNN,CandyCrunch_CNN2, SimpleDataset, transform_mz, transform_rt
from candycrunch.transformer import CandyCrunch_Transformer
from glycowork.motif.annotate import annotate_dataset, get_k_saccharides
from glycowork.motif.tokenization import glycan_to_composition
from sklearn.metrics import pairwise_distances
if __package__:
    from .training_utils import RawPretrainingInputs, ResampledDataset, load_dataset_inputs, seed_worker, train_model
else:
    from training_utils import RawPretrainingInputs, ResampledDataset, load_dataset_inputs, seed_worker, train_model

warnings.filterwarnings("ignore",category=RuntimeWarning,module="sklearn.utils.extmath")


CLASSIFICATION_LOSS_CHOICES = ["custom_loss", "xyz_loss", "cross_entropy", "PolyCrEnr", "focal_loss"]
LOSS_TAGS = {
    "xyz_loss": "_XYZLOSS",
    "cross_entropy": "_CELOSS",
    "custom_loss": "_CMLOSS",
    "PolyCrEnr": "_POLOSS",
    "focal_loss": "_FOLLOSS",
}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value such as True or False.")


def set_seed(seed=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_transformer_peak_hidden_dim(args):
    if args.peak_hidden_dim is not None:
        return args.peak_hidden_dim
    if args.ff_dim is not None:
        return args.ff_dim // 2
    if args.nheads is None:
        return None
    return args.nheads * 64

def resolve_prepared_dataset_dir(dataset):
    dataset_path = Path(dataset)
    if dataset_path.exists():
        return dataset_path
    base_dirs = [Path("."), Path(__file__).resolve().parent]
    for base_dir in base_dirs:
        exact = base_dir / f"prepared_datasets_{dataset}"
        if exact.exists():
            return exact
    matches = []
    for base_dir in base_dirs:
        matches.extend(base_dir.glob(f"prepared_datasets_{dataset}*"))
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find prepared dataset {dataset!r}.")
    names = ", ".join(str(match) for match in matches)
    raise ValueError(f"Prepared dataset {dataset!r} is ambiguous: {names}")



def main(args):
    set_seed(args.current_seed)
    if args.model not in {"CNN", "CNN2", "Transformer"}:
        raise ValueError(f"Unknown model={args.model!r}. Expected 'CNN', 'CNN2', or 'Transformer'.")
    if args.model == "CNN" and args.classifier_moe:
        raise ValueError("--classifier_moe requires --model CNN2 or Transformer.")
    pretraining_enabled = args.sup_pretraining or args.unsup_pretraining
    if args.sup_pretraining and args.unsup_pretraining:
        raise ValueError("Use either --sup-pretraining True or --unsup-pretraining True, not both.")
    if pretraining_enabled and args.pretraining_epochs < 1:
        raise ValueError("--pretraining-epochs must be at least 1 when pretraining is enabled.")
    if args.unsup_pretraining and args.unsup_pretraining_examples_per_batch < 1:
        raise ValueError("--unsup-pretraining-examples-per-batch must be positive.")
    if args.unsup_pretraining and args.unsup_pretraining_views_per_example < 2:
        raise ValueError("--unsup-pretraining-views-per-example must be at least 2.")
    if args.combine_loss and (not np.isfinite(args.contrastive_loss_weight) or args.contrastive_loss_weight <= 0):
        raise ValueError("--contrastive-loss-weight must be finite and positive when --combine-loss is True.")
    if args.model == "Transformer" and args.disable_mha_fastpath:
        torch.backends.mha.set_fastpath_enabled(False)
        print("Transformer attention fastpath disabled for CUDA diagnostics.")
    print("Reading data")
    # ============================================================
    # Load data
    # ============================================================
    dataset_dir = resolve_prepared_dataset_dir(args.dataset)
    input_dim = 2048
    train_source = dataset_dir / f"X_train_{args.split}.pkl"
    val_source = dataset_dir / f"X_test_{args.split}.pkl"
    if not train_source.is_file():
        train_source = dataset_dir / f"train_{args.split}"
    if not val_source.is_file():
        val_source = dataset_dir / f"test_{args.split}"
    with open(dataset_dir / f"y_train_{args.split}.pkl","rb",) as file:
        y_train = pickle.load(file)
    with open(dataset_dir / f"y_test_{args.split}.pkl","rb",) as file:
        y_test = pickle.load(file)
    with open(dataset_dir/"glycans.pkl", "rb") as file:
        glycans = pickle.load(file)
    glycan_comp_by_glycan = dict(zip(glycans["glycan"], glycans["glycan_comp"]))
    glycans = sorted(set(glycans["glycan"]))
    glycan_compositions = [glycan_comp_by_glycan[glycan] for glycan in glycans]
    # ============================================================
    # Preprocessing
    # ============================================================
    print("Preprocessing data")
    disallowed_glycans = []
    allowed_glycan_comps = {}
    for glyc in glycans:
        try:
            glycomp = glycan_to_composition(glyc)
            allowed_glycan_comps[glyc] = glycomp
        except KeyError:
            disallowed_glycans.append(glyc)
    if disallowed_glycans:
        raise ValueError(
            f"Cannot encode compositions for {len(disallowed_glycans)} glycans, "
            f"including {disallowed_glycans[:5]!r}. Correct the prepared dataset before training."
        )
    comp_vector_order = list(set(x for y in allowed_glycan_comps.values() for x in y))
    comp_vector_order = sorted(comp_vector_order,key=lambda x: x.lower())
    print(f"Comp_vector_order: {comp_vector_order}")
    glycan_comp_vect_map = {}
    for glyc, glycomp in allowed_glycan_comps.items():
        comp_vect = np.zeros(len(comp_vector_order))
        for mono, counts in glycomp.items():
            comp_vect[comp_vector_order.index(mono)] = counts
        glycan_comp_vect_map[glyc] = comp_vect
    composition_vectors = np.stack(
        [glycan_comp_vect_map[glycan] for glycan in glycans]
    ).astype(np.float32, copy=False)
    glycan_indices = {glycan: index for index, glycan in enumerate(glycans)}
    y_train = np.fromiter((glycan_indices[glycan] for glycan in y_train),dtype=np.int64,count=len(y_train))
    y_test = np.fromiter((glycan_indices[glycan] for glycan in y_test),dtype=np.int64,count=len(y_test))
    # Ambiguous labels (Gal(b1-3/4)GlcNAc, Hex, floating parts) also accept every more specific class: cand_mask[i, j] is True if class j is class i or a more specific version of it
    cand_mask = torch.eye(len(glycans), dtype=torch.bool)
    if args.candidate_sets:
        comp_groups = {}
        for index, composition in enumerate(composition_vectors):
            comp_groups.setdefault(composition.tobytes(), []).append(index)
        for group in comp_groups.values():
            for i in group:
                for j in group:
                    if i != j and compare_glycans(glycans[i], glycans[j], subsumes=True):
                        cand_mask[i, j] = True
        print(f"Candidate sets: {int((cand_mask.sum(dim=1) > 1).sum())} ambiguous classes also accept more specific classes")

    # ============================================================
    # Datasets
    # ============================================================
    print("Preparing dataloaders")
    X_train = load_dataset_inputs(train_source, targets=y_train, composition_vectors=composition_vectors, num_bins=input_dim)
    X_test = load_dataset_inputs(val_source, targets=y_test, composition_vectors=composition_vectors, num_bins=input_dim)
    trainset = ResampledDataset(
        X_train, y_train, transform_mz=transform_mz, transform_rt=transform_rt,
        resample_mode=args.resample_dataset,
    )
    valset = SimpleDataset(X_test, y_test)

    pretrain_trainset = None
    if args.sup_pretraining:
        pretrain_trainset = ResampledDataset(
            X_train, y_train, transform_mz=transform_mz, transform_rt=transform_rt,
            resample_mode=args.resample_dataset,
        )
    elif args.unsup_pretraining:
        unsup_pretrain_dir = resolve_prepared_dataset_dir(args.unsup_pretraining_dataset)
        unsup_pretrain_source = (unsup_pretrain_dir if unsup_pretrain_dir.is_file()
                                else unsup_pretrain_dir / args.unsup_pretraining_cache)
        if not unsup_pretrain_source.exists():
            raise FileNotFoundError(f"Could not find unsupervised pretraining inputs: {unsup_pretrain_source}")
        print(f"Loading unsupervised inputs from {unsup_pretrain_source}", flush=True)
        X_pretrain = load_dataset_inputs(
            unsup_pretrain_source,
            default_composition=np.zeros(len(comp_vector_order), dtype=np.float32),
            num_bins=input_dim,
        )
        pretrain_trainset = ResampledDataset(X_pretrain, np.zeros(len(X_pretrain), dtype=np.int64),
                                            transform_mz=transform_mz, transform_rt=transform_rt,
                                            resample_mode=args.resample_dataset)


    # ============================================================
    # Data loaders
    # ============================================================
    pin_memory = torch.cuda.is_available() if args.pin_memory is None else args.pin_memory

    def make_loader(dataset, shuffle, num_workers, generator, batch_sampler=None):
        loader_kwargs = {
            "pin_memory": pin_memory,
            "num_workers": num_workers,
        }
        if batch_sampler is None:
            loader_kwargs.update(
                batch_size=256,
                shuffle=shuffle,
                drop_last=shuffle,
                generator=generator,
            )
        else:
            loader_kwargs["batch_sampler"] = batch_sampler
        if num_workers > 0:
            loader_kwargs.update(
                persistent_workers=args.persistent_workers,
                prefetch_factor=args.prefetch_factor,
                multiprocessing_context=args.multiprocessing_context,
                worker_init_fn=seed_worker,
            )
        return torch.utils.data.DataLoader(dataset, **loader_kwargs)

    train_generator = torch.Generator().manual_seed(args.current_seed)
    val_generator = torch.Generator().manual_seed(args.current_seed + 1)
    pretrain_batch_sampler = None
    pretrain_val_batch_sampler = None
    pretrainloader = None
    pretrain_valloader = None
    if args.sup_pretraining:
        pretrain_batch_sampler = ClassAwareContrastiveBatchSampler(
            pretrain_trainset.y,
            classes_per_batch=args.supcon_classes_per_batch,
            examples_per_class=args.supcon_examples_per_class,
            views_per_example=args.supcon_views_per_example,
            seed=args.current_seed,
        )
        pretrainloader = make_loader(pretrain_trainset, False, args.num_workers, train_generator, pretrain_batch_sampler)
    elif args.unsup_pretraining:
        pretrain_sample_count = len(pretrain_trainset)
        pretrain_batch_sampler = UnlabeledContrastiveBatchSampler(
            pretrain_sample_count,
            examples_per_batch=args.unsup_pretraining_examples_per_batch,
            views_per_example=args.unsup_pretraining_views_per_example,
            seed=args.current_seed,
        )
        pretrain_val_batch_sampler = UnlabeledContrastiveBatchSampler(
            pretrain_sample_count,
            examples_per_batch=args.unsup_pretraining_examples_per_batch,
            views_per_example=args.unsup_pretraining_views_per_example,
            seed=args.current_seed + 1,
        )
        # Spawning workers would copy the entire raw DataFrame into each worker.
        raw_inputs = isinstance(X_pretrain, RawPretrainingInputs)
        pretrainloader = make_loader(pretrain_trainset, False, 0 if raw_inputs else args.num_workers, train_generator, pretrain_batch_sampler)
        pretrain_valloader = make_loader(pretrain_trainset, False, 0 if raw_inputs else args.val_num_workers, val_generator, pretrain_val_batch_sampler)
        if raw_inputs:
            print(f"Raw pretraining pickle: {unsup_pretrain_source}; spectra expanded per item; workers=0")
    # Repeated indices call the augmentation independently and guarantee a
    # positive partner for every SupCon anchor, including singleton classes.
    combined_batch_sampler = (UnlabeledContrastiveBatchSampler(
        len(trainset), examples_per_batch=min(32, len(trainset)), views_per_example=2,
        drop_last=False, seed=args.current_seed,
    ) if args.combine_loss else None)
    trainloader = make_loader(trainset, True, args.num_workers, train_generator, combined_batch_sampler)
    valloader = make_loader(valset, False, args.val_num_workers, val_generator)
    if args.sup_pretraining:
        pretrain_valloader = valloader
    dataloaders = {"train": trainloader,"val": valloader}
    pretrain_dataloaders = {"train": pretrainloader,"val": pretrain_valloader} if pretrainloader is not None else None
    print(
        f"DataLoader settings: train_workers={args.num_workers}, "
        f"val_workers={args.val_num_workers}, persistent={args.persistent_workers}, "
        f"prefetch_factor={args.prefetch_factor}, pin_memory={pin_memory}, "
        f"multiprocessing={args.multiprocessing_context}"
    )
    if pretrain_batch_sampler is not None:
        if args.sup_pretraining:
            print(
                "SupCon sampler: "
                f"classes_per_batch={args.supcon_classes_per_batch}, "
                f"examples_per_class={args.supcon_examples_per_class}, "
                f"views_per_example={args.supcon_views_per_example}, "
                f"original_examples={pretrain_batch_sampler.original_batch_size}, "
                f"embedded_views={pretrain_batch_sampler.batch_size}"
            )
        else:
            print(
                "Unsupervised contrastive sampler: "
                f"examples_per_batch={args.unsup_pretraining_examples_per_batch}, "
                f"views_per_example={args.unsup_pretraining_views_per_example}, "
                f"original_examples={pretrain_batch_sampler.original_batch_size}, "
                f"embedded_views={pretrain_batch_sampler.batch_size}"
            )
    # ============================================================
    # Structural/composition distances
    # ============================================================
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.loss_function == "custom_loss":
        print("Calculating composition/structure distance for loss")
        # Mapping glycans to presence/absence of motifs
        embs = annotate_dataset(glycans, feature_set=["exhaustive"], condense=True)
        # DataFrame of 3-saccharide counts or list of motifs per glycan
        embs2 = get_k_saccharides(glycans, size=3)
        embs2.index = glycans
        embs = pd.concat([embs, embs2],axis=1)
        embs = embs.loc[:, ~embs.T.reset_index().duplicated().to_numpy()]
        embs = (embs.apply(pd.to_numeric,errors="coerce").fillna(0).astype(np.float32))
        dist = pairwise_distances(embs,metric="cosine")
        # The structural distance to an ambiguous label is the distance to its closest candidate
        dist = np.stack([dist[row].min(axis=0) for row in cand_mask.numpy()])
        dist = dist * 1000 * 20
        dist2 = torch.as_tensor(dist, dtype=torch.float32, device=device)
        comps = [glycan_to_composition(k)for k in glycans]
        comp_df = (pd.DataFrame.from_dict(comps).fillna(0))
        dist = pairwise_distances(comp_df,metric="cosine")
        dist = dist * 1000 * 50
        dist3 = torch.as_tensor(dist, dtype=torch.float32, device=device)
        del embs, embs2, comp_df, comps, dist
    # ============================================================
    # Model
    # ============================================================
    print("Preparing the model")
    candidate_sets_tag = "_CS" if args.candidate_sets else "_NoCS"
    loss_tag = LOSS_TAGS.get(args.loss_function, "") + candidate_sets_tag
    if args.sup_pretraining:
        pretraining_tag = f"_SupConPtE{args.pretraining_epochs}T{args.supcon_temperature}"
    elif args.unsup_pretraining:
        pretraining_tag = f"_UnsupConPtE{args.pretraining_epochs}T{args.supcon_temperature}"
    else:
        pretraining_tag = ""
    combine_loss_tag = (f"_CombSupConW{args.contrastive_loss_weight:g}T{args.supcon_temperature:g}" if args.combine_loss else "")
    model_kwargs = {"num_classes": len(glycans),"input_precursor_dim": len(comp_vector_order)}
    model_class = CandyCrunch_CNN
    if args.model == "CNN":
        model_kwargs.update(input_dim=input_dim)
    if args.model == "CNN2":
        model_class = CandyCrunch_CNN2
        model_kwargs.update(
            hidden_dim=args.hidden_dim,
            classifier_moe=args.classifier_moe,
            classifier_num_experts=args.classifier_num_experts,
            classifier_top_k=args.classifier_top_k,
            classifier_expert_hidden_dim=args.classifier_expert_hidden_dim,
            norm_type=args.norm_type,
            activation=args.activation,
            encoder_activation=args.encoder_activation,
            se=args.se,
            se_reduction=args.se_reduction
        )
    elif args.model == "Transformer":
        model_class = CandyCrunch_Transformer
        model_kwargs.update(heads =args.nheads,
                        layers=args.nlayers,
                        ff_dim = args.ff_dim,
                        peak_hidden_dim=resolve_transformer_peak_hidden_dim(args),
                        peak_encoder=args.peak_encoder,
                        max_peaks = args.max_peaks,
                        min_mz=args.min_mz,
                        max_mz=args.max_mz,
                        intensity_threshold=args.intensity_threshold,
                        use_transformer_ff=args.use_transformer_ff,
                        encoder_type = args.encoder_type,
                        norm_type= args.norm_type,
                        activation=args.activation,
                        encoder_activation=args.encoder_activation,
                        num_experts = args.num_experts,
                        moe_top_k=args.moe_top_k,
                        classifier_moe=args.classifier_moe,
                        classifier_num_experts=args.classifier_num_experts,
                        classifier_top_k=args.classifier_top_k,
                        classifier_expert_hidden_dim= args.classifier_expert_hidden_dim)
    model = model_class(**model_kwargs)
    checkpoint_model_class = model_class.__name__
    # --------------------------------------------------------
    # Naming tags
    # --------------------------------------------------------
    encoder_tag = ("NOFF" if not args.use_transformer_ff else (
        f"MoE{args.num_experts}K{args.moe_top_k}" if args.encoder_type == "moe" else "DENSE"))
    if args.classifier_moe:
        classifier_tag = f"MoC'{args.classifier_num_experts}K{args.classifier_top_k}"
        if args.classifier_expert_hidden_dim is not None:
            classifier_tag += (f"H{args.classifier_expert_hidden_dim}")
    else:
        classifier_tag = "ShCl"

    dataset_tag = dataset_dir.name.removeprefix("prepared_datasets_")
    if args.model == "Transformer":
        setting_name = (f"{args.model}_{encoder_tag}_{classifier_tag}_{args.split}_H{args.nheads}L{args.nlayers}"
            f"PHD{resolve_transformer_peak_hidden_dim(args)}_{'FFD' + str(args.ff_dim) + '_' if args.ff_dim is not None else ''}"
            f"PE({args.peak_encoder})_N({args.norm_type})_ACT({args.activation})_EACT({args.encoder_activation}_FF({args.use_transformer_ff})"
            f"{loss_tag}{pretraining_tag}{combine_loss_tag}_{args.resample_dataset}_{dataset_tag}")

    elif args.model == "CNN2":
        se_tag = f"ES{args.se}RS{args.se_reduction}" if args.se else str(args.se)
        setting_name = (f"{args.model}2_{classifier_tag}_{args.split}_HD{args.hidden_dim}_N({args.norm_type})_ACT({args.activation})_EACT({args.encoder_activation})_{se_tag}"
            f"{loss_tag}{pretraining_tag}{combine_loss_tag}_{args.resample_dataset}_{dataset_tag}")
    else:
        setting_name = (f"{args.model}_{classifier_tag}_{args.split}{loss_tag}"
                        f"{pretraining_tag}{combine_loss_tag}_{args.resample_dataset}_{dataset_tag}")

    pretrain_setting_name = setting_name if pretraining_enabled else None
    # ============================================================
    # Checkpoint metadata
    # ============================================================
    checkpoint_metadata = {"checkpoint_version": 5, "model_class": checkpoint_model_class, "model_type":args.model,
        "model_kwargs":model_kwargs, "glycans":list(glycans), "comp_vector_order":list(comp_vector_order),
        "dataset":args.dataset, "split":args.split,
        "setting_name":setting_name,
        "feature_columns": ["binned_intensities","mz_remainder","reducing_mass","glycan_type","RT","mode",
            "lc","modification","trap"],"training_args":vars(args).copy(),}
    if args.model == "Transformer":
        checkpoint_metadata["spectrum_encoding"] = "binned_intensity_ranked_peaks_v1"
    checkpoint_metadata["loss_function"] = args.loss_function
    checkpoint_metadata["combine_loss"] = bool(args.combine_loss)
    checkpoint_metadata["sup_pretraining"] = bool(args.sup_pretraining)
    checkpoint_metadata["unsup_pretraining"] = bool(args.unsup_pretraining)
    if args.unsup_pretraining:
        checkpoint_metadata["unsup_pretraining_dataset"] = args.unsup_pretraining_dataset
        checkpoint_metadata["unsup_pretraining_cache"] = args.unsup_pretraining_cache
    if args.combine_loss:
        checkpoint_metadata["contrastive_loss_weight"] = float(args.contrastive_loss_weight)
        checkpoint_metadata["contrastive_loss_temperature"] = float(args.supcon_temperature)
    if args.loss_function == "xyz_loss":
        checkpoint_metadata["class_composition_vectors"] = composition_vectors.tolist()
    # ============================================================
    # Initialization
    # ============================================================
    model = model.apply(lambda module: init_weights(module,mode="kaiming"))
    # ============================================================
    # Multi-GPU
    # ============================================================
    if torch.cuda.device_count() > 1:
        print("Let's use",torch.cuda.device_count(),"GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    # ============================================================
    # Optimizer
    # ============================================================
    def make_training_components(current_model, patience):
        # ReduceLROnPlateau only cuts the LR after lr_patience + 1 bad epochs, while EarlyStopping stops after patience bad epochs, so lr_patience must stay well below patience
        return training_setup(current_model,
                              lr = 0.005 if args.model == "Transformer" else 0.0001,
                              lr_patience = max(1, patience // 2 - 1),
                              factor=0.2,
                              weight_decay = 0.00002,
                              mode = 'multiclass',
                              num_classes = len(set(glycans)),
                              gsam_alpha = 0.,
                              warmup_epochs = 5)
    # ============================================================
    # Main classification loss
    # ============================================================
    def make_finetuning_criterion():
        if args.loss_function == "cross_entropy":
            return CandidateSetLoss(cand_mask,epsilon=0,label_smoothing=0).to(device)
        if args.loss_function == "focal_loss":
            return FocalLoss(gamma=args.focal_gamma,cand_mask=cand_mask).to(device)
        if args.loss_function == "PolyCrEnr":
            return CandidateSetLoss(cand_mask).to(device)
        if args.loss_function == "xyz_loss":
            return xyz_loss(composition_vectors,cand_mask=cand_mask).to(device)
        if args.loss_function == "custom_loss":
            return custom_loss(CandidateSetLoss(cand_mask),dist2,dist3).to(device)
        raise ValueError(f"Unknown fine-tuning loss_function={args.loss_function!r}.")

    wandb.init(
        project="CandyCrunch_5",
        entity=("vahid-atabaigielmi-university-of-gothenburg"),name=setting_name,save_code=True)
    # Runs with several seeds share one setting_name, so each seed gets its own folder instead of overwriting the previous seed's checkpoint and metrics
    model_dir = Path("./models") if len(args.random_seeds) == 1 else Path("./models") / f"seed{args.current_seed}"
    if pretraining_enabled:
        pretrain_optimizer, pretrain_scheduler, _ = make_training_components(model,
                                                                             args.pretraining_patience if args.pretraining_patience is not None else args.patience)
        if args.sup_pretraining:
            pretrain_criterion = SupConLoss(temperature=args.supcon_temperature,cand_mask=cand_mask).to(device)
            pretrain_loss_name = "supcon"
            pretrain_stage = "supcon_pretraining"
            print("Start SupCon pretraining")
        else:
            pretrain_criterion = UnsupervisedContrastiveLoss(
                temperature=args.supcon_temperature,
                views_per_example=args.unsup_pretraining_views_per_example,
            ).to(device)
            pretrain_loss_name = "unsupcon"
            pretrain_stage = "unsupervised_contrastive_pretraining"
            print("Start unsupervised contrastive pretraining")
        pretrain_metadata = dict(checkpoint_metadata)
        pretrain_metadata["setting_name"] = pretrain_setting_name
        pretrain_metadata["loss_function"] = pretrain_loss_name
        pretrain_metadata["training_stage"] = pretrain_stage
        pretrain_metadata["fine_tune_loss_function"] = args.loss_function
        pretrain_metadata["training_args"] = vars(args).copy()
        pretrain_metadata["training_args"]["loss_function"] = pretrain_loss_name
        pretrain_metadata["training_args"]["fine_tune_loss_function"] = args.loss_function
        pretrain_model_dir = model_dir / "pretrained"
        model = train_model(model,pretrain_dataloaders,pretrain_criterion,pretrain_optimizer,pretrain_scheduler,glycans,
            num_epochs=args.pretraining_epochs,patience=args.pretraining_patience if args.pretraining_patience is not None else args.patience,
            model_type=args.model,setting_name=pretrain_setting_name,
            transformer_moe_aux_loss_weight=args.transformer_moe_aux_loss_weight,
        classifier_moe_aux_loss_weight=args.classifier_moe_aux_loss_weight,checkpoint_metadata=pretrain_metadata
            ,glycan_compositions=glycan_compositions,save_dir=pretrain_model_dir,
            wandb_prefix="pretrain",cand_mask=cand_mask)
        pretrain_model_path = pretrain_model_dir / f"CandyCrunch_{pretrain_setting_name}.pt"
        checkpoint = torch.load(pretrain_model_path,map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint_metadata["pretraining_checkpoint"] = str(pretrain_model_path)
        checkpoint_metadata["pretraining_loss_function"] = pretrain_loss_name
        print(f"Loaded contrastive pretrained weights from {pretrain_model_path}")
        # Release the large unlabelled DataFrame before fine-tuning.
        pretrain_dataloaders = pretrainloader = pretrain_valloader = pretrain_trainset = None
        if args.unsup_pretraining:
            X_pretrain = None

    optimizer_ft, scheduler, _ = make_training_components(model, args.patience)
    criterion = make_finetuning_criterion()
    contrastive_criterion = SupConLoss(temperature=args.supcon_temperature,cand_mask=cand_mask).to(device) if args.combine_loss else None
    print("Fine-tuning loss:", args.loss_function)
    if args.combine_loss:
        print(
            "Fine-tuning combined loss: "
            f"{args.loss_function} + {args.contrastive_loss_weight:g} * SupCon(T={args.supcon_temperature:g})"
        )
    # ============================================================
    # Training
    # ============================================================
    print("Start fine-tuning")
    if args.classifier_moe:
        print("Classifier MoE auxiliary weight:",args.classifier_moe_aux_loss_weight)
    model_ft = train_model(model,dataloaders,criterion,optimizer_ft,scheduler,glycans, num_epochs=args.epoch,
        patience=args.patience,model_type=args.model,setting_name=setting_name,
        transformer_moe_aux_loss_weight=args.transformer_moe_aux_loss_weight,
        classifier_moe_aux_loss_weight=args.classifier_moe_aux_loss_weight,checkpoint_metadata=checkpoint_metadata,glycan_compositions=glycan_compositions,
        contrastive_criterion=contrastive_criterion,contrastive_loss_weight=args.contrastive_loss_weight,
        save_dir=model_dir,cand_mask=cand_mask)
    wandb.finish()
# ================================================================
# Command-line interface
# ================================================================
def build_parser():
    parser = argparse.ArgumentParser(description="CandyCrunch Model")
    # ============================================================
    # Dataset and model type
    # ============================================================
    parser.add_argument("--dataset",type=str,required=True)
    parser.add_argument("--split", type=str,required=True)
    parser.add_argument("--model",type=str,required=True,choices=["CNN","CNN2","Transformer"])
    parser.add_argument("--resample-dataset", "--resample_dataset", dest="resample_dataset",
        choices=["avg", "explode"], default="avg",
        help="Training spectra mode: avg randomly re-pools member spectra (default); explode uses each member as a separate item.")
    # ============================================================
    # Training
    # ============================================================
    parser.add_argument("--epoch",type=int, default=30)
    parser.add_argument("--loss_function",choices=CLASSIFICATION_LOSS_CHOICES,default="custom_loss",
        help="Fine-tuning loss: custom_loss (default), xyz_loss, cross_entropy, PolyCrEnr, or focal_loss.")
    parser.add_argument("--focal_gamma",type=float,default=2.0,
        help="Non-negative focal-loss focusing exponent; 0 gives standard cross-entropy (default: 2).")
    parser.add_argument("--candidate-sets","--candidate_sets",dest="candidate_sets",action=argparse.BooleanOptionalAction,default=True,
        help=("Let ambiguous labels (e.g., Gal(b1-3/4)GlcNAc) also accept every more specific class in all losses and metrics "
              "(default: True; pass --candidate-sets to enable or --no-candidate-sets to train on single-class targets; "
              "setting names include _CS or _NoCS after the loss tag)."))
    parser.add_argument("--sup-pretraining","--sup_pretraining",dest="sup_pretraining",type=str_to_bool,nargs="?",const=True,default=False,
        help="If True, run supervised SupCon pretraining on the labeled --dataset first, then fine-tune with --loss_function (default: False).")
    parser.add_argument("--pretraining",dest="sup_pretraining",type=str_to_bool,nargs="?",const=True,help=argparse.SUPPRESS)
    parser.add_argument("--unsup-pretraining","--unsup_pretraining",dest="unsup_pretraining",type=str_to_bool,nargs="?",const=True,default=False,
        help="If True, run unsupervised contrastive pretraining on --unsup-pretraining-dataset first, then fine-tune with --loss_function (default: False).")
    parser.add_argument("--unsup-pretraining-dataset","--unsup_pretraining_dataset",dest="unsup_pretraining_dataset",
        type=str,default=str(Path(__file__).resolve().parents[1] / "data_processing" / "Final_Results_cc" / "Prefull20261031.pkl"),
        help="Raw CC pretraining pickle, prepared dataset name, or prepared directory (default: data_processing/Final_Results_cc/Prefull20261031.pkl).")
    parser.add_argument("--unsup-pretraining-cache","--unsup_pretraining_cache",dest="unsup_pretraining_cache",
        type=str,default="pretrain_data",help="Input pickle or spectrum-store directory inside the unsupervised pretraining dataset (default: pretrain_data).")
    parser.add_argument("--combine-loss","--combine_loss",type=str_to_bool,nargs="?",const=True,default=False,
        help=("If True, fine-tune with --loss_function plus "
              "--contrastive-loss-weight * SupConLoss using two augmented views per example. Can follow pretraining."))
    parser.add_argument("--contrastive-loss-weight","--contrastive_loss_weight",dest="contrastive_loss_weight",
        type=float,default=0.1,help="Weight for SupConLoss when --combine-loss is True (default: 0.1).")
    parser.add_argument("--pretraining-epochs",type=int,default=10,
        help="Number of contrastive pretraining epochs before fine-tuning (default: 10).")
    parser.add_argument("--pretraining-patience",type=int,default=None,
        help="Early-stopping patience for SupCon pretraining. Defaults to --patience when omitted.")
    parser.add_argument("--supcon-temperature",type=float,default=0.07,
        help="Temperature for supervised or unsupervised contrastive pretraining loss (default: 0.07).")
    parser.add_argument("--supcon-classes-per-batch",type=int,default=64,
        help="Classes sampled per SupCon pretraining batch (default: 64).")
    parser.add_argument("--supcon-examples-per-class",type=int,default=4,
        help="Distinct original examples sampled per class for SupCon pretraining (default: 4).")
    parser.add_argument("--supcon-views-per-example",type=int,default=2,
        help="Independent augmented views per original example for SupCon pretraining (default: 2).")
    parser.add_argument("--unsup-pretraining-examples-per-batch","--unsup_pretraining_examples_per_batch",
        dest="unsup_pretraining_examples_per_batch",type=int,default=128,
        help="Original unlabeled spectra per unsupervised contrastive pretraining batch (default: 128).")
    parser.add_argument("--unsup-pretraining-views-per-example","--unsup_pretraining_views_per_example",
        dest="unsup_pretraining_views_per_example",type=int,default=2,
        help="Independent augmented views per unlabeled spectrum for unsupervised pretraining (default: 2).")
    parser.add_argument("--patience", type = int, default = 6)
    parser.add_argument("--num-workers",type=int,default=2,help="Training DataLoader workers.")
    parser.add_argument("--val-num-workers",type=int,default=0,help="Validation DataLoader workers.")
    parser.add_argument("--prefetch-factor",type=int,default=1)
    parser.add_argument("--persistent-workers",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--pin-memory",action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument("--multiprocessing-context",choices=["spawn", "forkserver", "fork"],default="spawn")
    # ============================================================
    # CNN2
    # ============================================================
    parser.add_argument("--se","--se",type=str_to_bool,nargs="?",const=True,default=False,
        help=("If True, use SE blocks."))
    parser.add_argument("--se-reduction","--se-reduction",type=int,default=8,
        help=("SE reduction factor."))
    parser.add_argument("--hidden-dim","--hidden-dim",type=int,default=1024,
        help=("Hidden dimension."))
    # ============================================================
    # Transformer
    # ============================================================
    parser.add_argument("--nheads",type=int,default=4)
    parser.add_argument("--peak_encoder", "--peak-encoder", choices=["linear", "fourier"], default="fourier",
        help="Numeric m/z:intensity encoder, matching the reference Transformer implementation.")
    parser.add_argument(
        "--disable_mha_fastpath",
        action="store_true",
        help="Disable the optimized Transformer attention path to diagnose validation-time CUDA errors.",
    )
    parser.add_argument( "--nlayers",type=int,default=2)
    parser.add_argument( "--ff_dim",type=int,default=256)
    parser.add_argument("--peak_hidden_dim",type=int,default=None,help=("Transformer model width. If omitted, it defaults to ff_dim//2 when ff_dim is given, otherwise nheads * 64."))
    parser.add_argument("--max-peaks", "--max_peaks", "--max-seq-length", "--max_seq_length", dest = "max_peaks", type = int, default = None,
        help = "Keep only the most intense peaks of each spectrum (without [CLS]). Default: all input bins fit; 512 matches GlycoTrans's usual limit.")
    parser.add_argument("--min-mz", type = float, default = 39.714,
        help = "First m/z edge of the input bin grid, linspace(min_mz, max_mz, 2048) (default: 39.714).")
    parser.add_argument("--max-mz", type = float, default = 3000.0,
        help = "Last m/z edge of the input bin grid (default: 3000).")
    parser.add_argument("--intensity-threshold", type = float, default = 0.0,
        help = "Minimum share of a spectrum's total intensity for a bin to count as a peak (default: 0 keeps every non-empty bin).")
    parser.add_argument("--use_transformer_ff",action=argparse.BooleanOptionalAction,default=True,help=(
            "Enable the Transformer feed-forward block. When disabled(--no-use_transformer_ff), the encoder uses attention only and ignores --encoder_type for the token FF path."))
    parser.add_argument("--encoder_type",type=str,default="dense",choices=["dense","moe"])
    # ============================================================
    # Token-level Transformer MoE
    # ============================================================
    parser.add_argument("--num_experts",type=int,default=4,help=("Number of FFN experts in each Transformer MoE layer."))
    parser.add_argument("--moe_top_k", type=int,default=2, help=("Number of Transformer FFN experts selected per token."))
    parser.add_argument("--transformer_moe_aux_loss_weight", type=float,default=0.01,help=("Weight applied to the Transformer MoE load-balancing loss."))
    # ============================================================
    # NEW: spectrum-level classifier MoE(CNN2 and Transformer)
    # ============================================================
    parser.add_argument("--norm_type",type=str,default="layer",choices=["layer","rms"])
    parser.add_argument("--activation",type=str,default="leaky_relu", choices=["relu","gelu","leaky_relu","silu","elu"])
    parser.add_argument("--encoder_activation", default="leaky_relu", choices=["relu", "gelu", "leaky_relu", "silu", "elu"])
    parser.add_argument("--classifier_moe", action="store_true", help=("Replace the CNN2 or Transformer final classifier with a spectrum-level MoE classifier."))
    parser.add_argument("--classifier_num_experts",type=int,default=4,help=("Number of classifier experts."))
    parser.add_argument("--classifier_top_k",type=int,default=2,help=("Number of classifier experts selected for each spectrum."))
    parser.add_argument("--classifier_expert_hidden_dim",type=int,default=None,help=("Hidden dimension inside each classifier expert. If omitted, every expert is a"
                                                                                     "single Linear(512, num_classes) layer."))
    parser.add_argument("--classifier_moe_aux_loss_weight",type=float,default=0.01,help=("Weight applied to the classifier MoE load-balancing loss."))
    # ============================================================
    # Seeds
    # ============================================================
    parser.add_argument("--random_seeds",nargs="+",type=int,default=[42],help="List of random seeds.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    for seed in args.random_seeds:
        print(f"\n=== Running with seed {seed} ===")
        args.current_seed = seed
        main(args)
