import pickle
import warnings
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from candycrunch.losses import CandidateSetLoss, ClassAwareContrastiveBatchSampler, FocalLoss, SupConLoss, custom_loss, xyz_loss
from glycowork.ml.models import init_weights
from glycowork.ml.model_training import training_setup
from glycowork.motif.graph import compare_glycans
from memmap_dataset import ensure_memmap_cache
from memory_utils import report_process_tree_memory
import pandas as pd
import wandb
import candycrunch.model
from candycrunch.model import CandyCrunch_CNN, CandyCrunch_Transformer, MemmapSpectrumDataset, SimpleDataset, TransDataset, transform_mz, transform_rt
from glycowork.motif.annotate import annotate_dataset, get_k_saccharides
from glycowork.motif.tokenization import glycan_to_composition
from sklearn.metrics import pairwise_distances
from training_utils import seed_worker, train_model

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


def main(args):
    set_seed(args.current_seed)
    if args.pretraining and args.combine_loss:
        raise ValueError("Use either --pretraining True or --combine-loss True, not both.")
    if args.pretraining and args.pretraining_epochs < 1:
        raise ValueError("--pretraining-epochs must be at least 1 when --pretraining is True.")
    if args.combine_loss and (not np.isfinite(args.contrastive_loss_weight) or args.contrastive_loss_weight <= 0):
        raise ValueError("--contrastive-loss-weight must be finite and positive when --combine-loss is True.")
    if args.model == "Transformer" and args.disable_mha_fastpath:
        torch.backends.mha.set_fastpath_enabled(False)
        print("Transformer attention fastpath disabled for CUDA diagnostics.")
    memory_reporter = report_process_tree_memory if args.report_memory else None
    if memory_reporter is not None:
        memory_reporter("startup")
    print("Reading data")
    # ============================================================
    # Load data
    # ============================================================
    dataset_dir = Path(f"./prepared_datasets_{args.dataset}")
    train_feature_path = dataset_dir / f"X_train_{args.split}.pkl"
    val_feature_path = dataset_dir / f"X_test_{args.split}.pkl"
    train_cache = ensure_memmap_cache(
        train_feature_path,
        dataset_dir / ".candycrunch_memmap" / f"train_{args.split}",
        args.model,
        memory_reporter=memory_reporter,
    )
    val_cache = ensure_memmap_cache(
        val_feature_path,
        dataset_dir / ".candycrunch_memmap" / f"val_{args.split}",
        args.model,
        memory_reporter=memory_reporter,
    )

    # with open(dataset_dir / f"X_train_{args.split}.pkl", "rb") as file:
    #     X_train = pickle.load(file)
    # with open(dataset_dir / f"X_test_{args.split}.pkl", "rb") as file:
    #     X_test = pickle.load(file)
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

    # ## For SimpleDataset and TransDataset and deactivate for MemmapSpectrumDataset
    # X_train = [(*sample[:3], composition_vectors[target], *sample[4:]) for sample, target in zip(X_train, y_train, strict=True)]
    # X_test = [(*sample[:3], composition_vectors[target], *sample[4:]) for sample, target in zip(X_test, y_test, strict=True)]
    # if args.model == "Transformer" and args.max_peaks is not None:
    #     X_train = [(*sample[:1], sample[1][:args.max_peaks], *sample[2:]) for sample in X_train]
    #     X_test = [(*sample[:1], sample[1][:args.max_peaks], *sample[2:]) for sample in X_test]
    # ============================================================
    # Datasets
    # ============================================================
    print("Preparing dataloaders")
    trainset = MemmapSpectrumDataset(train_cache,
                                     y_train,
                                     composition_vectors,
                                     args.model,
                                     max_peaks=args.max_peaks,
                                     transform_mz=transform_mz if args.model == "CNN" else None,
                                     transform_rt=transform_rt if args.model == "CNN" else None)

    pretrain_trainset = None
    if args.pretraining:
        pretrain_trainset = MemmapSpectrumDataset(train_cache,
                                                  y_train,
                                                  composition_vectors,
                                                  args.model,
                                                  max_peaks=args.max_peaks,
                                                  transform_mz=transform_mz if args.model == "CNN" else None,
                                                  transform_rt=transform_rt)

    valset = MemmapSpectrumDataset(val_cache,
                                   y_test,
                                   composition_vectors,
                                   args.model,
                                   max_peaks=args.max_peaks)

    if memory_reporter is not None:
        memory_reporter("datasets ready")
    if args.prepare_data_only:
        print("Memory-map caches are ready; exiting before model setup.")
        return

    # if args.model == "CNN":
    #     trainset = SimpleDataset(X_train, y_train, transform_mz = transform_mz, transform_rt = transform_rt)
    #     valset = SimpleDataset(X_test, y_test)
    # elif args.model == "Transformer":
    #     trainset = TransDataset(X_train, y_train, transform_rt = transform_rt)
    #     valset = TransDataset(X_test, y_test)
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
    pretrainloader = None
    if args.pretraining:
        pretrain_batch_sampler = ClassAwareContrastiveBatchSampler(
            y_train,
            classes_per_batch=args.supcon_classes_per_batch,
            examples_per_class=args.supcon_examples_per_class,
            views_per_example=args.supcon_views_per_example,
            seed=args.current_seed,
        )
        pretrainloader = make_loader(pretrain_trainset, False, args.num_workers, train_generator, pretrain_batch_sampler)
    trainloader = make_loader(trainset, True, args.num_workers, train_generator)
    valloader = make_loader(valset, False, args.val_num_workers, val_generator)
    dataloaders = {"train": trainloader,"val": valloader}
    pretrain_dataloaders = {"train": pretrainloader,"val": valloader} if pretrainloader is not None else None
    print(
        f"DataLoader settings: train_workers={args.num_workers}, "
        f"val_workers={args.val_num_workers}, persistent={args.persistent_workers}, "
        f"prefetch_factor={args.prefetch_factor}, pin_memory={pin_memory}, "
        f"multiprocessing={args.multiprocessing_context}"
    )
    if pretrain_batch_sampler is not None:
        print(
            "SupCon sampler: "
            f"classes_per_batch={args.supcon_classes_per_batch}, "
            f"examples_per_class={args.supcon_examples_per_class}, "
            f"views_per_example={args.supcon_views_per_example}, "
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
    loss_tag = LOSS_TAGS.get(args.loss_function, "") + ("_CS" if args.candidate_sets else "")
    pretraining_tag = (f"_SupConPtE{args.pretraining_epochs}T{args.supcon_temperature}" if args.pretraining else "")
    combine_loss_tag = (f"_CombSupConW{args.contrastive_loss_weight:g}T{args.supcon_temperature:g}" if args.combine_loss else "")
    if args.model == "CNN":
        model_kwargs = {"input_dim": 2048,"num_classes": len(glycans),"input_precursor_dim": len(comp_vector_order),
            "classifier_moe": args.classifier_moe,"classifier_num_experts": args.classifier_num_experts,
            "classifier_top_k": args.classifier_top_k,"classifier_expert_hidden_dim": args.classifier_expert_hidden_dim}
        model = CandyCrunch_CNN(**model_kwargs)
        checkpoint_model_class = ("CandyCrunch_CNN")
        if args.classifier_moe:
            classifier_tag = (f"MoC{args.classifier_num_experts}K{args.classifier_top_k}")
            if args.classifier_expert_hidden_dim is not None:
                classifier_tag += (f"H{args.classifier_expert_hidden_dim}")
        else:
            classifier_tag = "ShCl"
        setting_name = (f"{args.model}_{classifier_tag}_{args.split}{loss_tag}{pretraining_tag}{combine_loss_tag}_{args.dataset}")
    elif args.model == "Transformer":
        model_kwargs = {
            "num_classes": len(glycans),"input_precursor_dim": len(comp_vector_order),"heads": args.nheads,"layers": args.nlayers,
            "ff_dim": args.ff_dim,"peak_hidden_dim": resolve_transformer_peak_hidden_dim(args),"peak_encoder": args.peak_encoder,
            "use_transformer_ff":args.use_transformer_ff,"encoder_type": args.encoder_type, "norm_type": args.norm_type,
            "activation": args.activation, "encoder_activation":args.encoder_activation, "use_resunits": args.use_resunits,
            "num_experts":args.num_experts,"moe_top_k":args.moe_top_k,"classifier_moe":args.classifier_moe,
            "classifier_num_experts":args.classifier_num_experts,"classifier_top_k":args.classifier_top_k,
            "classifier_expert_hidden_dim":args.classifier_expert_hidden_dim,}
        model = CandyCrunch_Transformer(**model_kwargs)
        checkpoint_model_class = ("CandyCrunch_Transformer")
        # --------------------------------------------------------
        # Naming tags
        # --------------------------------------------------------
        encoder_tag = ("NOFF" if not args.use_transformer_ff else (f"MoE{args.num_experts}K{args.moe_top_k}" if args.encoder_type == "moe" else "DENSE"))
        if args.classifier_moe:
            classifier_tag = (f"MoE{args.classifier_num_experts}K{args.classifier_top_k}")
            if (args.classifier_expert_hidden_dim is not None):
                classifier_tag += (f"H{args.classifier_expert_hidden_dim}")
        else:
            classifier_tag = "ShCl"
        setting_name = (f"{args.model}_{encoder_tag}_{classifier_tag}_{args.split}_H{args.nheads}L{args.nlayers}"
            f"PHD{resolve_transformer_peak_hidden_dim(args)}_{'FFD' + str(args.ff_dim) + '_' if args.ff_dim is not None else ''}"
            f"MP{args.max_peaks}_PE({args.peak_encoder})_N({args.norm_type})_ACT({args.activation})_FF({args.use_transformer_ff})_"
            f"RU({args.use_resunits}){loss_tag}{pretraining_tag}{combine_loss_tag}_{args.dataset}")
    pretrain_setting_name = setting_name if args.pretraining else None
    # ============================================================
    # Checkpoint metadata
    # ============================================================
    checkpoint_metadata = {"checkpoint_version": 5, "model_class": checkpoint_model_class, "model_type":args.model,
        "model_kwargs":model_kwargs, "glycans":list(glycans), "comp_vector_order":list(comp_vector_order),
        "max_peaks":args.max_peaks,"dataset":args.dataset, "split":args.split,
        "setting_name":setting_name,
        "feature_columns": ["binned_intensities","peak_list","mz_remainder","reducing_mass","glycan_type","RT","mode",
            "lc","modification","trap"],"training_args":vars(args).copy(),}
    checkpoint_metadata["loss_function"] = args.loss_function
    checkpoint_metadata["combine_loss"] = bool(args.combine_loss)
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
                              lr = 0.0001,
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
        project="CandyCrunch_3",
        entity=("vahid-atabaigielmi-university-of-gothenburg"),name=setting_name,save_code=True)
    # Runs with several seeds share one setting_name, so each seed gets its own folder instead of overwriting the previous seed's checkpoint and metrics
    model_dir = Path("./models") if len(args.random_seeds) == 1 else Path("./models") / f"seed{args.current_seed}"
    if args.pretraining:
        pretrain_optimizer, pretrain_scheduler, _ = make_training_components(model,
                                                                             args.pretraining_patience if args.pretraining_patience is not None else args.patience)
        pretrain_criterion = SupConLoss(temperature=args.supcon_temperature,cand_mask=cand_mask).to(device)
        pretrain_metadata = dict(checkpoint_metadata)
        pretrain_metadata["setting_name"] = pretrain_setting_name
        pretrain_metadata["loss_function"] = "supcon"
        pretrain_metadata["training_stage"] = "supcon_pretraining"
        pretrain_metadata["fine_tune_loss_function"] = args.loss_function
        pretrain_metadata["training_args"] = vars(args).copy()
        pretrain_metadata["training_args"]["loss_function"] = "supcon"
        pretrain_metadata["training_args"]["fine_tune_loss_function"] = args.loss_function
        print("Start SupCon pretraining")
        pretrain_model_dir = model_dir / "pretrained"
        model = train_model(model,pretrain_dataloaders,pretrain_criterion,pretrain_optimizer,pretrain_scheduler,glycans,
            num_epochs=args.pretraining_epochs,patience=args.pretraining_patience if args.pretraining_patience is not None else args.patience,
            model_type=args.model,setting_name=pretrain_setting_name,transformer_moe_aux_loss_weight=args.transformer_moe_aux_loss_weight,
            classifier_moe_aux_loss_weight=args.classifier_moe_aux_loss_weight,checkpoint_metadata=pretrain_metadata,
            memory_reporter=memory_reporter,glycan_compositions=glycan_compositions,save_dir=pretrain_model_dir,
            wandb_prefix="pretrain",cand_mask=cand_mask)
        pretrain_model_path = pretrain_model_dir / f"CandyCrunch_{pretrain_setting_name}.pt"
        checkpoint = torch.load(pretrain_model_path,map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint_metadata["pretraining_checkpoint"] = str(pretrain_model_path)
        checkpoint_metadata["pretraining_loss_function"] = "supcon"
        print(f"Loaded SupCon pretrained weights from {pretrain_model_path}")

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
    if args.model == "Transformer":
        print("Transformer feed-forward enabled:",args.use_transformer_ff)
        print("Transformer MoE auxiliary weight:",args.transformer_moe_aux_loss_weight)
    if args.classifier_moe:
        print("Classifier MoE auxiliary weight:",args.classifier_moe_aux_loss_weight)
    model_ft = train_model(model,dataloaders,criterion,optimizer_ft,scheduler,glycans, num_epochs=args.epoch,
        patience=args.patience,model_type=args.model,setting_name=setting_name, transformer_moe_aux_loss_weight=args.transformer_moe_aux_loss_weight,
        classifier_moe_aux_loss_weight=args.classifier_moe_aux_loss_weight,checkpoint_metadata=checkpoint_metadata,
        memory_reporter=memory_reporter,glycan_compositions=glycan_compositions,
        contrastive_criterion=contrastive_criterion,contrastive_loss_weight=args.contrastive_loss_weight,
        save_dir=model_dir,cand_mask=cand_mask)
    wandb.finish()
# ================================================================
# Command-line interface
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CandyCrunch Model")
    # ============================================================
    # Dataset
    # ============================================================
    parser.add_argument("--dataset",type=str,required=True)
    parser.add_argument("--split", type=str,required=True)
    parser.add_argument("--model",type=str,required=True,choices=["CNN","Transformer"])
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
              "(default: True; --no-candidate-sets trains on single-class targets)."))
    parser.add_argument("--pretraining",type=str_to_bool,nargs="?",const=True,default=False,
        help="If True, run SupCon pretraining first, then fine-tune with --loss_function (default: False).")
    parser.add_argument("--combine-loss",type=str_to_bool,nargs="?",const=True,default=False,
        help=("If True, fine-tune with --loss_function plus "
              "--contrastive-loss-weight * SupConLoss. Mutually exclusive with --pretraining."))
    parser.add_argument("--contrastive-loss-weight","--contrastive_loss_weight",dest="contrastive_loss_weight",
        type=float,default=0.1,help="Weight for SupConLoss when --combine-loss is True (default: 0.1).")
    parser.add_argument("--pretraining-epochs",type=int,default=10,
        help="Number of SupCon pretraining epochs before fine-tuning (default: 10).")
    parser.add_argument("--pretraining-patience",type=int,default=None,
        help="Early-stopping patience for SupCon pretraining. Defaults to --patience when omitted.")
    parser.add_argument("--supcon-temperature",type=float,default=0.07,
        help="Temperature for supervised contrastive pretraining loss (default: 0.07).")
    parser.add_argument("--supcon-classes-per-batch",type=int,default=64,
        help="Classes sampled per SupCon pretraining batch (default: 64).")
    parser.add_argument("--supcon-examples-per-class",type=int,default=4,
        help="Distinct original examples sampled per class for SupCon pretraining (default: 4).")
    parser.add_argument("--supcon-views-per-example",type=int,default=2,
        help="Independent augmented views per original example for SupCon pretraining (default: 2).")
    parser.add_argument("--patience", type = int, default = 6)
    parser.add_argument("--max_peaks",type=int,default=None)
    parser.add_argument("--num-workers",type=int,default=2,help="Training DataLoader workers.")
    parser.add_argument("--val-num-workers",type=int,default=0,help="Validation DataLoader workers.")
    parser.add_argument("--prefetch-factor",type=int,default=1)
    parser.add_argument("--persistent-workers",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--pin-memory",action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument("--multiprocessing-context",choices=["spawn", "forkserver", "fork"],default="spawn")
    parser.add_argument("--report-memory",action="store_true")
    parser.add_argument("--prepare-data-only",action="store_true")
    # ============================================================
    # Transformer
    # ============================================================
    parser.add_argument("--nheads",type=int,default=None)
    parser.add_argument(
        "--disable_mha_fastpath",
        action="store_true",
        help="Disable the optimized Transformer attention path to diagnose validation-time CUDA errors.",
    )
    parser.add_argument( "--nlayers",type=int,default=None)
    parser.add_argument( "--ff_dim",type=int,default=None)
    parser.add_argument("--peak_hidden_dim",type=int,default=None,help=("Transformer model width. If omitted, it defaults to ff_dim//2 when ff_dim is given, otherwise nheads * 64."))
    parser.add_argument("--peak_encoder",type=str, default="fourier", choices=["linear","fourier"],
        help="Peak encoder for the Transformer; 'linear' feeds raw m/z into a LayerNorm, which cancels its scale and leaves peaks nearly indistinguishable (default: fourier).")
    parser.add_argument("--use_transformer_ff",action=argparse.BooleanOptionalAction,default=True,help=(
            "Enable the Transformer feed-forward block. When disabled(--no-use_transformer_ff), the encoder uses attention only and ignores --encoder_type for the token FF path."))
    parser.add_argument("--encoder_type",type=str,default="dense",choices=["dense","moe"])
    parser.add_argument("--norm_type",type=str,default="layer",choices=["layer","rms"])
    parser.add_argument("--activation",type=str,default="leaky_relu", choices=["relu","gelu","leaky_relu","silu","elu"])
    parser.add_argument( "--encoder_activation", default="gelu", choices=["relu", "gelu", "leaky_relu", "silu","elu"])
    # ============================================================
    # Token-level Transformer MoE
    # ============================================================
    parser.add_argument("--num_experts",type=int,default=4,help=("Number of FFN experts in each Transformer MoE layer."))
    parser.add_argument("--moe_top_k", type=int,default=2, help=("Number of Transformer FFN experts selected per token."))
    parser.add_argument("--transformer_moe_aux_loss_weight", type=float,default=0.01,help=("Weight applied to the Transformer MoE load-balancing loss."))
    # ============================================================
    # NEW: spectrum-level classifier MoE
    # ============================================================
    parser.add_argument("--classifier_moe", action="store_true", help=("Replace the shared final classifier with a spectrum-level MoE classifier."))
    parser.add_argument("--classifier_num_experts",type=int,default=4,help=("Number of classifier experts."))
    parser.add_argument("--classifier_top_k",type=int,default=2,help=("Number of classifier experts selected for each spectrum."))
    parser.add_argument("--classifier_expert_hidden_dim",type=int,default=None,help=("Hidden dimension inside each classifier expert. If omitted, every expert is a"
                                                                                     "single Linear(512, num_classes) layer."))
    parser.add_argument("--classifier_moe_aux_loss_weight",type=float,default=0.01,help=("Weight applied to the classifier MoE load-balancing loss."))
    # ============================================================
    # ResUnits
    # ============================================================
    parser.add_argument("--use_resunits",action="store_true", help=("Use ResUnit convolution blocks before the Transformer encoder"))
    # ============================================================
    # Seeds
    # ============================================================
    parser.add_argument("--random_seeds",nargs="+",type=int,default=[42],help="List of random seeds.")
    args = parser.parse_args()
    for seed in args.random_seeds:
        print(f"\n=== Running with seed {seed} ===")
        args.current_seed = seed
        main(args)
