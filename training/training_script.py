import pickle
import pandas as pd
from candycrunch.model import (SimpleDataset, TransDataset, CandyCrunch_CNN, transform_mz, transform_rt,
                               CandyCrunch_Transformer)
from glycowork.motif.annotate import annotate_dataset, get_k_saccharides
from glycowork.motif.tokenization import get_stem_lib, glycan_to_composition
from training_utils import *
from sklearn.metrics import pairwise_distances
import warnings
import argparse
import random
import candycrunch.model

print("USING MODEL FILE:", candycrunch.model.__file__)

# Suppress the specific sklearn runtime warnings
warnings.filterwarnings("ignore", category = RuntimeWarning, module = "sklearn.utils.extmath")


def set_seed(seed = None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def truncate_peak_lists(features, max_peaks = None):
    if max_peaks is None:
        return features

    truncated = []

    for t in features:
        peak_list = t[1]

        if peak_list.shape[0] > max_peaks:
            peak_list = peak_list[:max_peaks]

        truncated.append(
            t[:1] + (peak_list,) + t[2:]
        )

    return truncated


def main(args):
    set_seed(args.current_seed)
    print("Reading data")
    # Train and test data can be found on zenodo at https://doi.org/10.5281/zenodo.7940046
    # Please modify the filepaths below to point to your downloaded files

    with open(f"./prepared_datasets_{args.dataset}/X_train_{args.split}.pkl", "rb") as file:
        X_train = pickle.load(file)
    with open(f"./prepared_datasets_{args.dataset}/X_test_{args.split}.pkl", "rb") as file:
        X_test = pickle.load(file)
    with open(f"./prepared_datasets_{args.dataset}/y_train_{args.split}.pkl", "rb") as file:
        y_train = pickle.load(file)
    with open(f"./prepared_datasets_{args.dataset}/y_test_{args.split}.pkl", "rb") as file:
        y_test = pickle.load(file)
    with open(f"./glycans.pkl", "rb") as file:
        glycans = pickle.load(file)

    print("Preprocessing data")
    # X_train, y_train, glycans = filter_data_exceptions(X_train, y_train, glycans)
    # X_test, y_test, glycans = filter_data_exceptions(X_test, y_test, glycans)

    X_train = truncate_peak_lists(X_train, args.max_peaks)
    X_test = truncate_peak_lists(X_test, args.max_peaks)

    disallowed_glycans = []
    allowed_glycan_comps = {}
    for glyc in glycans:
        try:
            glycomp = glycan_to_composition(glyc)
            allowed_glycan_comps[glyc] = glycomp
        except KeyError:
            disallowed_glycans.append(glyc)
    comp_vector_order = list(set(x for y in allowed_glycan_comps.values() for x in y))
    comp_vector_order = sorted(comp_vector_order, key = lambda x: x.lower())
    print(f"Comp_vector_order: {comp_vector_order}")
    glycan_comp_vect_map = {}
    for glyc, glycomp in allowed_glycan_comps.items():
        comp_vect = np.zeros(len(comp_vector_order))
        for mono, counts in glycomp.items():
            comp_vect[comp_vector_order.index(mono)] = counts
        glycan_comp_vect_map[glyc] = comp_vect
    X_train = [t[:3] + (glycan_comp_vect_map[gt],) + t[4:]
               for t, gt in zip(X_train, y_train)]
    X_test = [t[:3] + (glycan_comp_vect_map[gt],) + t[4:]
              for t, gt in zip(X_test, y_test)]

    y_train = [glycans.index(c) for c in y_train]
    y_test = [glycans.index(c) for c in y_test]

    print("Preparing dataloaders")
    if args.model == "CNN":
        trainset = SimpleDataset(X_train, y_train, transform_mz = transform_mz, transform_rt = transform_rt)
        valset = SimpleDataset(X_test, y_test)
    elif args.model == "Transformer":
        trainset = TransDataset(X_train, y_train, transform_rt = transform_rt)
        valset = TransDataset(X_test, y_test)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size = 256,
        shuffle = True,
        drop_last = True,
        pin_memory = True,
        num_workers = 4,
        persistent_workers = True,
        prefetch_factor = 2,
    )

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size = 256,
        shuffle = False,
        drop_last = True,
        pin_memory = True,
        num_workers = 4,
        persistent_workers = True,
        prefetch_factor = 2,
    )
    dataloaders = {'train': trainloader, 'val': valloader}
    print("Calculating composition/structure distance for loss")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    embs = annotate_dataset(glycans, feature_set = ['exhaustive'], condense = True)
    embs2 = get_k_saccharides(glycans, size = 3)
    embs2.index = glycans
    embs = pd.concat([embs, embs2], axis = 1)
    embs = embs.apply(pd.to_numeric, errors = 'coerce').fillna(0).astype(np.float32)
    dist = pairwise_distances(embs, metric = 'cosine')
    dist = dist * 1000 * 20
    dist2 = torch.tensor(dist, requires_grad = True).to(device)
    comps = [glycan_to_composition(k) for k in glycans]
    comp_df = pd.DataFrame.from_dict(comps).fillna(0)
    dist = pairwise_distances(comp_df, metric = 'cosine')
    dist = dist * 1000 * 50
    dist3 = torch.tensor(dist, requires_grad = True).to(device)
    print("Preparing the model")

    if args.model == "CNN":
        model_kwargs = {
            "input_dim": 2048,
            "num_classes": len(glycans),
            "input_precursor_dim": len(comp_vector_order),
        }

        model = CandyCrunch_CNN(**model_kwargs)
        checkpoint_model_class = "CandyCrunch_CNN"
        setting_name = f"{args.model}_{args.split}_{args.dataset}"

    elif args.model == "Transformer":
        model_kwargs = {
            "num_classes": len(glycans),
            "input_precursor_dim": len(comp_vector_order),
            "heads": args.nheads,
            "layers": args.nlayers,
            "ff_dim": args.ff_dim,
            "peak_encoder": args.peak_encoder,
            "encoder_type": args.encoder_type,
            "norm_type": args.norm_type,
            "activation": args.activation,
            "encoder_activation": args.encoder_activation,
            "use_resunits": args.use_resunits,
            "num_experts": args.num_experts,
            "moe_top_k": args.moe_top_k,
        }

        model = CandyCrunch_Transformer(**model_kwargs)
        checkpoint_model_class = "CandyCrunch_Transformer"

        encoder_tag = "MOE" if args.encoder_type == "moe" else "DENSE"

        setting_name = (f"{args.model}_{encoder_tag}_{args.split}_H{args.nheads}L{args.nlayers}FFD{args.ff_dim}"
                        f"MP{args.max_peaks}_PE({args.peak_encoder})_N({args.norm_type})_ACT({args.activation})"
                        f"_RU({args.use_resunits})_{args.dataset}"
                        )
    checkpoint_metadata = {"checkpoint_version": 1, "model_class": checkpoint_model_class, "model_type": args.model,
                           "model_kwargs": model_kwargs, "glycans": list(glycans),
                           "comp_vector_order": list(comp_vector_order), "max_peaks": args.max_peaks,
                           "dataset": args.dataset,
                           "split": args.split, "setting_name": setting_name,
                           "feature_columns": ["binned_intensities", "peak_list", "mz_remainder", "reducing_mass",
                                               "glycan_type",
                                               "RT", "mode", "lc", "modification", "trap"],
                           "training_args": vars(args).copy()}
    os.environ["WANDB_API_KEY"] = "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"
    wandb.init(project = f'CandyCrunch', entity = 'vahid-atabaigielmi-university-of-gothenburg', name = setting_name, save_code = True)
    model = model.apply(lambda module: init_weights(module, mode = 'kaiming'))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    if args.model == "CNN":
        optimizer_ft, scheduler, criterion = training_setup(model, 0.0001, weight_decay = 0.00002,
                                                            num_classes = len(set(glycans)))
    elif args.model == "Transformer":
        optimizer_ft, scheduler, criterion = training_setup(model, 0.001, weight_decay = 0.000002,
                                                            num_classes = len(set(glycans)))
    primary_loss = Poly1CrossEntropyLoss(num_classes = len(glycans), epsilon = 1, reduction = 'mean').to(device)
    criterion = custom_loss(primary_loss, dist2, dist3).to(device)

    print("Start training")
    model_ft = train_model(
        model,
        dataloaders,
        criterion,
        optimizer_ft,
        scheduler,
        glycans,
        num_epochs = args.epoch,
        patience = args.patience,
        model_type = args.model,
        setting_name = setting_name,
        moe_aux_loss_weight = args.moe_aux_loss_weight,
        checkpoint_metadata = checkpoint_metadata)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'CandyCrunch Model')
    parser.add_argument('--dataset', type = str, required = True)
    parser.add_argument('--split', type = str, required = True)
    parser.add_argument('--model', type = str, required = True, choices = ['CNN', 'Transformer'])
    parser.add_argument('--epoch', type = int, required = False, default = 30)
    parser.add_argument('--patience', type = int, required = False, default = 4)
    parser.add_argument('--max_peaks', type = int, required = False)
    parser.add_argument('--nheads', type = int, required = False)
    parser.add_argument('--nlayers', type = int, required = False)
    parser.add_argument('--ff_dim', type = int, required = False)
    parser.add_argument("--peak_encoder", type = str, default = "linear", required = False,
                        choices = ["linear", "fourier"])
    parser.add_argument("--moe_aux_loss_weight", type = float, default = None, required = False,
                        help = "Weight for MoE load-balancing auxiliary loss.")
    parser.add_argument("--encoder_type", type = str, default = "dense", choices = ["dense", "moe"])
    parser.add_argument("--norm_type", type = str, default = "layer", choices = ["layer", "rms"])
    parser.add_argument("--activation", type = str, default = "leaky_relu",
                        choices = ["relu", "gelu", "leaky_relu", "silu", "elu"])
    parser.add_argument("--encoder_activation", type = str, default = "gelu",
                        choices = ["relu", "gelu", "leaky_relu", "silu", "elu"])
    parser.add_argument("--num_experts", type = int, default = 4)
    parser.add_argument("--moe_top_k", type = int, default = 2)
    parser.add_argument("--use_resunits", action = "store_true", required = False,
                        help = "Use ResUnit convolution blocks before the Transformer encoder.")
    parser.add_argument('--random_seeds', nargs = '+', type = int, default = [42],
                        help = 'List of random seeds (default: [42, 123, 456, 789, 999])')
    args = parser.parse_args()

    for seed in args.random_seeds:
        print(f"\n=== Running with seed {seed} ===")
        args.current_seed = seed
        main(args)
