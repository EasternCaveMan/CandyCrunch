"""Contrastive pretraining of CandyCrunch's spectrum encoder on unlabelled pooled spectra (data_processing/Final_Results_cc/Prefull*.pkl).

Both modes train the encoder through a projection head that fine-tuning drops (ProjectionHeadModel) with
candycrunch.losses.PretrainingContrastiveLoss on batches of augmented views, and pick the checkpoint by IsomerProbe, a nearest-neighbour
isomer accuracy on the labelled spectra, as a low contrastive loss alone does not show a useful encoder (raw inputs already reach one).
Batches come from GroupedViewBatchSampler, several spectra per group, so that spectra which may be one molecule, or its isomers, meet:
supervised:   groups and pair keys are compositions (the composition column, or each spectrum's clearly best candidate composition).
unsupervised: groups and pair keys are precursor m/z, which needs no annotation; without an m/z column, ViewBatchSampler draws at random
              and the views of a spectrum are its only positives.
Within a key, the loss makes spectra of one setting group within same_peak_rt positives and spectra of one run isomer_rt apart negatives:
in labelled spectra of five projects, the former were one glycan in 96.5% of cross-run pairs (all within-run pairs), the latter different
glycans in 77% of within-run pairs. Views change both model input channels (ViewAugmentation), as two scans of one precursor do.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from candycrunch.inference import transform_rt


NUM_BINS = 2048
MIN_MZ = 39.714
# prediction.bin_intensities: bin k holds MIN_MZ + k * BIN_WIDTH < m/z <= MIN_MZ + (k + 1) * BIN_WIDTH, its remainder the m/z above the lower edge
BIN_WIDTH = (3000.0 - MIN_MZ) / (NUM_BINS - 1)
MODE_CODES = {"positive": 0, "negative": 1}
LC_CODES = {"PGC": 0, "C18": 1}
MODIFICATION_CODES = {"reduced": 0, "permethylated": 1}
TRAP_CODES = {"linear": 0, "orbitrap": 1, "amazon": 2}
# Long names such as Neu5Gc_a2-31 mean one a2-3-linked Neu5Gc; the general composition parser otherwise reads -31 as a mass delta
LINKAGE_CODES = (("Neu5Ac_a2-6", "E"), ("Neu5Ac_a2-3", "L"), ("Neu5Gc_a2-6", "Ge"), ("Neu5Gc_a2-3", "Gl"), ("Kdn_a2-6", "KdnE"),
                 ("Kdn_a2-3", "KdnL"))


def normalize_linkage_names(value):
    """Translate resolved sialic-acid names in a composition string to glycowork's composition codes."""
    if isinstance(value, str):
        for name, code in LINKAGE_CODES:
            value = value.replace(name, code)
    return value


def composition_ids(values):
    """An integer id per composition (a string in any glycowork notation, or a mapping of residue counts) and the canonical label of each id."""
    from glycowork.motif.processing import canonicalize_composition
    cache, keys = {}, []
    for index, value in enumerate(values):
        if isinstance(value, str) and value in cache:
            keys.append(cache[value])
            continue
        try:
            composition = value if isinstance(value, dict) else canonicalize_composition(normalize_linkage_names(value))
            key = tuple(sorted((name, int(count)) for name, count in composition.items() if count > 0))
            if not key or any(not np.isfinite(count) or count < 0 or int(count) != count for count in composition.values()):
                raise ValueError("Composition counts must be non-negative integers.")
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Invalid composition at pretraining row {index}: {value!r}") from error
        if isinstance(value, str):
            cache[value] = key
        keys.append(key)
    classes = sorted(set(keys))
    lookup = {key: index for index, key in enumerate(classes)}
    labels = [canonicalize_composition(dict(key), as_string = True) for key in classes]
    return np.fromiter((lookup[key] for key in keys), dtype = np.int64, count = len(keys)), labels


def best_compositions(candidates, scores, min_score_gap = 1.0):
    """Each spectrum's best-scoring candidate composition, or None where a runner-up scores within min_score_gap of it (or none is listed)."""
    best = []
    for options, values in zip(candidates, scores):
        values = np.asarray(values, dtype = float).reshape(-1)
        if len(options) == 0 or len(values) != len(options):
            best.append(None)
            continue
        order = np.argsort(-values, kind = "stable")
        ambiguous = len(options) > 1 and values[order[0]] - values[order[1]] < min_score_gap
        best.append(None if ambiguous else options[order[0]])
    return best


class PretrainingPool:
    """Sparse spectra and per-spectrum settings as flat arrays: spectrum i's (bin, intensity, m/z remainder) peaks are entries
    offsets[i]:offsets[i + 1], so a view costs one slice and DataLoader workers receive compact arrays rather than a DataFrame.

    run_ids (GlycoPost_ID and filename) and setting_ids (project, LC, ion mode, derivatization) are None when the source does not name runs,
    precursor_mz when it has no m/z column; composition_ids is None without composition labels and -1 for spectra without a confident one."""

    def __init__(self, bins, intensities, remainders, offsets, glycan_type, rt, mode, lc, modification, trap, run_ids = None,
                 run_names = None, setting_ids = None, precursor_mz = None, composition_ids = None, composition_labels = None):
        self.bins = np.asarray(bins, dtype = np.int16)
        self.intensities = np.asarray(intensities, dtype = np.float32)
        self.remainders = np.asarray(remainders, dtype = np.float32)
        self.offsets = np.asarray(offsets, dtype = np.int64)
        count = len(self.offsets) - 1
        if count < 0 or self.offsets[0] != 0 or self.offsets[-1] != len(self.bins) or np.any(np.diff(self.offsets) < 0):
            raise ValueError("Pretraining peak offsets do not match the peak arrays.")
        if len(self.bins) and (self.bins.min() < 0 or self.bins.max() >= NUM_BINS):
            raise ValueError(f"Pretraining peaks must lie in bins 0 to {NUM_BINS - 1}.")
        settings = {"glycan_type": glycan_type, "rt": rt, "mode": mode, "lc": lc, "modification": modification, "trap": trap}
        for name, values in settings.items():
            values = np.asarray(values, dtype = np.float32 if name == "rt" else np.int64)
            if len(values) != count:
                raise ValueError(f"Pretraining {name} has {len(values)} entries for {count} spectra.")
            setattr(self, name, values)
        self.run_ids = None if run_ids is None else np.asarray(run_ids, dtype = np.int64)
        self.run_names = run_names
        self.setting_ids = None if setting_ids is None else np.asarray(setting_ids, dtype = np.int64)
        self.precursor_mz = None if precursor_mz is None else np.asarray(precursor_mz, dtype = np.float32)
        self.composition_ids = None if composition_ids is None else np.asarray(composition_ids, dtype = np.int64)
        self.composition_labels = composition_labels

    def __len__(self):
        return len(self.offsets) - 1

    def peaks(self, row):
        start, end = self.offsets[row], self.offsets[row + 1]
        return self.bins[start:end], self.intensities[start:end], self.remainders[start:end]

    def rows_in_runs(self, runs):
        """Boolean mask of the spectra whose (GlycoPost_ID, filename) is in runs."""
        if self.run_ids is None:
            return np.zeros(len(self), dtype = bool)
        wanted = {(str(project), str(filename)) for project, filename in runs}
        hit = np.array([name in wanted for name in self.run_names], dtype = bool)
        return hit[self.run_ids] if len(hit) else np.zeros(len(self), dtype = bool)


def _pool_from_frame(frame, supervised = False, min_score_gap = 1.0):
    required = {"model_input", "RT2", "mode", "lc_type", "modification", "trap"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Pretraining inputs are missing columns: {sorted(missing)}")
    frame = frame.reset_index(drop = True)
    spectra = [np.asarray(spectrum, dtype = np.float32).reshape(3, -1) for spectrum in frame["model_input"]]
    offsets = np.concatenate([[0], np.cumsum([spectrum.shape[1] for spectrum in spectra])])
    peaks = np.concatenate(spectra, axis = 1) if spectra else np.zeros((3, 0), dtype = np.float32)
    if "glycan_type" in frame:
        types = [[value] for value in frame["glycan_type"]]
    else:
        types = frame["glycan_types"] if "glycan_types" in frame else [[] for _ in range(len(frame))]
    glycan_type = []
    for value in types:
        value = np.asarray(value).reshape(-1)
        glycan_type.append(int(value[0]) if len(value) and np.all(value == value[0]) else 3)
    mode = frame["mode"].map(lambda value: MODE_CODES.get(value, value))
    unknown = ~mode.isin([0, 1])
    if unknown.any():
        index = int(np.flatnonzero(unknown.to_numpy())[0])
        raise ValueError(f"Unknown ion mode at pretraining row {index}: {frame['mode'].iloc[index]!r}")
    pool = {"bins": peaks[0], "intensities": peaks[1], "remainders": peaks[2], "offsets": offsets, "glycan_type": glycan_type,
            "rt": frame["RT2"].to_numpy(dtype = np.float32), "mode": mode.to_numpy(dtype = np.int64),
            "lc": frame["lc_type"].map(lambda value: LC_CODES.get(value, 2)).to_numpy(),
            "modification": frame["modification"].map(lambda value: MODIFICATION_CODES.get(value, 2)).to_numpy(),
            "trap": frame["trap"].map(lambda value: TRAP_CODES.get(value, 3)).to_numpy()}
    if {"GlycoPost_ID", "filename"}.issubset(frame.columns):
        project, filename = frame["GlycoPost_ID"].astype(str), frame["filename"].astype(str)
        run_ids, run_names = pd.factorize(pd.Series(list(zip(project, filename))))
        settings = pd.Series(list(zip(project, frame["lc_type"].astype(str), frame["mode"].astype(str), frame["modification"].astype(str))))
        pool.update(run_ids = run_ids, run_names = list(run_names), setting_ids = pd.factorize(settings)[0])
    if "m/z" in frame:
        pool["precursor_mz"] = frame["m/z"].to_numpy(dtype = np.float32)
    if supervised:
        if "run_ids" not in pool:
            raise ValueError("Supervised pretraining needs GlycoPost_ID and filename columns to pair spectra within runs.")
        if "composition" in frame:
            compositions = list(frame["composition"])
        elif {"compositions", "annotation_scores"}.issubset(frame.columns):
            compositions = best_compositions(frame["compositions"], frame["annotation_scores"], min_score_gap = min_score_gap)
        else:
            raise ValueError("Supervised pretraining requires a composition column, or compositions with annotation_scores.")
        known = np.array([value is not None and not (isinstance(value, float) and np.isnan(value)) for value in compositions], dtype = bool)
        ids = np.full(len(frame), -1, dtype = np.int64)
        labels = []
        if known.any():
            ids[known], labels = composition_ids([value for value, ok in zip(compositions, known) if ok])
        pool.update(composition_ids = ids, composition_labels = labels)
    return PretrainingPool(**pool)


def _pool_from_store(directory, supervised = False):
    if supervised:
        raise ValueError("Supervised pretraining needs a pretraining DataFrame with runs and compositions, not a spectrum store.")
    with (directory / "manifest.json").open() as file:
        manifest = json.load(file)
    if manifest.get("num_bins", NUM_BINS) != NUM_BINS:
        raise ValueError(f"Spectrum store {directory} has {manifest['num_bins']} bins; the models take {NUM_BINS}.")
    arrays = {name: np.load(directory / f"{name}.npy") for name in ["metadata", "bin_offsets", "bin_index", "bin_intensity", "bin_remainder"]}
    metadata = arrays["metadata"]
    return PretrainingPool(arrays["bin_index"], arrays["bin_intensity"], arrays["bin_remainder"] * manifest["remainder_scale"],
                           arrays["bin_offsets"], metadata[:, 0], metadata[:, 1], metadata[:, 2], metadata[:, 3], metadata[:, 4], metadata[:, 5])


def load_pretraining_pool(source, supervised = False, min_score_gap = 1.0):
    """Read a pretraining DataFrame pickle (Prefull*.pkl) or a spectrum store directory (build_training_pickles_2.py) into a PretrainingPool.

    Supervised pretraining takes the composition column, else the best of each spectrum's candidate compositions where no other candidate
    scores within min_score_gap of it (others get no composition and stay out of supervised batches)."""
    source = Path(source)
    if source.is_dir():
        return _pool_from_store(source, supervised = supervised)
    frame = pd.read_pickle(source)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"Pretraining source {source} is not a DataFrame pickle or a spectrum store.")
    return _pool_from_frame(frame, supervised = supervised, min_score_gap = min_score_gap)


def fine_tuning_validation_runs(dataset_dir, split):
    """(GlycoPost_ID, filename) of every run in the fine-tuning validation split, so pretraining can leave them out: runs_test_<split>.json, which
    build_training_pickles.py writes, else read once from test_second_<split>.pkl and cached as that file; None when neither exists."""
    dataset_dir = Path(dataset_dir)
    cache = dataset_dir / f"runs_test_{split}.json"
    if cache.is_file():
        return {tuple(run) for run in json.loads(cache.read_text())}
    frame_path = dataset_dir / f"test_second_{split}.pkl"
    if not frame_path.is_file():
        return None
    frame = pd.read_pickle(frame_path)
    runs = sorted(set(zip(frame["GlycoPost_ID"].astype(str), frame["filename"].astype(str))))
    try:
        cache.write_text(json.dumps([list(run) for run in runs], indent = 1))
    except OSError:
        pass
    return set(runs)


def split_pretraining_runs(pool, rows, validation_fraction = 0.15, seed = 0):
    """Training and validation rows, holding out whole runs (pooled spectra of one run share peaks and settings, so a row split leaks); by row
    where the pool names no runs or all rows come from one run."""
    if not 0 < validation_fraction < 1:
        raise ValueError("pretraining validation fraction must be between 0 and 1.")
    rows = np.asarray(rows, dtype = np.int64)
    if len(rows) < 4:
        raise ValueError("Pretraining requires at least four spectra for train/validation.")
    rng = np.random.default_rng(seed)
    if pool.run_ids is not None and len(np.unique(pool.run_ids[rows])) > 1:
        runs, counts = np.unique(pool.run_ids[rows], return_counts = True)
        order = rng.permutation(len(runs))
        taken = np.cumsum(counts[order])
        held = runs[order[:int(np.searchsorted(taken, validation_fraction * len(rows))) + 1]]
        validation = np.isin(pool.run_ids[rows], held)
        if 2 <= validation.sum() <= len(rows) - 2:
            return rows[~validation], rows[validation]
    val_count = max(2, min(len(rows) - 2, round(len(rows) * validation_fraction)))
    permuted = rng.permutation(rows)
    return np.sort(permuted[val_count:]), np.sort(permuted[:val_count])


class ViewAugmentation:
    """A random view of a sparse spectrum that changes both input channels, the weaker peaks more (counting noise), as scans of one precursor do.

    With r a peak's intensity relative to the base peak, each peak is dropped with probability peak_dropout * (1 - sqrt(r)), so the base peak
    stays; its intensity is multiplied by log-normal noise of sd intensity_jitter / sqrt(r) (r floored at 0.01, sd capped at 1.5); its m/z
    remainder moves by a spectrum-wide normal offset of sd mz_jitter plus a per-peak one of sd mz_jitter / sqrt(r) (capped at half a bin),
    reflected at the bin edges (pooled remainders are means over scans, which do not place a peak exactly enough to move it between bins); and
    Poisson(noise_peaks * peak count) noise peaks of up to noise_intensity of the base peak, with random remainders, join empty bins within the
    spectrum's m/z range. Intensities are normalized again. The defaults reproduce how two half-pools of a labelled spectrum's member scans
    differ (median occupied-bin Jaccard 0.41, intensity cosine 0.94, remainder difference sd 0.24 on shared bins, in 1,287 spectra of five
    projects), where transform_mz's views keep the remainders and a cosine of 0.98.
    """

    def __init__(self, peak_dropout = 0.3, intensity_jitter = 0.1, mz_jitter = 0.02, noise_peaks = 0.4, noise_intensity = 0.05):
        if not 0 <= peak_dropout < 1:
            raise ValueError("peak_dropout must be in [0, 1).")
        if min(intensity_jitter, mz_jitter, noise_peaks, noise_intensity) < 0:
            raise ValueError("Augmentation strengths must be non-negative.")
        self.peak_dropout = peak_dropout
        self.intensity_jitter = intensity_jitter
        self.mz_jitter = mz_jitter
        self.noise_peaks = noise_peaks
        self.noise_intensity = noise_intensity

    def settings(self):
        return {"peak_dropout": self.peak_dropout, "intensity_jitter": self.intensity_jitter, "mz_jitter": self.mz_jitter,
                "noise_peaks": self.noise_peaks, "noise_intensity": self.noise_intensity}

    def __call__(self, bins, intensities, remainders):
        if not len(bins) or intensities.max() <= 0:
            return bins, intensities, remainders
        relative = np.clip(intensities / intensities.max(), 0, 1)
        keep = np.random.random(len(bins)) >= self.peak_dropout * (1 - np.sqrt(relative))
        bins, intensities, remainders, relative = bins[keep], intensities[keep], remainders[keep], relative[keep]
        spread = 1 / np.sqrt(np.maximum(relative, 0.01))
        intensities = intensities * np.exp(np.random.normal(size = len(bins)) * np.minimum(self.intensity_jitter * spread, 1.5))
        shift = np.random.normal() * self.mz_jitter + np.random.normal(size = len(bins)) * np.minimum(self.mz_jitter * spread, BIN_WIDTH / 2)
        remainders = np.abs(remainders + shift)
        remainders = np.clip(np.where(remainders > BIN_WIDTH, 2 * BIN_WIDTH - remainders, remainders), 0, BIN_WIDTH)
        noise = np.random.poisson(self.noise_peaks * len(bins))
        if noise:
            empty = np.setdiff1d(np.arange(bins.min(), bins.max() + 1), bins)
            chosen = np.random.choice(empty, size = min(noise, len(empty)), replace = False) if len(empty) else empty
            bins = np.concatenate([bins, chosen.astype(bins.dtype)])
            intensities = np.concatenate([intensities, np.random.uniform(0, self.noise_intensity, len(chosen)) * intensities.max()])
            remainders = np.concatenate([remainders, np.random.uniform(0, BIN_WIDTH, len(chosen))])
        return bins, (intensities / intensities.sum()).astype(np.float32), remainders.astype(np.float32)


class PretrainingDataset(torch.utils.data.Dataset):
    """Views of pool rows in SimpleDataset's ten-field layout: the composition input is default_composition (unknown in pretraining) and the
    target the pool row, by which PretrainingContrastiveLoss looks up the pair labels. Without augmentation, a row's stored spectrum and RT."""

    def __init__(self, pool, rows, default_composition, augmentation = None):
        self.pool = pool
        self.rows = np.asarray(rows, dtype = np.int64)
        self.y = self.rows
        self.default_composition = torch.as_tensor(np.asarray(default_composition, dtype = np.float32))
        self.augmentation = augmentation

    def __len__(self):
        return len(self.rows)

    def group_ids(self, key = "composition", mz_tolerance = 0.5):
        """Per dataset position, the group GroupedViewBatchSampler draws by: setting group and composition (key 'composition'; -1 without
        one) or setting group and precursor m/z in mz_tolerance-wide bins (key 'precursor')."""
        pool = self.pool
        values = pool.composition_ids if key == "composition" else pool.precursor_mz
        if values is None or pool.setting_ids is None:
            raise ValueError(f"The pretraining pool has no {'compositions' if key == 'composition' else 'precursor m/z'} and runs to group by.")
        values = values[self.rows]
        if key != "composition":
            values = np.where(np.isfinite(values), np.floor(values / mz_tolerance), -1).astype(np.int64)
        groups = pd.factorize(pd.Series(list(zip(pool.setting_ids[self.rows], values))))[0]
        return np.where(values >= 0, groups, -1)

    def __getitem__(self, index):
        row = int(self.rows[index])
        bins, intensities, remainders = self.pool.peaks(row)
        rt = float(self.pool.rt[row])
        if self.augmentation is not None:
            bins, intensities, remainders = self.augmentation(bins, intensities, remainders)
            rt = transform_rt(rt)
        mz = np.zeros(NUM_BINS, dtype = np.float32)
        mz_r = np.zeros(NUM_BINS, dtype = np.float32)
        mz[bins.astype(np.intp)], mz_r[bins.astype(np.intp)] = intensities, remainders
        pool = self.pool
        return (torch.from_numpy(mz), torch.from_numpy(mz_r), self.default_composition, torch.LongTensor([pool.glycan_type[row]]),
                torch.FloatTensor([rt]), torch.LongTensor([pool.mode[row]]), torch.LongTensor([pool.lc[row]]),
                torch.LongTensor([pool.modification[row]]), torch.LongTensor([pool.trap[row]]), torch.LongTensor([row]))


class ViewBatchSampler(torch.utils.data.Sampler):
    """Random batches of examples_per_batch dataset positions, each repeated views_per_example times (each repeat gets its own augmentation)."""

    def __init__(self, num_samples, examples_per_batch = 128, views_per_example = 2, drop_last = True, seed = 0):
        if num_samples < 1 or examples_per_batch < 1:
            raise ValueError("num_samples and examples_per_batch must be positive.")
        if views_per_example < 2:
            raise ValueError("views_per_example must be at least 2.")
        self.num_samples = int(num_samples)
        self.examples_per_batch = int(examples_per_batch)
        self.views_per_example = int(views_per_example)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.batch_size = self.examples_per_batch * self.views_per_example
        self._num_batches = self.num_samples // self.examples_per_batch if self.drop_last else -(-self.num_samples // self.examples_per_batch)
        if self._num_batches < 1:
            raise ValueError("Contrastive sampler has fewer samples than examples_per_batch.")

    def __len__(self):
        return self._num_batches

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        order = np.random.default_rng(self.seed + self.epoch).permutation(self.num_samples)
        for start in range(0, self._num_batches * self.examples_per_batch, self.examples_per_batch):
            examples = order[start:start + self.examples_per_batch]
            yield [int(position) for position in examples for _ in range(self.views_per_example)]


class GroupedViewBatchSampler(torch.utils.data.Sampler):
    """Batches of examples_per_batch distinct dataset positions, each repeated views_per_example times, filled with up to examples_per_group
    positions of one group after another (groups in a new random order each pass), so a batch holds several spectra of each of its
    composition and setting groups. Groups of one spectrum and group id -1 are left out; an epoch has as many batches as the kept spectra fill."""

    def __init__(self, group_ids, examples_per_batch = 128, examples_per_group = 4, views_per_example = 2, seed = 0):
        if examples_per_group < 2:
            raise ValueError("examples_per_group must be at least 2.")
        if views_per_example < 2:
            raise ValueError("views_per_example must be at least 2.")
        group_ids = np.asarray(group_ids, dtype = np.int64)
        members = defaultdict(list)
        for position, group in enumerate(group_ids):
            if group >= 0:
                members[int(group)].append(position)
        self.groups = [np.asarray(positions) for _, positions in sorted(members.items()) if len(positions) >= 2]
        if not self.groups:
            raise ValueError("Supervised pretraining found no composition with two spectra in one setting group.")
        self.num_examples = sum(len(positions) for positions in self.groups)
        self.examples_per_batch = min(int(examples_per_batch), self.num_examples)
        if self.examples_per_batch < 2:
            raise ValueError("examples_per_batch must be at least 2.")
        self.examples_per_group = int(examples_per_group)
        self.views_per_example = int(views_per_example)
        self.seed = int(seed)
        self.epoch = 0
        self.batch_size = self.examples_per_batch * self.views_per_example
        self._num_batches = max(1, self.num_examples // self.examples_per_batch)

    def __len__(self):
        return self._num_batches

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order, cursor = rng.permutation(len(self.groups)), 0
        for _ in range(self._num_batches):
            batch, seen = [], set()
            while len(batch) < self.examples_per_batch:
                if cursor == len(order):
                    order, cursor = rng.permutation(len(self.groups)), 0
                positions = self.groups[order[cursor]]
                cursor += 1
                for position in rng.choice(positions, size = min(len(positions), self.examples_per_group), replace = False):
                    if int(position) not in seen and len(batch) < self.examples_per_batch:
                        seen.add(int(position))
                        batch.append(int(position))
            yield [position for position in batch for _ in range(self.views_per_example)]


def describe_pairs(loss, sampler, dataset, batches = 20):
    """Mean pairs per view over the sampler's first batches: positives besides the view's partner views, isomer negatives, pairs left out."""
    totals = defaultdict(float)
    views = 0
    for batch, _ in zip(sampler, range(batches)):
        types = loss.pair_types(torch.as_tensor(dataset.rows[batch]))
        views += len(batch)
        for name in ("same_peak", "isomer", "ignored"):
            totals[name] += float(types[name].sum())
    return {name: totals[name] / max(views, 1) for name in ("same_peak", "isomer", "ignored")}


class ProjectionHeadModel(torch.nn.Module):
    """A backbone and the MLP projection head the contrastive loss acts on (SimCLR / SupCon practice), so its invariances shape the head's
    output, not the representation the classifier is fine-tuned on; fine-tuning takes self.backbone alone. With rep = True the backbone's
    outputs come back with the projected embedding in place of the representation."""

    def __init__(self, backbone, embedding_dim, projection_dim = 128):
        super().__init__()
        self.backbone = backbone
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(embedding_dim, embedding_dim), torch.nn.ReLU(),
                                                   torch.nn.Linear(embedding_dim, projection_dim))
        # train_model and forward_training_batch read these off the model; the aux-loss getters only tell which outputs forward returns
        for name in ("classifier_moe", "use_transformer_ff", "encoder_type", "get_classifier_aux_loss", "get_transformer_aux_loss"):
            if hasattr(backbone, name):
                setattr(self, name, getattr(backbone, name))

    def forward(self, *inputs, rep = False, **kwargs):
        values = list(self.backbone(*inputs, rep = True, **kwargs))
        values[1] = self.projection_head(values[1])
        if not rep:
            values.pop(1)
        return tuple(values) if len(values) > 1 else values[0]


def embedding_dim(model, composition_dim):
    """Width of the representation model(..., rep = True) returns, from one forward pass of a small dummy batch."""
    was_training = model.training
    model.eval()
    spectrum = torch.zeros(2, 2, NUM_BINS)
    spectrum[:, 0, 100:110] = 0.1
    spectrum[:, 1, 100:110] = BIN_WIDTH / 2
    category = torch.zeros(2, 1, dtype = torch.long)
    with torch.no_grad():
        output = model(spectrum, torch.zeros(2, composition_dim), category, torch.zeros(2, 1), category, category, category, category, rep = True)
    model.train(was_training)
    return int(output[1].shape[1])


class IsomerProbe:
    """Checkpoint-selection score of contrastive pretraining: over labelled validation spectra whose composition has several classes among the
    reference (training) spectra, the share whose nearest reference spectrum of the same composition, by cosine in the encoder's representation
    (rep output, composition input zeroed as in pretraining), carries an accepted label (accepts[true, predicted]: the candidate-set mask)."""

    name = "isomer_knn_accuracy"

    def __init__(self, references, reference_labels, queries, query_labels, class_compositions, accepts, composition_dim, max_references = 20000,
                 max_queries = 5000, batch_size = 512, seed = 0):
        rng = np.random.default_rng(seed)
        reference_labels, query_labels = np.asarray(reference_labels, dtype = np.int64), np.asarray(query_labels, dtype = np.int64)
        class_compositions = np.asarray(class_compositions, dtype = np.int64)
        classes = defaultdict(set)
        for label in np.unique(reference_labels):
            classes[int(class_compositions[label])].add(int(label))
        isomeric = np.array(sorted(composition for composition, members in classes.items() if len(members) > 1), dtype = np.int64)
        reference_rows = np.flatnonzero(np.isin(class_compositions[reference_labels], isomeric))
        reference_rows = np.sort(rng.choice(reference_rows, size = min(len(reference_rows), max_references), replace = False))
        reference_compositions = class_compositions[reference_labels[reference_rows]]
        kept = defaultdict(set)
        for row, composition in zip(reference_rows, reference_compositions):
            kept[int(composition)].add(int(reference_labels[row]))
        isomeric = np.array(sorted(composition for composition, members in kept.items() if len(members) > 1), dtype = np.int64)
        query_rows = np.flatnonzero(np.isin(class_compositions[query_labels], isomeric))
        query_rows = np.sort(rng.choice(query_rows, size = min(len(query_rows), max_queries), replace = False))
        if not len(query_rows):
            raise ValueError("The isomer probe found no validation spectrum of a composition with several reference classes.")
        self.references, self.queries = references, queries
        self.reference_rows, self.query_rows = reference_rows, query_rows
        self.reference_labels, self.query_labels = reference_labels[reference_rows], query_labels[query_rows]
        self.accepts = np.asarray(accepts, dtype = bool)
        self.composition_dim = composition_dim
        self.batch_size = batch_size
        query_compositions = class_compositions[self.query_labels]
        self.by_composition = [(np.flatnonzero(reference_compositions == composition), np.flatnonzero(query_compositions == composition))
                               for composition in np.unique(query_compositions)]
        majority = 0
        for reference_positions, query_positions in self.by_composition:
            labels, counts = np.unique(self.reference_labels[reference_positions], return_counts = True)
            majority += self.accepts[self.query_labels[query_positions], labels[np.argmax(counts)]].sum()
        self.majority_accuracy = majority / len(query_rows)

    def __len__(self):
        return len(self.query_rows)

    def _embed(self, model, samples, rows, device):
        embeddings = []
        for start in range(0, len(rows), self.batch_size):
            batch = [samples[int(row)] for row in rows[start:start + self.batch_size]]
            spectra = torch.stack([torch.stack([torch.as_tensor(sample[0], dtype = torch.float32), torch.as_tensor(sample[1], dtype = torch.float32)])
                                   for sample in batch])
            category = lambda field: torch.as_tensor([[int(sample[field])] for sample in batch], dtype = torch.long)
            rt = torch.as_tensor([[float(sample[4])] for sample in batch], dtype = torch.float32)
            inputs = (spectra, torch.zeros(len(batch), self.composition_dim), category(3), rt, category(5), category(6), category(7), category(8))
            embeddings.append(model(*(value.to(device) for value in inputs), rep = True)[1])
        return torch.nn.functional.normalize(torch.cat(embeddings), dim = 1)

    def __call__(self, model):
        base = model.module if isinstance(model, torch.nn.DataParallel) else model
        base = getattr(base, "backbone", base)
        was_training = base.training
        base.eval()
        device = next(base.parameters()).device
        with torch.no_grad():
            references = self._embed(base, self.references, self.reference_rows, device)
            queries = self._embed(base, self.queries, self.query_rows, device)
            correct = 0
            for reference_positions, query_positions in self.by_composition:
                similarity = queries[torch.as_tensor(query_positions, device = device)] @ references[torch.as_tensor(reference_positions, device = device)].T
                nearest = reference_positions[similarity.argmax(dim = 1).cpu().numpy()]
                correct += self.accepts[self.query_labels[query_positions], self.reference_labels[nearest]].sum()
        base.train(was_training)
        return float(correct / len(self.query_rows))
