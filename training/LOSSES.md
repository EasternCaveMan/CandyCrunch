# Choosing a classification loss

## Supervised contrastive loss

Use `--pretraining True` to train the model representations with the
supervised contrastive loss from Khosla et al., Eq. 2, then fine-tune the same
model with the selected classification loss. For example, from `training/`:

```bash
python training_script.py --dataset OP20260908 --split GShS --model CNN \
  --pretraining True --pretraining-epochs 10 \
  --loss_function cross_entropy --supcon-temperature 0.07
```

The SupCon training loader uses a class-aware sampler. By default it selects
`64` classes and `4` distinct original examples per class, then emits two
independently augmented views of each selected example. The effective forward
batch is therefore `64 * 4 * 2 = 512` embeddings, while the sampler covers
`256` original examples per batch. Tune this with
`--supcon-classes-per-batch`, `--supcon-examples-per-class`, and
`--supcon-views-per-example`.

For each anchor, every other view with the same glycan label is a positive and
every view with a different label is a negative. The anchor itself is excluded.
The implementation in `candycrunch/losses.py` normalizes the existing
`rep=True` model embeddings, subtracts the row maximum before the denominator,
and gathers embeddings across initialized `torch.distributed` workers. The
project's existing `DataParallel` path works because PyTorch gathers the model
outputs before the loss is evaluated.

CNN SupCon views use the existing spectrum and RT jitter. Transformer SupCon
views use RT jitter through the shared memmap dataset path. No model
architecture, inference interface, or checkpoint weight layout changes are
required. The pretraining checkpoint is saved first, then reloaded before
fine-tuning starts. When `--pretraining False`, SupCon is not used and the
script trains directly with `--loss_function`.

Use `--combine-loss True` instead of `--pretraining True` to apply SupCon during
fine-tuning itself. This optimizes
`loss_function + contrastive_loss_weight * SupConLoss`, using
`--supcon-temperature` for the SupCon temperature:

```bash
python training_script.py --dataset OP20260908 --split GShS --model CNN \
  --loss_function cross_entropy --combine-loss True \
  --contrastive-loss-weight 0.1 --supcon-temperature 0.07
```

`--pretraining True` and `--combine-loss True` are mutually exclusive: the first
uses SupCon before classification fine-tuning, and the second uses SupCon inside
the fine-tuning objective.

## Ambiguous labels (candidate sets)

Many classes leave linkages, monosaccharides or substituent positions open
(`Gal(b1-3/4)GlcNAc`, `Hex`, `GalOS`, floating `{Fuc(a1-?)}`). With
`--candidate-sets` (the default), the training script marks for every class the
classes that are the same glycan or a more specific version of it
(`compare_glycans(..., subsumes=True)` from glycowork, within each composition).
Every loss then treats an ambiguous label as satisfied by probability on any of
them: `CandidateSetLoss` (used by `custom_loss`, `PolyCrEnr`, and
`cross_entropy`) scores the summed probability of the candidates, `focal_loss`
and `xyz_loss` fold the candidates into the target logit before their usual
computation, the structure-distance term of `custom_loss` uses the closest
candidate, and SupCon treats such pairs as neither positives nor negatives.
Unambiguous labels keep a single target, so the model is still pushed towards
the most specific class a spectrum supports, and ambiguous classes remain
ordinary outputs. Accuracy, top-k, F1 and MCC count a candidate as a hit, so
they are not comparable to runs without candidate sets. These runs carry a
`_CS` filename tag. `--no-candidate-sets` restores single-class targets and is
identical to the previous behaviour. Inference is unchanged.

## Classification loss options

Add one of these options to an existing `training_script.py` command:

```bash
--loss_function custom_loss
--loss_function xyz_loss
--loss_function cross_entropy
--loss_function focal_loss --focal_gamma 2.0
```

`custom_loss` is the default and preserves the existing Poly1 classification
loss and composition/structure distance penalties.

`cross_entropy` uses standard `torch.nn.CrossEntropyLoss` over all glycan
classes, with mean reduction and no label smoothing, Poly1 term, or distance
penalties. The training script skips the structure/composition distance
matrices for this option. Composition vectors are still prepared as model
inputs. Both SAM passes use cross-entropy, and configured MoE auxiliary losses
are still added to the training objective.

`focal_loss` uses `torch_focalloss.MultiClassFocalLoss` from the
[`pytorch-focalloss` package](https://pypi.org/project/pytorch-focalloss/), which
is included in the project dependencies. The adapter in `candycrunch/losses.py`
validates inputs and reduces per-sample losses over the batch. It also handles
fully confident correct predictions when `0 < gamma < 1` to avoid undefined
gradients in the library's focusing term. The library computes multiclass
softmax focal loss over all glycan classes:
`mean(-(1 - p_target)**gamma * log(p_target))`, where `p_target` is the
probability of the correct class. It follows the focusing term from
[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
`--focal_gamma` defaults to `2.0`; `0` gives standard cross-entropy. Classes
have equal weight. This option uses the existing single glycan classification
task, with raw `[batch, classes]` logits and one integer target per spectrum.
For a model with multiple classification tasks, each task would need its own
logits and targets and the per-task losses would need to be combined.
Both SAM passes and validation use the selected focal loss. Structure and
composition distance matrices are skipped; configured MoE auxiliary losses
are still added during training. Checkpoints record `focal_gamma` in
`training_args` and use the `FOCALLOSS` filename tag.

`xyz_loss` restricts each sample to every glycan class with exactly the same
composition as its target, including compatible classes absent from the batch.
It masks incompatible logits before applying standard
`torch.nn.CrossEntropyLoss`, with mean reduction and no label smoothing,
Poly1 term, or distance penalties. The training script skips the
structure/composition distance matrices for this option. A composition
containing only one class has zero classification loss and gradients.

The implementation is in `candycrunch/losses.py`. Composition groups use the
class-aligned count vectors already prepared by the training script, in
`comp_vector_order`, rather than depending on the ordering of `glycans.pkl`.
Both SAM passes use the selected loss. Training and validation metrics use the
same composition restriction. Classifier and Transformer MoE routing and their
auxiliary loss weights remain separate from this choice.

Runs using `xyz_loss` have an `XYZLOSS` filename tag; `cross_entropy` runs use
`CELOSS`, and `focal_loss` runs use `FOCALLOSS`. These tags keep their output
files separate from `custom_loss` runs, and the test leaderboard recognizes
these tags. All checkpoints record the
loss choice; `xyz_loss` checkpoints also record the aligned composition
vectors as metadata. Inference uses the same procedure for all loss choices:
optional temperature scaling, softmax over all classes, and top-k selection.
The loss choice and saved class composition vectors do not enable masking
during inference. Direct calls to the model still return raw logits.

For every loss choice, `wrap_inference` proposes compositions from precursor
mass and supplies them as model inputs. After combining augmented predictions,
it filters candidates by glycan class, confidence, and composition. This
filtering does not renormalize the probabilities. The existing mass and fragment
annotation fallback can recover a rejected candidate and update the proposed
composition.

For `xyz_loss`, training and validation use the true target composition to
restrict the loss and metrics. These metrics therefore measure structure
prediction given that composition, whereas inference ranks all output classes
before wrapper filtering. Inference confidence for a matching structure is not
forced to one when it is the only class with the proposed composition.

For Transformer runs affected by the optimized attention CUDA issue, keep
`--disable_mha_fastpath` in the training command as well.
