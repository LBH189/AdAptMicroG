# AdAptMicroG

AdAptMicroG uses a graph attention network (GAT) and domain adaptation to classify host disease status from gut microbiome abundance profiles. It also reports feature importance and estimated feature contributions, and constructs a sample-neighbour graph for exploring relationships among samples.

The classifier is binary: the selected disease is the case class and `healthy` is the control class. It supports two workflows:

- **Prediction/evaluation mode** (`-t 0`, the default): train on rows marked `train`, then predict rows marked `test`.
- **Training mode** (`-t 1`): run k-fold cross-validation over the training samples. Do not include test rows in this mode.

> **Platform note:** the supplied environment is pinned to legacy Python/R packages and includes Linux-specific Conda dependencies. The pipeline invokes `Rscript` and shell utilities such as `cp`, `mv`, and `rm`; Linux or WSL is recommended. Native Windows execution is not covered by the supplied environment.

## Contents

- [Requirements and installation](#requirements-and-installation)
- [Configuration in `parameters.yaml`](#configuration-in-parametersyaml)
- [Quick start](#quick-start)
- [Input format](#input-format)
- [Command-line options](#command-line-options)
- [Output files](#output-files)
- [Datasets and reproducibility](#datasets-and-reproducibility)
- [Troubleshooting and interpretation](#troubleshooting-and-interpretation)

## Requirements and installation

The environment file specifies **Python 3.7.3** and **R 3.6.1**, along with the Python and R packages used by the pipeline. Conda is required to create the supplied environment.

From the project root:

```bash
conda env create -f environment_clean.yaml
conda activate adaptmicrog
```

Check that the program starts and inspect its options:

```bash
python AdAptMicroG.py -h
```

The R normalization step reads `parameters.yaml` and calls `norm_features.R`. Run the program from the project root so these relative paths, `allmeta.tsv`, and the input files can be found.

## Configuration in `parameters.yaml`

The selected configuration block comes from a broader SIAMCAT/meta-analysis workflow and is not all active in the current AdAptMicroG command-line pipeline. The main program calls `norm_features.R` to normalize abundance data; it does not use the study lists below to choose samples or split the input. Do not confuse the SIAMCAT settings here with AdAptMicroG's own CLI options such as `-c` (GAT cross-validation folds), `-f` (top features for contribution analysis), or `-d` (case label).

The study lists below are retained in `parameters.yaml` from the project's original configuration. The current AdAptMicroG pipeline does not use them to select or split samples: study/site labels come from the input CSV's `study` column, while its `class` column determines the split.

| Configuration key | Retained values |
| --- | --- |
| `ref.studies` | `FR-CRC`, `AT-CRC`, `CN-CRC`, `US-CRC`, `DE-CRC` |
| `ext.crc.studies` | `IT-CRC`, `IT-CRC-2`, `JP-CRC` |
| `ext.studies` | `CN-T2D`, `SE-T2D`, `DE-PD`, `US-HMP`, `DN-metaHIT` |

### `model.building`

| Key | Current value | Effect in this repository |
| --- | ---: | --- |
| `ab.cutoff` | `1e-03` | Declared in the YAML, but not applied by the current `norm_features.R` preprocessing path. |
| `norm.method` | `log.std` | Passed to SIAMCAT's `normalize.features()` during preprocessing. |
| `log.n0` | `1e-05` | Pseudocount parameter supplied for `species`/`genus` normalization tags. |
| `log.n0.func` | `1e-08` | Pseudocount parameter supplied for other normalization tags. |
| `sd.min.q` | `0.1` | Passed through to SIAMCAT as a normalization parameter. |
| `n.p` | `2` | Passed through to SIAMCAT as a normalization parameter. |
| `norm.margin` | `1` | Passed through to SIAMCAT as a normalization parameter. |
| `num.folds` | `10` | Used by the auxiliary `feature_select_model*.R` scripts for SIAMCAT data splits; does not set AdAptMicroG's GAT fold count. |
| `num.resample` | `10` | Used by the auxiliary SIAMCAT feature-selection scripts for resampling; not the GAT training loop's fold count. |
| `stratify` | `TRUE` | Present in the YAML but not read by the current preprocessing or auxiliary feature-selection scripts. |
| `ml.method` | `lasso` | Read into an R variable, but the auxiliary scripts pass `method = 'lasso'` directly to `train.model()`; changing this YAML value alone does not change the method. |
| `modsel.crit` | `pr` | Used by the auxiliary SIAMCAT feature-selection scripts; not used by the main GAT classifier. |
| `min.nonzero.coeff` | `1` | Used by the auxiliary SIAMCAT feature-selection scripts; not used by the main GAT classifier. |
| `feature.selection.type` | `AUC` | Used to configure feature selection in the auxiliary scripts. |
| `feature.selection.cutoff` | `800` | Read as the auxiliary feature-selection threshold for the single-study configuration. The auxiliary scripts set the LOSO threshold separately to `3200`. The main `-f` option is independent. |

For the main CLI workflow, `norm.method`, `log.n0`/`log.n0.func`, `sd.min.q`, `n.p`, and `norm.margin` affect SIAMCAT normalization. The other settings belong to auxiliary R feature-selection/model-evaluation scripts and should not be interpreted as switches for the GAT/domain-adaptation model. Changes to normalization settings can change model inputs and therefore downstream predictions.

## Quick start

### Cross-validation on a labelled dataset

Use `-t 1` when the CSV contains only samples assigned to the training set:

```bash
python AdAptMicroG.py \
  -i Input_files/CRC_10fold.csv \
  -t 1 \
  -d CRC \
  -o AdAptMicroG_res/CRC_10fold
```

This runs 10-fold cross-validation by default. To choose a different number of folds or set a seed:

```bash
python AdAptMicroG.py -i Input_files/CRC_10fold.csv -t 1 -d CRC \
  -c 5 -s 10 -o AdAptMicroG_res/CRC_5fold
```

### Train on one study and evaluate on held-out study samples

Use the default prediction/evaluation mode (`-t 0`) when the input contains both `train` and `test` rows:

```bash
python AdAptMicroG.py \
  -i Input_files/CRC_FRA_LOSO.csv \
  -d CRC \
  -s 10 \
  -o AdAptMicroG_res/CRC_FRA_LOSO
```

The `-d` value must match the case label in the `disease` column exactly. For example, use `-d IBD` for samples labelled `IBD`. The held-out examples included in this repository have known labels and can therefore be used to evaluate predictions.

## Input format

Provide a comma-separated CSV with a header. The first four columns are required in this exact order; every remaining column is one microbial feature:

| Column | Meaning | Example |
| --- | --- | --- |
| `subject_id` | Unique sample identifier | `S101_a_WGS` |
| `class` | Dataset partition; exactly `train` or `test` | `train` |
| `disease` | `healthy` for controls, or the case label specified by `-d` | `CRC` |
| `study` | Study/site identifier used as sample metadata | `FRA` |
| Remaining columns | Numeric microbial abundance features; one feature per column | `Bacteroides_vulgatus` |

Example:

```csv
subject_id,class,disease,study,Bacteroides_vulgatus,Faecalibacterium_prausnitzii
S1,train,healthy,StudyA,0.12,0.08
S2,train,CRC,StudyA,0.03,0.01
S3,test,healthy,StudyB,0.09,0.11
S4,test,CRC,StudyB,0.02,0.03
```

Input requirements:

1. Use unique, non-empty sample IDs and unique feature-column names.
2. The `disease` values for labelled samples must be `healthy` or the exact case label passed to `-d`. This is a binary classifier; do not mix multiple case labels in one run.
3. For prediction/evaluation mode, provide training samples from both classes and mark the held-out samples `test`. For cross-validation mode, provide labelled samples marked `train`; the program creates the folds.
4. Feature values must be numeric abundance measurements. Keep the same feature columns for all samples, and avoid blank or non-numeric entries.
5. `study` should be populated with a meaningful study/site value; it is used as metadata by the preprocessing pipeline.
6. Do not assume that replacing an unknown class with the literal `Unknown` makes it a supported, label-free inference input. The preprocessing code has special handling for unknown labels and may infer or substitute labels. Use labelled held-out samples for documented evaluation, and verify the current implementation before using unlabeled deployment data.

The preprocessing code reads the first four columns by position, so keep their order as shown. Their header names are descriptive, but additional metadata columns must not be inserted before the abundance columns.

## Command-line options

Run `python AdAptMicroG.py -h` for the CLI help. The parser does not define argparse defaults for most optional arguments, but the program assigns the defaults shown below when an option is omitted.

| Option | Default | Description |
| --- | ---: | --- |
| `-i`, `--input_file` | Required in practice | Path to the input CSV file. |
| `-t`, `--train_mode` | `0` | `0`: use `train` rows and predict/evaluate `test` rows. `1`: run k-fold cross-validation on training data only. |
| `-d`, `--disease` | Required in practice | Case-class label as written in the CSV `disease` column. |
| `-o`, `--outdir` | `AdAptMicroG_res` | Output directory. Main results are written under `<outdir>/Res_File/`. |
| `-c`, `--cvfold` | `10` | Number of folds for cross-validation. |
| `-s`, `--randomseed` | `0` | Random seed value; provide an explicit value to make a run repeatable. |
| `-k`, `--kneighbor` | `5` | Number of neighbours used to construct the k-nearest-neighbour graph. |
| `-b`, `--batchsize` | `64` | Batch size used during training. |
| `-a`, `--domain_adapt` | `1` | `1`: use domain adaptation; `0`: use the non-domain-adaptation MLP path. |
| `-r`, `--run_fi` | `1` | `1`: calculate feature importance and contribution outputs; `0`: skip these calculations. |
| `-f`, `--feature_num` | `10` | Number of top features to examine for contribution analysis. |
| `-e`, `--apply_node` | `0` | Request node-importance analysis in training mode. This can be computationally expensive; the regular prediction path does not run this analysis. |
| `-n`, `--node_num` | `20` | Number of nodes requested by node-importance analysis. |

Boolean-like options are passed as numeric values (`0` or `1`), for example `-r 0`. Because the CLI declares these values as strings and converts them internally, pass integers rather than words such as `true` or `false`.

## Output files

For an output directory named `AdAptMicroG_res`, the main result files are created in:

```text
AdAptMicroG_res/
├── Res_File/
│   ├── final_predict_metrics.txt
│   ├── sample_prob.txt
│   ├── sample_kneighbors_all.txt
│   ├── feature_importance.txt
│   ├── feature_local_importance.txt
│   └── driver_sp_change.txt
├── Preprocess_data/
├── Feature_File/
└── Graph_File/
```

Some files are conditional or may have fold-specific names:

| File | Description |
| --- | --- |
| `final_predict_metrics.txt` | Summary of the selected run/fold, including reported train, validation, and/or test metrics. Which metrics appear depends on the workflow and available labels. |
| `sample_prob.txt` | Per-sample class probabilities, predicted class, and (when available) observed class. In cross-validation mode, predictions can be aggregated across folds. |
| `sample_kneighbors_all.txt` | Neighbours of each sample in the constructed sample graph. |
| `feature_importance.txt` | Features ranked by the model's feature-importance calculation. Produced when feature-importance calculation is enabled. |
| `feature_local_importance.txt` | Local feature-importance results associated with the samples/model calculations. |
| `driver_sp_change.txt` | Estimated directions and magnitudes of feature contributions, including increases/decreases associated with disease or health. The number of features examined is controlled by `-f`. |
| `node_importance_single_fold*.txt`, `node_importance_combination_fold*.txt` | Fold-specific node-importance outputs when node analysis is enabled in training mode. |

Treat importance and contribution scores as model-derived evidence for interpretation, not as proof of biological causality.

## Datasets and reproducibility

Example CSVs are provided in [`Input_files/`](Input_files). They include CRC, IBD, T2D, cirrhosis, ASD, and MHE datasets, with both cross-validation and leave-one-study-out (LOSO) examples. For example:

```bash
python AdAptMicroG.py -i Input_files/T2D_10fold.csv -t 1 -d T2D \
  -s 10 -o AdAptMicroG_res/T2D_10fold

python AdAptMicroG.py -i Input_files/IBD_UK_LOSO.csv -d IBD -s 10 \
  -o AdAptMicroG_res/IBD_UK_LOSO
```

The repository also provides [`run_AdAptMicroG_demo.sh`](run_AdAptMicroG_demo.sh). Its current contents include a reference to `Input_files/Obt_10fold.csv`, which is not present in the checked-in `Input_files/` directory; therefore, the script may stop at that command unless the missing dataset is supplied or the command is removed. For an individual run, use the matching command above or adapt the script to the datasets available locally.

An explicit `-s` seed supports reproducibility, but exact results may still depend on the software environment and hardware. Keep the input data, environment, options, and output directory recorded alongside results.

## Troubleshooting and interpretation

- **Input file not found:** run commands from the project root and check the relative path passed to `-i`.
- **R normalization fails:** ensure the Conda environment is active and `Rscript` is available; run from the project root so `norm_features.R` and `parameters.yaml` are found.
- **Unexpected class labels or metrics:** check that the `-d` label matches the CSV exactly, that `healthy` is spelled consistently, and that both classes are represented in the training data.
- **No feature outputs:** check that `-r` is omitted or set to `1`; `-r 0` skips feature importance and contribution calculations.
- **Long runtime:** feature contribution analysis is enabled by default. Use `-r 0` for a prediction run without those analyses; node-importance analysis can also be expensive when enabled.
