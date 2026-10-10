# Ribonucleic Acid Folding Oracle - RAFO
<p align="center">
  <img src="./assets/logo.png" alt="Demo" width="300">
</p>

RAFO is a deep learning model for predicting the 3D structure of RNA IFEs (Integrated Functional Elements) from sequence. This repository contains everything needed to prepare datasets, train the model, evaluate it, and run inference.

## Installation

RAFO runs inside an [Apptainer](https://github.com/apptainer/apptainer) container so that all dependencies (CUDA, OpenFold, RiNALMo, QRNAS, IPknot, ...) and environment settings are reproducible.

Build the image (`.sif`) from the provided definition file:

```console
apptainer build rafo.sif apptainer.def
```

Then run any command below by prefixing it with:

```console
apptainer exec --nv rafo.sif <COMMAND>
```

(The `--nv` flag exposes the host's NVIDIA GPUs to the container.)

## Prediction

### Pretrained weights

Download the pretrained RAFO checkpoint:

| Model | Config | Download |
| --- | --- | --- |
| RAFO | `legolas` | https://zenodo.org/records/23260085/files/rafo_2nd_stage.ckpt |

```console
export WEIGHTS_URL="https://zenodo.org/records/23260085/files/rafo_2nd_stage.ckpt"
mkdir -p weights
wget $WEIGHTS_URL -O weights/rafo.ckpt
```

The `--config` passed to `run.py` must match the one the checkpoint was trained with (see the table above).

> The RiNALMo language model weights are downloaded automatically while the container is being built, so you don't need to fetch them yourself.

### Input format

Put your sequences in a FASTA file, one entry per structure. The FASTA header (up to the first whitespace) is used as the output file name. For multi-chain complexes, separate the chains with `&`:

```
>single
GGGAAACUUCGGUUUCCC
>duplex
GGCGCUUAGC&GCUAAGCGCC
```

### Running prediction

```console
apptainer exec --nv rafo.sif python run.py predict sequences.fasta \
    --config legolas --init_params weights/rafo.ckpt \
    --output_dir predictions \
    --accelerator gpu --devices 1 \
    --num_workers 4 --pin_memory
```

This writes one `<header>.pdb` file per FASTA entry to `--output_dir`. The per-residue confidence (pLDDT) is stored in the B-factor column.

Useful options:

- `--seed <int>` — fix the random seed so results are reproducible.
- `--no_cycles <int>` — override the number of recycling iterations used at inference.
- `--chunk_size <int>` / `--use_lma` — memory optimizations for long sequences.
- `--dropout` — keep dropout on at inference. Running with different `--seed` values then gives an ensemble of predictions.

Raw predictions can be refined afterwards with `scripts/refine_structs.py` (see [Helper scripts](#helper-scripts)).

## Data preparation

Datasets are generated with `prepare_data.py`. Each subcommand has a `--help` flag listing all available options.

### Training data — `train`

Builds the training dataset from the PDB / RNA 3D Hub.

```console
python prepare_data.py train ~/train_data \
    --num_workers 64 --log_file ~/train_data/pipeline.log \
    --rna3dhub_version 4.1 --max_resolution 9.0 \
    --min_seq_len 32 --max_seq_len 1024 --max_undef_seq_perc 0.3 \
    --min_max_chain_len 16 --cutoff_date 2022-04-13 \
    --dup_min_seq_id 0.9 --dup_coverage 0.8
```

### Test data — `seq-test`

Builds an evaluation dataset that is dissimilar to a given training set (sequence-wise), so that the model is tested on genuinely novel RNAs. Command requires `--train_structs_dir` (the directory of parsed training structures).

```console
python prepare_data.py seq-test ~/seq_test_data \
    --num_workers 64 --log_file ~/seq_test_data/pipeline.log \
    --rna3dhub_version 4.1 --max_resolution 4.0 \
    --min_seq_len 32 --max_seq_len 1024 --cutoff_date 2022-04-13 \
    --train_structs_dir ~/train_data/parsed/ \
    --overlap_min_seq_id 0.9 --overlap_coverage 0.8
```

## Training and evaluation

These are driven by `run.py` via the `fit` and `test` subcommands (inference is covered in [Prediction](#prediction)). The `--config` flag selects a model/loss preset (default: `frodo`). Run `python run.py <command> --help` for the full option list.

### Training — `fit`

```console
python run.py fit ~/train_data \
    --val_data_dirs ~/val_data \
    --config gimli --batch_size 4 \
    --accelerator gpu --devices 2 --precision bf16-mixed
```

### Evaluation — `test`

```console
python run.py test ~/test_data \
    --config gimli --ckpt_path ~/trained_model.ckpt \
    --accelerator gpu --devices 2 --precision bf16-mixed
```

## Helper scripts

The `scripts/` directory contains auxiliary tools (run any with `--help`):

- **`refine_structs.py`** — refine predicted structures with QRNAS (energy minimization / geometry optimization):

  ```console
  python scripts/refine_structs.py \
      --input_dir <predicted_structs_dir> \
      --output_dir <refined_structs_dir> \
      --num_refinement_steps 20000 --num_workers 4
  ```

- **`eval.py`** — compute evaluation metrics for predicted structures against reference targets and write them to a CSV:

  ```console
  python scripts/eval.py \
      --target_dir <target_structs_dir> \
      --pred_dirs <pred_dir_1> <pred_dir_2> \
      --output_file metrics.csv
  ```

## Repository layout

| Path | Description |
| --- | --- |
| `rafo/` | Core package (model, data pipeline, Lightning wrapper, config) |
| `prepare_data.py` | Dataset generation (train / seq-test) |
| `run.py` | Training, evaluation, and inference entry point |
| `scripts/` | Helper scripts (refinement, evaluation) |
| `notebooks/` | Analysis and plotting notebooks |
| `apptainer.def` | Container definition file |
| `environment.yml` | Conda environment specification |

## License

Released under the [MIT License](LICENSE).
