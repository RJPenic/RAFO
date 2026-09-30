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

## Training, evaluation, and inference

These are driven by `run.py` via the `fit`, `test`, and `predict` subcommands. The `--config` flag selects a model/loss preset (default: `frodo`). Run `python run.py <command> --help` for the full option list.

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

### Inference — `predict`

Predict structures for the RNA sequences in a FASTA file:

```console
python run.py predict ~/sequences.fasta --output_dir ~/predictions \
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
