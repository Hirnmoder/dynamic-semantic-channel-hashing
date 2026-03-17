# Dynamic Semantic Channel Hashing
**Open Source Code Repository for paper "DSCH-Loss: A Dynamic Semantic Channel Objective for Deep Semantic Hashing"**

*currently under review*

## Repository Structure
- `.devcontainer`
  - `devcontainer.json` Configuration file for Visual Studio Code Devcontainer for development
- `code`
  - `deep-semantic-hashing` Python package code for DSH
    - `src`
      - `dsh`
        - `config` Classes for configuration parsing
        - `data` Datasets, Dataloader
        - `embedding` Embedder adapter
        - `metric` Hamming metrics
        - `model` TDSRDH, ClipHash
        - `run` Trainer, Model adapter
        - `utils` Utilities
  - `run` Code for running a training/inference/evaluation pipeline
    - `main.py` Main Python file to run
    - `config.all.json5` Full experiment pipeline configuration
- `data` Folder that contains datasets, see [below](#how-to-set-up-dataset-folder)
- `env`
  - `docker-compose.yml` Example Docker-Compose file with optional GPU support
  - `Dockerfile` Build instructions for Docker image used as Devcontainer or Docker-Compose Container
  - `requirements.dev.txt` Development requirements (code formatter, linting, type checker)
  - `requirements.txt` Runtime requirements
- `logs` Folder that contains TensorBoard logs of experiment runs
- `models` Folder that contains PyTorch models of experiment runs


## How to set up dataset folder
The following files and folder structures must be present in the `data` folder to be able to conduct experiments.
- `MIR-FLICKR25K` Folder containing the MIR-FLICKR25K dataset
  - `annotations_v080` Folder containing 39 TXT files (labels), of which 24 are used; [Download](https://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip)
  - `data` Folder containing 25000 image files named `im<#id>.jpg`; [Download](https://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip)
    - `doc` Folder containing informational TXT files, ignored
    - `meta`
      - `exif` Folder containing EXIF data, ignored
      - `exif_raw` Folder containing EXIF data, ignored
      - `license` Folder containing license data, ignored
      - `tags` Folder containing 25000 TXT files named `tags<#id>.txt`
      - `tags_raw` Folder containing 25000 TXT files named `tags<#id>.txt`
- `NUS-WIDE` Folder containing the NUS-WIDE dataset; [Download](https://huggingface.co/datasets/Lxyhaha/NUS-WIDE)
  - `Groundtruth` Folder
    - `AllLabels` Folder containing 81 TXT files (labels)
    - `TrainTestLabels` Folder, ignored
  - `ImageList` Folder with 3 TXT files
    - `Imagelist.txt` List of all image file names
    - `TestImagelist.txt` List of test set image file names
    - `TrainImagelist.txt` List of train set image file names
  - `kaggle` Folder [Download](https://www.kaggle.com/datasets/xinleili/nuswide)
    - `images` Folder containing 269655 image files
    - Various TXT and MD files, ignored
  - `NUS_WID_Tags` Folder
    - `All_Tags.txt` List of all user-defined tags for all images
    - Various other TXT and DAT files, ignored
- `GoogleNews-vectors-negative300.bin` Google-News-W2V file; [Download](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM), Link taken from [code.google.com](https://code.google.com/archive/p/word2vec/)

## Run the code

### Run a full experiment pipeline

```bash
cd /app/code/deep-semantic-hashing/run
python3 main.py all
```

Uses configuration file `config.all.json5` in the same folder.

### Additional commands

```bash
cd /app/code/deep-semantic-hashing/run
python3 main.py help
```

Available commands:
- `help` Displays a short CLI help
- `train` Runs a training-only pipeline
- `infer` Runs a inference-only pipeline
- `metrics` Runs a metrics-only pipeline
- `all` Runs a full pipeline
- `create-config` TUI for creating a configuration file using presets; not all features implemented
