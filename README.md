# README

## Audio Tune Notebook

This README provides instructions for setting up and running the `tune.ipynb` Jupyter notebook in the `audio-tune` repository. This notebook includes steps for setting up the environment, installing dependencies, and running training and evaluation commands.

## Prerequisites

Before running the notebook, ensure you have the following:

- A machine with an NVIDIA GPU and CUDA installed.
- Python 3.10 or higher.
- Jupyter Notebook installed.
- Internet connection for downloading dependencies and datasets.

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/jackshuxu/audio-tune.git
    cd audio-tune
    ```

2. **Install Dependencies**

    Open the `tune.ipynb` notebook and execute the following cells to install the required dependencies:

    ```python
    !nvidia-smi
    ```

    ```python
    !git clone https://github.com/Stability-AI/stable-audio-tools.git
    %cd stable-audio-tools
    !pip install -e .
    %cd ..
    ```

    ```python
    !apt-get update -y
    !apt-get install ffmpeg -y
    !pip install gdown
    ```

## Usage

Follow these steps to run the notebook:

1. **Setup Environment**

    Execute the following cells to set up the environment:

    ```python
    # from huggingface_hub import hf_hub_download
    # from huggingface_hub import notebook_login

    # notebook_login()
    ```

    ```python
    # !git config --global credential.helper store
    ```

2. **Download Models and Datasets**

    Uncomment and execute the following cells if you need to download models and datasets:

    ```python
    # hf_hub_download(repo_id="stabilityai/stable-audio-open-1.0", filename="model.ckpt", local_dir="./")
    # hf_hub_download(repo_id="stabilityai/stable-audio-open-1.0", filename="model_config.json", local_dir="./")
    ```

    ```python
    # !gdown 1GAslDjTS_RK_Bc2b7ZqLdmwbf2kW9LJb
    ```

    ```python
    # import zipfile

    # zip_path = "/workspace/Full ARP Pack.zip"    
    # extract_dir = "/workspace/dataset"

    # with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #     zip_ref.extractall(extract_dir)
    ```

3. **Login to Weights & Biases**

    Execute the following cell to log in to Weights & Biases:

    ```python
    !wandb login <your_wandb_api_key>
    ```

4. **Train the Model**

    Execute the following cell to start the training process:

    ```python
    %cd /workspace/stable-audio-tools

    command = (
        "python3 train.py"
        " --dataset-config /workspace/dataset.json"
        " --model-config /workspace/model_config.json"
        " --name stable_audio_open_finetune"
        " --save-dir /workspace/checkpoints"
        " --checkpoint-every 200"
        " --batch-size 16"
        " --num-gpus 2"
        " --precision 16-mixed"
        " --seed 128"
        " --pretrained-ckpt-path /workspace/model.ckpt"
    )

    !{command}
    ```

5. **Unwrap the Model**

    Execute the following cell to unwrap the trained model:

    ```python
    %cd /workspace/stable-audio-tools

    command = (
        "python ./unwrap_model.py"
        " --model-config /workspace/model_config.json"
        " --ckpt-path /workspace/checkpoints/stable_audio_open_finetune/x6qp450d/checkpoints/epoch=53-step=800.ckpt"
        " --name AIRP_53_800"
    )
    !{command}
    ```

6. **Run Gradio Interface**

    Execute the following cell to run the Gradio interface:

    ```python
    %cd /workspace

    command = (
        "python stable-audio-tools/run_gradio.py"
        " --model-config /workspace/model_config.json"
        " --ckpt-path /workspace/AIRP_26_400.ckpt"
        " --share"
    )
    !{command}
    ```

## Example

Ensure all cells are executed in the order provided to avoid any issues.

## Troubleshooting

- Ensure all dependencies are installed correctly.
- Verify NVIDIA drivers and CUDA are properly installed and configured.
- Check internet connectivity for downloading models and datasets.

For additional support, refer to the repository's issues section.
