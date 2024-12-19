# Audio Tune Notebook

## Motivation

### Fine-Tuned Collaboration: 
Exploring Generative Music Through Participatory Data Collection

About a year ago, I experienced for the first time generating music using AI models, I was amazed at how seamlessly it mimicked human compositions. The melodies were complex, the rhythms were convincing, and it felt like I was witnessing the future of music. But the more I used generative music models, that amazement shifted to unease: I began to hear repetitive patterns and lyrics, it was as if the model was spewing out some variations on the mean value in the latent space of popular music. And it was easy to see why.

The current two most well-known generative music models are Suno and Udio. Both relied on datasets that were often opaque and generic, reinforcing mainstream genres while marginalizing niche and experimental practices. The lawsuits brought against these AI music generators by major record labels further underscored the ethical and legal issues at play. In June 2024, the Recording Industry Association of America (RIAA) filed suits alleging “mass infringement of copyrighted sound recordings” by companies using protected music to train their models without proper licenses [1]. The data collection practices of these models raised deep concerns for many musicians including myself. Many datasets for Automatic Music Generation are poorly documented, embedding values that exclude experimental genres and artistic practices [2]. The business models of platforms like Suno and Udio only compound these issues. By producing music that mimics established artists without licensing or compensation, they disrupt traditional revenue streams and make it harder for human artists to thrive [3].

These realizations motivated me to seek an alternative: a way to infuse humanity back into the process of making music with AI. I started to envision a participatory approach where musicians themselves play an active role in creating the AI model that best fits their own creative practice. What if, instead of relying entirely on foundation models to generate music, we use AI to support and reflect the unique creative practices of artists? This led me to experiment with a creative workflow where a group of musicians collaboratively produce a fine-tuned model for themselves, training it on the experimental sounds they create together.

My project uses the ARP 2500 synthesizer, a rare instrument known for its rich history in electronic music. I wanted to capture the uniqueness of its soundscape and the artistic styles of the musicians I worked with. By focusing on participatory methodologies and community-driven data collection, my project provides an alternative to the homogenizing tendencies of foundation models. I aim to challenge the status quo by exploring whether a group of artists can create and own an AI model that authentically reflects their creative practices.

## Design

The decisions behind the design of this project were informed by the values of ownership, collaboration, and creative agency. Each step of the process reflected an attempt to navigate the complexities of working with generative AI models in a way that centered the musicians’ control over their own creative labor.

### A Data Governance Plan

The first step in the project was to establish trust and agree on a way to manage our data responsibly. Who should own the recordings we create? Who has the right to use or profit from them? These questions are tied to real anxieties about how creative labor is commodified in AI-driven systems. I wanted to ensure that me and the musicians I collaborated with retained ownership of our work.

In our agreement, each musician retained access to the recordings, the annotations, and the weights of the fine-tuned model. The governance plan was a way to ensure that the musicians were not just contributors but co-owners of the tools we built.

### Creating the Data

Our creative process centered around the ARP 2500 synthesizer. I scheduled jam sessions with the four artists I collaborated with. The jam sessions were exploratory and messy, we did not have to create a coherent piece of music, instead, it was about capturing sounds that we found unique and interesting. This freeform way of jamming out sounds instead of songs is different from any other workflow that we are accustomed to. 

From here, the raw recordings were then segmented into smaller audio snippets to prepare them for use in training the model. We collaboratively annotated the audio snippets with detailed descriptions of their mood, type, and texture. The annotations included descriptions “frequency modulated wet bubbly FX,” “heavy resonance acid techno bassline,” and “gentle ominous ambient pads.” This annotation process drew on the musicians’ shared vocabulary and creative intuition, ensuring that the metadata captured the nuanced qualities of the sounds. 

### Fine-Tuning the Model

For the generative model, I selected Stable Audio Open, an open-source system trained on publicly available audio samples. This choice aligned with the project’s emphasis on transparency and accessibility. I created a simple pipeline to process the recordings and annotations into a format suitable for fine-tuning the model. The pipeline [5] enabled precise control over training parameters, such as sample rate, batch size, and training precision. Through trial and error, I found that 1000 training steps struck the right balance, producing a model that captured the qualities of the training data without overfitting. The fine-tuned model weights were saved as a checkpoint file.

To make the model accessible to the group, the fine-tuned model was hosted on Gradio, thus we have an interface that allows us to generate as many ARP style samples as we wanted, which we could then incorporate into our future productions. This completes the processes of creating the data, fine-tuning a model, and owning the weights and outputs of our labor.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Install Dependencies](#2-install-dependencies)
3. [Usage](#usage)
    - [1. Setup Environment](#1-setup-environment)
    - [2. Download Models and Datasets](#2-download-models-and-datasets)
    - [3. Login to Weights & Biases](#3-login-to-weights--biases)
    - [4. Train the Model](#4-train-the-model)
    - [5. Unwrap the Model](#5-unwrap-the-model)
    - [6. Run Gradio Interface](#6-run-gradio-interface)
4. [Example](#example)
5. [Troubleshooting](#troubleshooting)
6. [Additional Resources](#additional-resources)

## Prerequisites

Before running the notebook, ensure you have the following:

- **Hardware:**
  - A machine with an NVIDIA GPU.
  - CUDA installed and properly configured.

- **Software:**
  - Python 3.10 or higher.
  - Jupyter Notebook or JupyterLab installed.
  - Git installed.
  - Internet connection for downloading dependencies and datasets.

- **Accounts:**
  - [Weights & Biases](https://wandb.ai/) account for experiment tracking.

## Installation

### 1. Clone the Repository

Open your terminal and execute the following commands to clone the repository and navigate into it:

```bash
git clone https://github.com/jackshuxu/audio-tune.git
cd audio-tune

2. Install Dependencies

Open the tune.ipynb notebook in Jupyter and execute the following cells sequentially to install the required dependencies.

a. Verify NVIDIA GPU and CUDA Installation

!nvidia-smi

This command checks if your NVIDIA GPU and CUDA are properly installed and recognized.

b. Clone and Install Stable Audio Tools

!git clone https://github.com/Stability-AI/stable-audio-tools.git
%cd stable-audio-tools
!pip install -e .
%cd ..

Clones the stable-audio-tools repository and installs it in editable mode.

c. Install Additional Dependencies

!apt-get update -y
!apt-get install ffmpeg -y
!pip install gdown

Updates the package list, installs ffmpeg for audio processing, and installs gdown for downloading files from Google Drive.

Usage

Follow these steps to run the notebook effectively:

1. Setup Environment

Execute the following cells to configure your environment. These cells are currently commented out and should be uncommented when needed.

# from huggingface_hub import hf_hub_download
# from huggingface_hub import notebook_login

# notebook_login()

Use these commands to log in to Hugging Face Hub if you need to download models from there.

# !git config --global credential.helper store

Stores Git credentials globally to avoid repeated prompts.

2. Download Models and Datasets

Uncomment and execute the following cells to download necessary models and datasets.

a. Download Models from Hugging Face

# hf_hub_download(repo_id="stabilityai/stable-audio-open-1.0", filename="model.ckpt", local_dir="./")
# hf_hub_download(repo_id="stabilityai/stable-audio-open-1.0", filename="model_config.json", local_dir="./")

b. Download Datasets from Google Drive

# !gdown 1GAslDjTS_RK_Bc2b7ZqLdmwbf2kW9LJb

Replace the URL with your specific Google Drive file ID if different.

c. Extract Datasets

# import zipfile

# zip_path = "/workspace/Full ARP Pack.zip"    
# extract_dir = "/workspace/dataset"

# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_dir)

Extracts the downloaded dataset ZIP file to the specified directory.

3. Login to Weights & Biases

Execute the following cell to authenticate with Weights & Biases for experiment tracking.

!wandb login <your_wandb_api_key>

Replace <your_wandb_api_key> with your actual W&B API key. You can obtain this from your W&B account settings.

4. Train the Model

Start the training process by executing the following cell:

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

Ensure the paths to dataset.json, model_config.json, and model.ckpt are correct.

5. Unwrap the Model

After training, unwrap the model using the following cell:

%cd /workspace/stable-audio-tools

command = (
    "python ./unwrap_model.py"
    " --model-config /workspace/model_config.json"
    " --ckpt-path /workspace/checkpoints/stable_audio_open_finetune/x6qp450d/checkpoints/epoch=53-step=800.ckpt"
    " --name AIRP_53_800"
)
!{command}

Update the --ckpt-path to point to your specific checkpoint file.

6. Run Gradio Interface

Launch the Gradio web interface to interact with the trained model:

%cd /workspace

command = (
    "python stable-audio-tools/run_gradio.py"
    " --model-config /workspace/model_config.json"
    " --ckpt-path /workspace/AIRP_53_800.ckpt"
    " --share"
)
!{command}

Ensure the --ckpt-path corresponds to the unwrapped model’s checkpoint.

Example

To avoid potential issues, ensure all notebook cells are executed in the order provided. This sequential execution ensures that dependencies are installed, environments are set up, and all necessary files are downloaded before proceeding to training and deployment stages.

Troubleshooting
	•	Dependency Issues:
	•	Ensure all dependencies are installed by rerunning the installation cells.
	•	Check for any error messages during pip install and resolve them accordingly.
	•	NVIDIA Drivers and CUDA:
	•	Verify that NVIDIA drivers and CUDA are correctly installed by running nvidia-smi.
	•	Ensure the CUDA version is compatible with your PyTorch installation.
	•	Internet Connectivity:
	•	Ensure a stable internet connection for downloading models and datasets.
	•	If downloads fail, check your network settings or try using a different network.
	•	Weights & Biases Login:
	•	Ensure your W&B API key is correct.
	•	If login fails, regenerate your API key from your W&B account.
	•	Checkpoint Paths:
	•	Verify that the checkpoint paths provided in the training and unwrapping steps are correct and the files exist.
	•	Gradio Interface Issues:
	•	If the Gradio interface doesn’t launch, check for errors in the training and unwrapping steps.
	•	Ensure the model checkpoint specified in the Gradio command exists.

For further assistance, refer to the repository’s Issues section or contact the repository maintainer.

Additional Resources
	•	Stable Audio Tools Repository
	•	Weights & Biases Documentation
	•	Gradio Documentation

