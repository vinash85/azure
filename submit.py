"""
Submits the DermLIP HAM10000 fine-tuning job to Azure ML.
Distributed: NODES x GPUS_PER_NODE ranks.
Checkpoints are written directly to blob storage (preemption-safe).
Uses a custom environment built from conda.yaml on top of ACPT PyTorch.
"""
from azure.ai.ml import Input, MLClient, Output, command, PyTorchDistribution
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# --- Workspace ---
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="028e301e-1349-4216-8553-0573fe245382",
    resource_group_name="trial",
    workspace_name="Trial-nodemo",
)

# --- Tunable parameters ---
NODES = 3
GPUS_PER_NODE = 1
EPOCHS = 20
LR = 1e-4
BATCH_SIZE = 64
WEIGHT_DECAY = 0.05
FINETUNE_MODE = "full"          # "full" | "linear" | "lora"
MODEL_NAME = "hf-hub:redlessone/DermLIP_ViT-B-16"

DATA_ASSET = "azureml:ham10k:1"
CKPT_PATH = "azureml://datastores/workspaceblobstore/paths/checkpoints/dermlip-ham10k/"
COMPUTE = "trailsft"

# --- Custom environment: ACPT PyTorch base + conda.yaml extras ---
custom_env = Environment(
    name="dermlip-env",
    description="DermLIP fine-tuning env on top of ACPT PyTorch 2.8 CUDA 12.6",
    conda_file="./conda.yaml",
    image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.8-cuda12.6:latest",
)
custom_env = ml_client.environments.create_or_update(custom_env)
print(f"Environment registered: {custom_env.name}:{custom_env.version}")

# --- Build job ---
job = command(
    code="./",
    command=(
        "python train.py "
        "--data ${{inputs.training_data}} "
        "--output-dir ${{outputs.checkpoints}} "
        f"--model-name {MODEL_NAME} "
        f"--finetune-mode {FINETUNE_MODE} "
        f"--epochs {EPOCHS} "
        f"--batch-size {BATCH_SIZE} "
        f"--lr {LR} "
        f"--weight-decay {WEIGHT_DECAY} "
        f"--nodes {NODES} "
        f"--gpus-per-node {GPUS_PER_NODE}"
    ),
    inputs={
        "training_data": Input(
            type=AssetTypes.URI_FOLDER,
            path=DATA_ASSET,
            mode=InputOutputModes.RO_MOUNT,
        ),
    },
    outputs={
        "checkpoints": Output(
            type=AssetTypes.URI_FOLDER,
            path=CKPT_PATH,
            mode=InputOutputModes.RW_MOUNT,
        ),
    },
    environment=custom_env,
    compute=COMPUTE,
    instance_count=NODES,
    distribution=PyTorchDistribution(process_count_per_instance=GPUS_PER_NODE),
    display_name="dermlip-ham10k-ft",
    experiment_name="dermlip-ham10k",
)

returned = ml_client.jobs.create_or_update(job)
print(returned.studio_url)
