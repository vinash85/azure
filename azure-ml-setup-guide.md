# Azure ML End-to-End Setup Guide

A practical, distilled walkthrough from a fresh Azure subscription to a running distributed PyTorch training job. Based on hands-on setup for DermLIP fine-tuning on HAM10000, but the structure applies to any deep learning workload.

> **Reality check on authentication:** Many enterprise Azure tenants block device-code logins (`az login --use-device-code`, `azcopy login`) via Conditional Access policies. If that's your situation, you'll authenticate by other means — **SAS tokens for data upload**, and a **compute instance with managed identity for job submission**. This guide covers both the happy path and the workarounds.

---
## Video Links:
- All video series: [Full video](https://drive.google.com/drive/folders/1h4Sy1yrMCmXdbAWB5jZDO2nHyTj12y-Q?usp=sharing)
- [0 Intro](https://drive.google.com/file/d/1LtGpahFcrrn3KM6YDGLReejE3c3CIjXQ/view?usp=sharing)
- [1 Create Workspace](https://drive.google.com/file/d/1obqniTMIFLF64jCTCqOH4g9I9Ns5oI-a/view?usp=drive_link)
- [2.1 Data Loading into Azure](https://drive.google.com/file/d/1e6Pnqn09rsJMOfKzNuEQOBVN_ft-4eZv/view?usp=drive_link)
- [2.2 Creating asset/ data path](https://drive.google.com/file/d/1Vmt-GwyQb7H_9qDzdroFrlJJhzE4KC5r/view?usp=drive_link)
- [3 Create Cluster and VLM](https://drive.google.com/file/d/1d-tmeAZ2ToZRGqiMHqAK1z_qDDBwqkFX/view?usp=drive_link)
- [4 Code and Dry Run](https://drive.google.com/file/d/1dWCxpDopUByPu7WxnZZcvKkmgWOVjrC6/view?usp=drive_link)
- [5 Launch Cluster from Instance VM](https://drive.google.com/file/d/1Oo-zz5Os_duXlS2Sv-oLr3S2kieD1-wh/view?usp=drive_link)
- [6 Monitoring](https://drive.google.com/file/d/1y7VyYO1iJ2ZP65E7P5OGOa1H2fZZSgOT/view?usp=drive_link)

## Phase 1 — Foundations (one-time setup)

### 1. Resource group
- Portal → Resource groups → **+ Create** → name it, pick a region with your GPU quota.

### 2. Storage account
- Portal → Storage accounts → **+ Create** → same RG, Standard / LRS.
- Leave public network access enabled for managed-VNet setup.

### 3. Azure ML workspace
- Portal → Azure Machine Learning → **+ Create**.
- Pick your RG, storage account, let it auto-create Key Vault and Application Insights.
- Networking: **Allow internet outbound** (simplest managed VNet).
- Container Registry: leave as **None** — auto-created when first needed.

### 4. Verify in Studio (ml.azure.com)
- Confirm four auto-created datastores exist:
  - `workspaceblobstore`
  - `workspacefilestore`
  - `workspaceartifactstore`
  - `workspaceworkingdirectory`

---

## Phase 2 — Data upload (one-time per dataset)

> Conditional Access often blocks `azcopy login`. Use **SAS tokens** instead — they bypass user-level auth policies entirely.

### 5. Get storage URL
- Studio → Data → Datastores → click `workspaceblobstore` → copy the **storage URL**.
- Format: `https://<account>.blob.core.windows.net/<container>`.

### 6. Generate SAS token
- Portal → that storage account → Containers → click the `azureml-blobstore-...` container → **Shared access tokens**.
- Permissions: **Read, Add, Create, Write, List**.
- Expiry: 7 days. HTTPS only.
- Generate → copy the **Blob SAS URL** (the long one ending in `&sig=...`).
- SAS bypasses Conditional Access entirely — it's a signed URL, not an interactive login.

### 7. Upload with AzCopy from remote server

```bash
# Install on Linux
wget https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar.gz
tar -xvf azcopy.tar.gz && sudo cp ./azcopy_linux_amd64_*/azcopy /usr/local/bin/

# Copy (NOT azcopy login — append SAS to the URL instead)
azcopy copy '/path/to/data/' \
  'https://<account>.blob.core.windows.net/<container>/<dest-folder>?<sas>' \
  --recursive
```

- **Wrap URL in single quotes** — bash mangles `&` otherwise.
- Trailing slash on source uploads *contents* into the destination folder.
- If you get "no valid source-destination combination" — you used an `azureml://` URI; AzCopy only speaks raw blob URLs.
- If all transfers fail silently — you forgot to append the SAS token. AzCopy will run unauthenticated and fail on every file with no clear message.

### 8. Register as Data Asset
- Studio → Data → Data assets → **+ Create** → name it → **Folder (uri_folder)** → **From a datastore** → `workspaceblobstore` → browse to your uploaded folder → Create.
- A data asset is just a *named pointer* to the storage path. It doesn't move or copy anything.
- Get the named URI: `azureml:<name>:1` (use this in submit.py).

---

## Phase 3 — Compute setup (one-time)

### 9. Create GPU compute cluster
- Studio → Compute → Compute clusters → **+ New**.
- VM type: **GPU**. Priority: **Low priority** for ~80% cost savings.
- VM size: `Standard_NC24ads_A100_v4` (1× 80GB A100/node) or similar.
- **Min nodes = 0** (critical — scales to zero between jobs, $0 idle cost).
- Max nodes = whatever your job needs (e.g. 3).
- Idle seconds = 120.
- SSH: optional. Note: SSH on a cluster is only for debugging a *running* job — you cannot launch jobs via SSH.
- VNet / Managed identity: leave off for standard setups.

### 10. Verify quota
- Studio → Quotas → confirm the VM family has enough cores.
- Example: 3 × 24 = 72 cores for 3 A100 nodes.
- Low-priority and dedicated have separate quotas.

---

## Phase 4 — Scripts (training code)

### 11. `train.py` design principles
- Auto-detect single vs distributed via `WORLD_SIZE` env var — same script works locally and on cluster.
- Save `latest.pt` every epoch + `best.pt` on improvement.
- Auto-resume from `latest.pt` if present (handles spot preemption).
- Use class weights for imbalanced data.
- Mixed precision (bfloat16) when on GPU, fp32 fallback on CPU.

```python
# Pattern for the WORLD_SIZE auto-detect:
def setup_ddp():
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 0, 1, False   # local_rank, rank, world_size, distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), True
```

### 12. `submit.py` design principles
- Use `DefaultAzureCredential()` — picks up `az login`, service principal env vars, or compute instance managed identity automatically.
- Reference data asset via `Input(path="azureml:<name>:1", mode=RO_MOUNT)`.
- Write checkpoints directly to blob: `Output(path="azureml://datastores/.../paths/checkpoints/", mode=RW_MOUNT)` — preemption-safe.
- Set `instance_count=N` and `PyTorchDistribution(process_count_per_instance=G)` — that's the only place that controls ranks.

### Import note (SDK has moved things around)

```python
# Current (works):
from azure.ai.ml import Input, MLClient, Output, PyTorchDistribution, command
from azure.ai.ml.entities import Environment

# Older docs may show this — fails on newer SDKs:
# from azure.ai.ml.entities import PyTorchDistribution   # WRONG
```

---

## Phase 5 — Environment

### 13. Custom environment via conda.yaml
- Build on top of latest ACPT base image (currently `acpt-pytorch-2.8-cuda12.6`).
- Write conda.yaml **by hand** — `conda env export --from-history` misses pip-installed packages.
- **Don't pin torch/torchvision** — base image provides them.
- Pin everything else with versions from your local smoke-tested env for reproducibility.

```yaml
name: dermlip
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
    - open_clip_torch==3.3.0
    - peft==0.19.1
    - scikit-learn==1.7.2
    - pandas==2.3.3
    - Pillow==12.2.0
```

### 14. Register environment in submit.py

```python
custom_env = Environment(
    name="dermlip-env",
    conda_file="./conda.yaml",
    image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.8-cuda12.6:latest",
)
custom_env = ml_client.environments.create_or_update(custom_env)
```

- First job: 5–10 min to build image. After: cached, fast startup.

---

## Phase 6 — Local validation (before burning cluster time)

### 15. Smoke-test train.py locally

Create local conda env mirroring the cluster:

```bash
conda create -n dermlip python=3.10 -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install open_clip_torch peft scikit-learn pandas Pillow
```

- Match PyTorch CUDA build to local driver. Check with `nvidia-smi` (top right shows max supported CUDA). If `torch.version.cuda` > driver CUDA → reinstall PyTorch with matching `cu1XX` index URL.
- Run **without DDP env vars** (script auto-detects single-process mode and skips NCCL):

```bash
python train.py --data /local/path --epochs 1 --batch-size 2 --num-workers 0
```

- If it runs a few steps without error, code logic is sound.
- Local CUDA mismatch ≠ cluster issue (cluster has ACPT image with everything pre-matched).
- NCCL errors like "out of memory" on a single GPU with plenty of free memory usually mean DDP shouldn't be initialized at all — the `WORLD_SIZE==1` short-circuit handles this.

---

## Phase 7 — Authentication (the realistic flowchart)

This is where most people get stuck. Pick the first option that works for your situation:

### Option A — `az login` from a corporate/managed device
- Install Azure CLI: `curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash`
- Run: `az login` (browser opens — interactive flow)
- Works if your device is recognized as compliant by Conditional Access.
- If you get "doesn't meet the criteria" → your tenant blocks this. Skip to Option C.

### Option B — Service principal (if you have Entra ID permissions)
- Portal → Entra ID → App registrations → New registration.
- Generate client secret. **Copy the value immediately — it's shown once.**
- Resource group → IAM → Add role assignment → Contributor → assign to the service principal.
- On the server:
  ```bash
  export AZURE_TENANT_ID=...
  export AZURE_CLIENT_ID=...
  export AZURE_CLIENT_SECRET=...
  ```
- `DefaultAzureCredential` auto-detects these env vars.
- **Bypasses Conditional Access** because it's not a user signing in.
- Caveat: many orgs restrict who can create app registrations. If "you don't have permission to register applications" → skip to Option C.

### Option C — Compute instance with managed identity (universal fallback)

**Use this when:**
- You don't have permission to create app registrations in Entra ID.
- Conditional Access blocks all device-code flows on your server.
- You need a quick path with zero auth setup.

**Why it works:** The compute instance has the workspace's managed identity attached. `DefaultAzureCredential` picks it up automatically. No user login, no Conditional Access policy applies.

**Steps:**
1. Studio → Compute → Compute instances → **+ New**.
2. Pick a small CPU SKU (`Standard_A1_v2` or `Standard_DS3_v2` is plenty — it only orchestrates).
3. SSH: optional. If you want to SSH in from outside:
   - Generate an **RSA** key on your local machine: `ssh-keygen -t rsa -b 4096 -f ~/.ssh/azure_rsa`
   - Paste the contents of `~/.ssh/azure_rsa.pub` into the form.
   - Azure rejects ed25519 keys with "The public key must be in ssh-rsa format."
4. SSO: leave enabled (default).
5. Create. Wait ~3 min.
6. Once **Running**, get the IP/port from the instance details page in Studio.
7. Connect:
   - From browser: Studio → click the instance → **Terminal** or **JupyterLab**.
   - From CLI: `ssh -i ~/.ssh/azure_rsa -p <port> azureuser@<ip>` (port is NOT 22; usually 50000+).
8. Activate the SDK env: `conda activate azureml_py310_sdkv2`
9. Verify: `python -c "from azure.ai.ml import MLClient; print('ok')"`
10. Upload your code (git clone, scp, or Studio's file browser).
11. `python submit.py` — managed identity handles auth automatically.
12. **Stop the instance** when done (Studio → Compute → select instance → Stop). $0 when stopped.

---

## Phase 8 — Submit and monitor

### 16. Project folder layout

```
project/
├── .amlignore             # excludes from job upload
├── .gitignore             # excludes from git
├── README.md
├── azureml-job-env.yml    # local SDK env spec (for reproducibility)
├── conda.yaml             # cluster job env spec
├── submit.py
└── train.py
```

`.amlignore` works like `.gitignore` — keeps local-only files (logs, caches, outputs) out of the code upload to Azure.

### 17. Submit

```bash
python submit.py
```

Expected output:
```
Environment registered: dermlip-env:1
Uploading aztrial (X.X MBs): 100%|...
https://ml.azure.com/runs/<random-name>?wsid=...
```

Warnings about "experimental class" and "pathOnCompute" are noise — ignore them.

### 18. Job phases in Studio
- **Preparing**: building image (5–10 min first time, cached after).
- **Queued**: cluster scaling from 0 to N nodes (3–5 min).
- **Running**: training happening.
- **Completed / Failed**: artifacts in Outputs + logs tab.

### 19. Where to look for output
- `Outputs + logs/user_logs/std_log_process_0.txt` — rank 0's prints.
- `Outputs + logs/user_logs/std_log_process_N.txt` — other ranks.
- Checkpoints stream to `workspaceblobstore/checkpoints/<run>/` as epochs finish (only because we used `RW_MOUNT` output).
- Cluster auto-scales back to 0 within 120 seconds after job ends.

### 20. After submission
- The job runs independently. You can close the terminal, stop the compute instance, even shut down your laptop — the cluster keeps going.
- `submit.py` is just a dispatcher (~30 seconds of work). The cluster does the long-running training.

---

## Key Concepts to Internalize

### Compute cluster ≠ compute instance
- **Cluster**: batch job runner, scales 0→N, ephemeral nodes. Runs your training.
- **Instance**: persistent dev box, like a personal VM. Used for orchestration (submitting jobs) or interactive development.

### SSH on cluster is for debugging only
- You can SSH into a cluster node *while it's running a job*, not before.
- Useful for VS Code Remote attach, `nvidia-smi` checks during a job.
- **Not** a way to launch jobs.

### Local env and cluster env are independent
- Your local PyTorch/CUDA doesn't need to match the cluster's.
- Connected only by your Python code and your conda.yaml's extras list.
- ACPT base image on the cluster provides PyTorch + CUDA already matched to the GPU drivers.

### Data flow
- **Storage account** holds bytes.
- **Datastore** is a workspace connection to it.
- **Data asset** is a versioned pointer to a folder/file inside the datastore.
- Jobs reference data assets, not raw paths.

### Cost levers
- **Compute instance**: stop when not in use (~$0.04–0.19/hr otherwise depending on SKU).
- **Compute cluster**: min nodes = 0 (no charge between jobs).
- **Low-priority VMs**: ~80% cheaper, can be preempted (need checkpoint/resume in code).
- **Storage**: pennies per GB per month.

### The mental model
| Concept | What it is |
|---|---|
| Workspace | Your project's command center (free) |
| Storage | Where data and artifacts live (pennies) |
| Compute | What costs real money (only when running) |
| Data asset | Named pointer for jobs to find data |
| Environment | Container spec defining the job's runtime |
| Job | One execution: cluster scales up → runs script → scales down |

---

## The Pipeline at a Glance

From fresh subscription to running distributed training:

```
RG → storage → workspace → upload data (SAS + AzCopy) → register asset
   → create cluster → write scripts → register environment
   → authenticate (compute instance is the safe fallback) → submit
```

Once the pieces exist, the iteration loop is just:

```
edit code → python submit.py → watch
```

---

## Common Pitfalls (Lessons Learned)

### Authentication
1. **Conditional Access can block `az login` / `azcopy login` / device-code SDK flows** — and the error message is misleading ("doesn't meet the criteria"). Don't fight it; pivot to SAS tokens for data and a compute instance for job submission.
2. **You may not have permission to create Entra ID app registrations** — service principal is then off the table without an admin.
3. **The compute instance is the universal workaround** — managed identity bypasses every auth headache.

### Data upload
4. **AzCopy needs raw blob URLs, not `azureml://` URIs.**
5. **Without a SAS token appended, AzCopy uploads fail silently** — every file errors out, no clear message.
6. **Wrap URLs in single quotes** in bash — `&` in SAS query strings breaks unquoted commands.
7. **Data asset wizard doesn't help upload from remote servers** — it's a pointer creator, not an uploader. Upload first, then register.

### Environment & code
8. **`conda env export --from-history` misses pip packages** — write conda.yaml by hand.
9. **Don't pin torch in conda.yaml** — let the ACPT base image provide it.
10. **Curated environment names change** — verify the latest version in Studio → Environments tab before hardcoding.
11. **Compute instance default Python env may not have azure-ai-ml** — `conda activate azureml_py310_sdkv2` first.
12. **`PyTorchDistribution` import moved** — newer SDK: `from azure.ai.ml import PyTorchDistribution`, not from `entities`.

### Local testing
13. **Local CUDA mismatch is NOT a cluster problem** — they're independent environments. Fix locally just for smoke testing.
14. **NCCL fails weirdly with `WORLD_SIZE=1`** ("out of memory" even with plenty of memory) — auto-detect and skip distributed init for local smoke tests.

### Cluster behavior
15. **Cluster scaling 0→N takes 3–5 min** — be patient on first run.
16. **Spot/low-priority VMs can be preempted** — checkpoint every epoch, auto-resume from `latest.pt`.
17. **SSH on cluster nodes only works during an active job** — not for offline access.

### Compute instance gotchas
18. **SSH keys must be RSA**, not ed25519 — `ssh-keygen -t rsa -b 4096` works; default `ssh-keygen` (ed25519 on newer systems) doesn't.
19. **SSH port is not 22** — Azure assigns a custom port (usually 50000+). Check the instance details page.
20. **Stop the compute instance manually** — it doesn't auto-scale like the cluster.
