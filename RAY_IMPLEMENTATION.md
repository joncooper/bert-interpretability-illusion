# Ray Implementation Guide for BERT Interpretability Experiments

**Document Version:** 2.0 (Revised for Realistic Use Case)
**Last Updated:** 2025-11-06
**Author:** Claude Code
**Purpose:** Guide for using Ray to offload GPU work from Mac to home PC

---

## Table of Contents

1. [Do You Actually Need Ray?](#do-you-actually-need-ray)
2. [Realistic Use Case: Remote GPU Access](#realistic-use-case-remote-gpu-access)
3. [Simple Setup: Mac → Home PC](#simple-setup-mac--home-pc)
4. [Ray Client vs. Ray Jobs](#ray-client-vs-ray-jobs)
5. [Alternative Simpler Solutions](#alternative-simpler-solutions)
6. [If You Really Do Need Scale...](#if-you-really-do-need-scale)
7. [References](#references)

---

## Do You Actually Need Ray?

### The Reality Check

Based on your actual setup:
- **Your hardware:** M4 MacBook Pro (work) + 4070 Ti Super (home)
- **Your experience:** M4 MBP runs the experiments fine
- **Actual computation:** Doc/REPLICATION_PLAN.md estimates 4-8 hours for full embedding generation on 625K sentences

**Honest assessment: You probably don't need Ray for this project.**

### Why Ray Might Be Overkill

1. **Single GPU is sufficient:** The 4070 Ti Super has 16GB VRAM - plenty for BERT inference
2. **M4 Mac already works:** You said it runs fine on your laptop
3. **Not truly distributed:** You have 2 machines, not 20
4. **Complexity overhead:** Ray adds debugging complexity and network dependencies
5. **Simpler alternatives exist:** SSH, Remote Desktop, or just running locally

### When You *Would* Need Ray

You'd benefit from Ray if:
- ✅ Running the same experiment across 100s of hyperparameter combinations
- ✅ Processing millions of sentences (not 625K)
- ✅ Need to utilize multiple GPUs across multiple machines simultaneously
- ✅ Building a production service that needs auto-scaling
- ✅ Sharing compute resources across a team

### What You Probably Actually Want

**Option 1: Just use the M4 Mac**
- It works fine (your own assessment)
- No network complexity
- Easiest debugging

**Option 2: SSH into home PC when you need GPU**
```bash
# From Mac at work
ssh home-pc
cd ~/bert-interpretability-illusion
uv run python extract_embeddings.py
# Let it run, detach with tmux/screen
```

**Option 3: Remote development (VS Code Remote-SSH)**
- Edit code on Mac
- Execute on home PC
- No Ray needed

**Option 4 (last resort): Ray for remote execution**
- If you really want to submit jobs from Mac that run on home PC
- Most complex setup
- Only worth it if other options don't fit your workflow

---

## Realistic Use Case: Remote GPU Access

If you've decided you genuinely want Ray (after considering simpler alternatives above), here's the realistic setup for your actual use case.

### Your Actual Scenario

```
┌─────────────────┐                    ┌──────────────────┐
│  Mac (Work)     │                    │  Home PC         │
│  - M4 chip      │  ─── Internet ──>  │  - 4070 Ti Super │
│  - Dev env      │                    │  - Ray cluster   │
│  - Submit jobs  │                    │  - Runs jobs     │
└─────────────────┘                    └──────────────────┘
```

**Goal:** Submit BERT embedding extraction from Mac, execute on home PC's GPU

**Challenges:**
1. Home PC must be reachable from internet (port forwarding/VPN)
2. Connection stability (if connection drops, job might fail)
3. Debugging across machines is harder

### Computation Profile (Based on doc/REPLICATION_PLAN.md)

| Task | Estimated Time | GPU Needed? | RAM Needed |
|------|---------------|-------------|------------|
| Full embedding extraction (625K) | 4-8 hours | Yes (optional but 10-20x faster) | 32GB+ |
| Neuron analysis | Minutes | No | 16GB |
| Locality computation | Minutes-Hours | No | 32GB+ |
| UMAP visualization | Minutes | No | 32GB+ |

**Reality:** The M4 Mac has unified memory and strong CPU. The 4070 Ti Super would mainly help with embedding extraction IF you're running the full 625K sentences.

---

## Simple Setup: Mac → Home PC

If you've decided Ray is worth it, here's the minimal setup.

### Step 1: Set Up Ray on Home PC

```bash
# On home PC (Windows/Linux with 4070 Ti Super)
# Install Ray
pip install "ray[default]"

# Start Ray head node
ray start --head --port=6379 --dashboard-port=8265

# Note the output - it will show something like:
# Ray runtime started.
# To connect from another machine: ray.init(address='192.168.1.100:6379')
```

### Step 2: Make Home PC Accessible

**Option A: Tailscale/Zerotier (Easiest and Secure)**
```bash
# Install Tailscale on both machines
# Get your home PC's Tailscale IP (e.g., 100.x.x.x)
# No port forwarding needed!
```

**Option B: SSH Tunnel (Simple)**
```bash
# From Mac, create SSH tunnel
ssh -L 10001:localhost:10001 -L 8265:localhost:8265 user@home-pc

# Now Ray Client can connect via localhost
```

**Option C: Direct Internet (Requires router setup)**
- Port forward 10001 (Ray Client) on your router
- Know your home's external IP or use Dynamic DNS
- Security risk - use with VPN or restrict by IP

### Step 3: Connect from Mac

**Using Ray Client (Interactive):**
```python
import ray

# Connect to home PC
# Replace with your home PC's IP (Tailscale, VPN, or public IP)
ray.init("ray://100.x.x.x:10001")  # Ray Client port

# Now define and run GPU tasks remotely
@ray.remote(num_gpus=1)
def extract_embeddings_batch(sentences):
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained("bert-base-uncased").cuda()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# This runs on your home PC's GPU!
result = ray.get(extract_embeddings_batch.remote(["test sentence"]))
```

**Important Limitations:**
- Ray Client requires stable connection
- Jobs terminate if connection lost >30 seconds
- Not ideal for long-running tasks (4-8 hour embedding extraction)

---

## Ray Client vs. Ray Jobs

### Ray Client: Interactive Remote Execution

**Best for:**
- Short tasks (minutes)
- Interactive development
- Real-time feedback

**Setup:**
```python
# Mac
ray.init("ray://home-pc:10001")

@ray.remote(num_gpus=1)
def quick_task():
    # Runs on home PC
    pass

result = ray.get(quick_task.remote())
```

**Drawbacks:**
- Connection must stay alive
- Mac must stay connected for entire execution
- Not good for 4-8 hour jobs

### Ray Jobs: Submit-and-Forget

**Best for:**
- Long-running tasks (hours)
- Batch processing
- Don't need real-time interaction

**Setup:**
```bash
# On Mac: Create a script for the home PC to run
cat > embedding_job.py <<'EOF'
import ray
import numpy as np
from transformers import AutoModel, AutoTokenizer

ray.init(address="auto")  # Auto-connect to local Ray cluster

@ray.remote(num_gpus=1)
def extract_embeddings():
    # Your embedding extraction code
    # This can run for hours
    pass

result = ray.get(extract_embeddings.remote())
np.save("/path/to/results.npy", result)
EOF

# Submit job to home PC (via SSH or Ray Jobs API)
ray job submit --address http://home-pc:8265 \
  --working-dir . \
  -- python embedding_job.py

# Disconnect Mac - job keeps running!
# Check status later:
ray job status <job-id>
```

**Benefits:**
- Disconnect after submission
- Job continues running
- Better for long tasks

---

## Alternative Simpler Solutions

Before investing time in Ray setup, consider these simpler approaches:

### 1. Just Use the M4 Mac

You already said it works fine. The M4 has:
- Strong CPU performance
- Unified memory (up to 128GB on Max models)
- No network complexity
- Easy debugging

**When to use:** Most of the time, honestly.

### 2. SSH + tmux/screen

```bash
# From Mac
ssh home-pc

# Start tmux session
tmux new -s bert-experiments

# Run your experiment
cd ~/bert-interpretability-illusion
uv run python extract_embeddings.py

# Detach: Ctrl-B then D
# Job keeps running!

# Later: reattach
tmux attach -t bert-experiments
```

**Benefits:**
- Simple, no new tools
- Reliable
- Easy to check progress

### 3. VS Code Remote-SSH

1. Install "Remote - SSH" extension in VS Code
2. Connect to home PC
3. Edit files locally, execute remotely
4. Integrated terminal runs on home PC

**Benefits:**
- Familiar IDE
- No mental model change
- Terminal is already remote

### 4. Jupyter Notebook on Home PC

```bash
# On home PC
uv run jupyter lab --no-browser --port=8888

# On Mac (SSH tunnel)
ssh -L 8888:localhost:8888 home-pc

# Open http://localhost:8888 on Mac
# Runs on home PC's GPU!
```

**Benefits:**
- Interactive
- Visualizations work
- Can disconnect (with caveats)

---

## Practical Getting Started: Batch Job Pattern

Okay, let's assume you want to actually use Ray as a learning exercise and to build good patterns for future scaling. Here's a practical, minimal setup for your Mac → home PC workflow.

### The Goal

Submit embedding extraction jobs from your Mac that:
- Run on your home PC's 4070 Ti Super
- Handle datasets properly
- Cache/store results
- Let you disconnect after submission

### Step-by-Step Setup

#### 1. Install Ray on Both Machines

```bash
# On home PC
pip install "ray[default]"

# On Mac
pip install "ray[default]"

# Or with uv (recommended)
uv add "ray[default]"
```

#### 2. Start Ray Cluster on Home PC

```bash
# On home PC
ray start --head --dashboard-port=8265

# You'll see output like:
# Ray runtime started.
# Dashboard URL: http://127.0.0.1:8265
```

**Note the dashboard URL** - you can use SSH tunnel to view it from Mac:
```bash
# On Mac
ssh -L 8265:localhost:8265 user@home-pc
# Then open http://localhost:8265 in browser
```

#### 3. Create Your First Ray Job Script

Create `jobs/extract_embeddings.py`:

```python
"""
Ray job for extracting BERT embeddings.
This script runs on the home PC.
"""
import ray
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import json
from datetime import datetime

# Connect to the local Ray cluster (auto-detect when running as job)
ray.init(address="auto")

@ray.remote(num_gpus=1)
class BERTEmbedder:
    """GPU actor for BERT embedding extraction."""

    def __init__(self):
        print("Loading BERT model on GPU...")
        self.device = torch.device("cuda")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model.eval()
        print(f"Model loaded on {torch.cuda.get_device_name(0)}")

    def embed_batch(self, sentences):
        """Extract embeddings for a batch of sentences."""
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Return [CLS] token embeddings
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def main():
    # Configuration
    output_dir = Path("/path/to/results")  # Adjust for your home PC
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 32

    # Example: Load your sentences
    # In real code, use Ray Data (see below)
    sentences = [
        "This is a test sentence.",
        "Another example here.",
        # ... your actual data
    ]

    print(f"Processing {len(sentences)} sentences...")

    # Create embedder actor
    embedder = BERTEmbedder.remote()

    # Process in batches
    all_embeddings = []
    futures = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        future = embedder.embed_batch.remote(batch)
        futures.append(future)

    # Collect results
    print(f"Submitted {len(futures)} batches. Waiting for results...")
    batch_results = ray.get(futures)
    all_embeddings = np.vstack(batch_results)

    # Save results
    output_file = output_dir / f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
    np.save(output_file, all_embeddings)

    # Save metadata
    metadata = {
        "num_sentences": len(sentences),
        "embedding_dim": all_embeddings.shape[1],
        "timestamp": datetime.now().isoformat(),
        "model": "bert-base-uncased",
        "batch_size": batch_size
    }

    metadata_file = output_file.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved embeddings to {output_file}")
    print(f"✓ Shape: {all_embeddings.shape}")

    ray.shutdown()

if __name__ == "__main__":
    main()
```

#### 4. Submit Job from Mac

```bash
# From Mac - submit to home PC
# Assumes you have SSH tunnel or network access

ray job submit \
  --address http://home-pc:8265 \
  --working-dir . \
  -- python jobs/extract_embeddings.py

# You'll get a job ID like: raysubmit_xxxxx
# Job runs on home PC - you can disconnect!
```

#### 5. Check Job Status

```bash
# Check status
ray job status raysubmit_xxxxx --address http://home-pc:8265

# View logs
ray job logs raysubmit_xxxxx --address http://home-pc:8265

# List all jobs
ray job list --address http://home-pc:8265
```

### Using Ray Data for Dataset Handling

Ray Data is better for large datasets. Here's how to integrate it:

```python
"""
Using Ray Data for scalable dataset loading and processing.
"""
import ray
from ray.data import read_json, read_parquet, from_items
import numpy as np

ray.init(address="auto")

# Option 1: Load from files (JSON, Parquet, CSV, etc.)
ds = ray.data.read_json("s3://bucket/sentences.json")
# Or local files:
# ds = ray.data.read_json("/path/to/sentences.json")

# Option 2: Create from Python objects
sentences_data = [
    {"text": "First sentence", "dataset": "QQP", "idx": 0},
    {"text": "Second sentence", "dataset": "QQP", "idx": 1},
    # ...
]
ds = ray.data.from_items(sentences_data)

# Option 3: Load HuggingFace datasets
from datasets import load_dataset as hf_load_dataset

def load_qqp_dataset():
    """Load QQP dataset from HuggingFace."""
    dataset = hf_load_dataset("glue", "qqp", split="train")
    # Convert to Ray Data format
    items = [
        {"text": item["question1"], "dataset": "QQP", "idx": i}
        for i, item in enumerate(dataset)
    ]
    return ray.data.from_items(items)

ds = load_qqp_dataset()

# Define embedding function for Ray Data
class BERTEmbedding:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model.eval()

    def __call__(self, batch):
        """Process a batch of rows."""
        texts = batch["text"]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Add embeddings to batch
        batch["embedding"] = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return batch

# Apply embedding transformation
# This automatically parallelizes across available GPUs
ds_with_embeddings = ds.map_batches(
    BERTEmbedding,
    batch_size=32,
    num_gpus=1,  # Use 1 GPU per worker
    concurrency=1  # Number of parallel workers (1 GPU = 1 worker for you)
)

# Save results
ds_with_embeddings.write_parquet("/path/to/output/embeddings/")

print("✓ Embeddings saved!")
```

**Benefits of Ray Data:**
- Automatically handles batching
- Streaming processing (doesn't load all data into memory)
- Built-in progress tracking
- Easy to switch between local files, S3, etc.
- Fault tolerance built-in

### Result Storage and Caching Patterns

#### Pattern 1: Local Filesystem (Simplest)

```python
# In your job script
from pathlib import Path
import numpy as np

# Define output structure
output_dir = Path("/path/to/results")
output_dir.mkdir(parents=True, exist_ok=True)

# Save embeddings by dataset
for dataset_name, embeddings in results.items():
    output_file = output_dir / f"{dataset_name}_embeddings.npy"
    np.save(output_file, embeddings)
    print(f"Saved {dataset_name}: {embeddings.shape}")
```

**Access from Mac:**
```bash
# Use scp to fetch results
scp home-pc:/path/to/results/*.npy ./local_results/

# Or mount home PC filesystem
# (on Mac with SSHFS or SMB share)
```

#### Pattern 2: Ray Object Store (For Inter-Job Sharing)

```python
import ray

# Store large objects in Ray's shared memory
embeddings_ref = ray.put(large_embeddings_array)

# Multiple tasks can access without copying
@ray.remote
def analyze_embeddings(embeddings_ref):
    embeddings = ray.get(embeddings_ref)  # Zero-copy on same node
    return compute_stats(embeddings)

futures = [analyze_embeddings.remote(embeddings_ref) for _ in range(10)]
results = ray.get(futures)
```

**Note:** Object store is in-memory and cleared when cluster stops. Use for intermediate results, not final storage.

#### Pattern 3: Persistent Caching with Checkpoints

```python
"""
Checkpoint pattern for long-running jobs.
Allows resuming if job fails.
"""
import pickle
from pathlib import Path

class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data, name):
        """Save checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{name}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Checkpoint saved: {checkpoint_file}")

    def load(self, name):
        """Load checkpoint if exists."""
        checkpoint_file = self.checkpoint_dir / f"{name}.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None

    def exists(self, name):
        """Check if checkpoint exists."""
        checkpoint_file = self.checkpoint_dir / f"{name}.pkl"
        return checkpoint_file.exists()

# Usage in job
checkpoint_mgr = CheckpointManager("/path/to/checkpoints")

# Check if we can resume
if checkpoint_mgr.exists("embeddings_progress"):
    progress = checkpoint_mgr.load("embeddings_progress")
    print(f"Resuming from batch {progress['last_batch']}")
    start_batch = progress['last_batch']
else:
    start_batch = 0

# Process batches
for batch_idx in range(start_batch, total_batches):
    # Process batch
    result = process_batch(batch_idx)

    # Checkpoint every 100 batches
    if (batch_idx + 1) % 100 == 0:
        checkpoint_mgr.save({
            'last_batch': batch_idx + 1,
            'results_so_far': accumulated_results
        }, "embeddings_progress")
```

#### Pattern 4: Ray Data with Caching

```python
# Ray Data automatically handles caching
import ray

ds = ray.data.read_json("/path/to/data.json")

# Compute embeddings (expensive)
ds_with_embeddings = ds.map_batches(BERTEmbedding, batch_size=32, num_gpus=1)

# Write to disk for future use
ds_with_embeddings.write_parquet("/path/to/cached_embeddings/")

# Later, in another job/session:
# Load from cache instead of recomputing
cached_ds = ray.data.read_parquet("/path/to/cached_embeddings/")

# Continue with analysis
results = cached_ds.map_batches(AnalyzeEmbeddings, batch_size=100)
```

### Recommended Workflow for Your Use Case

Here's what I'd suggest for your Mac → home PC setup:

```python
"""
Complete workflow: Mac submits job to home PC.
"""

# 1. PREPARE DATA (on Mac or home PC)
# Create a simple JSON file with your sentences
import json
from pathlib import Path

sentences_file = Path("data/sentences.json")
sentences_file.parent.mkdir(exist_ok=True)

data = [
    {"text": sentence, "dataset": dataset_name, "idx": idx}
    for dataset_name, sentences in your_datasets.items()
    for idx, sentence in enumerate(sentences)
]

with open(sentences_file, 'w') as f:
    json.dump(data, f)

# 2. UPLOAD DATA TO HOME PC (if prepared on Mac)
# scp data/sentences.json home-pc:/path/to/bert-project/data/

# 3. CREATE JOB SCRIPT (save as jobs/run_embeddings.py on home PC)
"""
(see complete script below)
"""

# 4. SUBMIT FROM MAC
# ray job submit --address http://home-pc:8265 \
#   --working-dir /path/to/bert-project \
#   -- uv run python jobs/run_embeddings.py

# 5. MONITOR
# ray job logs <job-id> --address http://home-pc:8265 --follow

# 6. RETRIEVE RESULTS
# scp home-pc:/path/to/results/*.npy ./results/
```

**Complete Job Script:**

```python
# jobs/run_embeddings.py
import ray
from ray.data import read_json
import numpy as np
from pathlib import Path
import json
from datetime import datetime

@ray.remote(num_gpus=1)
class BERTEmbedder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device("cuda")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model.eval()

    def __call__(self, batch):
        texts = batch["text"]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        batch["embedding"] = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return batch

def main():
    ray.init(address="auto")

    # Load data
    ds = read_json("data/sentences.json")

    # Apply embeddings
    ds_embedded = ds.map_batches(
        BERTEmbedder,
        batch_size=32,
        num_gpus=1,
        concurrency=1
    )

    # Save by dataset
    output_dir = Path("results/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by dataset and save separately
    for dataset_name in ["QQP", "QNLI", "Wiki", "BookCorpus"]:
        dataset_rows = ds_embedded.filter(lambda row: row["dataset"] == dataset_name)

        # Collect embeddings
        rows = dataset_rows.take_all()
        embeddings = np.array([row["embedding"] for row in rows])

        # Save
        output_file = output_dir / f"{dataset_name}_embeddings.npy"
        np.save(output_file, embeddings)
        print(f"✓ {dataset_name}: {embeddings.shape}")

    print("✓ All embeddings saved!")
    ray.shutdown()

if __name__ == "__main__":
    main()
```

This gives you:
- ✅ Submit from Mac, run on home PC
- ✅ Proper dataset handling
- ✅ GPU utilization
- ✅ Structured result storage
- ✅ Easy to scale later (just change `concurrency` when you add more GPUs)

---

## If You Really Do Need Scale...

If you later find yourself needing true distributed computing (multiple GPUs, multiple machines, hundreds of experiments), the original massive Ray implementation is below for reference.

The scenarios where this becomes relevant:
- Hyperparameter sweeps (100s of model configurations)
- Scaling to millions of sentences
- Multiple GPUs across multiple machines
- Production deployment with auto-scaling

### Quick Reference: Distributed Ray Patterns

*See "Appendix: Advanced Distributed Patterns" section below for the full implementation guide.*

---

## References

### Ray Documentation - Remote Execution
- [Ray Client Documentation](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/ray-client.html)
- [Ray Jobs API](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html)
- [Ray Core Walkthrough](https://docs.ray.io/en/latest/ray-core/walkthrough.html)

### Simpler Alternatives
- [tmux basics](https://github.com/tmux/tmux/wiki/Getting-Started)
- [VS Code Remote-SSH](https://code.visualstudio.com/docs/remote/ssh)
- [Tailscale](https://tailscale.com/) - Easy VPN for home lab access
- [Zerotier](https://www.zerotier.com/) - Alternative to Tailscale

---

# Appendix: Advanced Distributed Patterns

*The following sections describe large-scale distributed Ray implementations for reference. These are likely overkill for a 2-machine setup but useful if your needs grow.*

## Original Ray Framework Overview

### Core Primitives

Ray provides three fundamental abstractions for distributed computing:

#### 1. Tasks (@ray.remote functions)

**Stateless parallel functions** that execute on remote workers.

```python
import ray

@ray.remote
def compute_activations(neuron_idx, embeddings):
    """Compute activations for a single neuron across all embeddings."""
    return embeddings @ neuron_direction[neuron_idx]

# Execute 25 tasks in parallel
futures = [compute_activations.remote(i, emb) for i in range(25)]
results = ray.get(futures)  # Blocking wait for all results
```

**When to Use:**
- Independent, stateless computations
- Short-lived operations (seconds to minutes)
- No shared state between invocations

#### 2. Actors (@ray.remote classes)

**Stateful workers** that maintain internal state across multiple method calls.

```python
@ray.remote(num_gpus=1)
class BERTEmbedder:
    def __init__(self):
        # Load model once, reuse across calls
        self.model = AutoModel.from_pretrained("bert-base-uncased").cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def embed_batch(self, sentences):
        """Process a batch of sentences. Model stays loaded."""
        inputs = self.tokenizer(sentences, return_tensors="pt",
                               padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs.to("cuda"))
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# Create 8 actors, one per GPU
embedders = [BERTEmbedder.remote() for _ in range(8)]

# Reuse actors across batches (model stays loaded)
futures = [embedders[i % 8].embed_batch.remote(batch)
           for i, batch in enumerate(sentence_batches)]
```

**When to Use:**
- Long-lived stateful computations
- Model loading (avoid repeated initialization)
- Resource management (GPU workers, database connections)

#### 3. Objects (Ray Object Store)

**Distributed shared memory** for zero-copy data sharing between workers.

```python
# Store large data in object store
embeddings_ref = ray.put(large_embedding_array)  # ~300MB

# All workers can access without copying
@ray.remote
def analyze_neuron(neuron_idx, embeddings_ref):
    embeddings = ray.get(embeddings_ref)  # Zero-copy if on same node
    return compute_analysis(neuron_idx, embeddings)

# 100 tasks share the same embeddings efficiently
results = ray.get([
    analyze_neuron.remote(i, embeddings_ref)
    for i in range(100)
])
```

**When to Use:**
- Sharing large data across multiple tasks
- Avoiding redundant data transfers
- Memory-efficient distributed computation

### Ray Libraries for ML Workloads

#### Ray Data: Distributed Data Processing

**Scalable ETL and batch inference** for large datasets.

```python
import ray

# Create dataset from 625K sentences
ds = ray.data.from_items([
    {"text": sentence, "dataset": dataset_name, "idx": i}
    for dataset_name, sentences in all_datasets.items()
    for i, sentence in enumerate(sentences)
])

# Define preprocessing class
class BERTPreprocessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def __call__(self, batch):
        texts = batch["text"]
        return self.tokenizer(texts, padding=True, truncation=True,
                            max_length=512, return_tensors="np")

# Apply preprocessing in parallel across cluster
ds = ds.map_batches(BERTPreprocessor, batch_size=256)

# Inference with GPU batching
class BERTInference:
    def __init__(self):
        self.model = AutoModel.from_pretrained("bert-base-uncased").cuda()

    def __call__(self, batch):
        with torch.no_grad():
            outputs = self.model(**{k: torch.tensor(v).cuda()
                                   for k, v in batch.items()})
        batch["embedding"] = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return batch

# Distributed GPU inference
ds = ds.map_batches(BERTInference, batch_size=32,
                    num_gpus=1, concurrency=8)

# Save to distributed storage
ds.write_parquet("s3://bucket/embeddings/")
```

**Key Features:**
- Automatic batching and parallelization
- Built-in GPU support with `num_gpus` parameter
- Streaming processing (doesn't load entire dataset into memory)
- Integration with PyTorch, TensorFlow, Hugging Face

#### Ray Train: Distributed Training

**Unified API for distributed model training** (not heavily needed for our inference-focused workload, but useful for future model fine-tuning).

```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def train_func(config):
    # Standard PyTorch training loop
    model = create_model()
    train_dataset = load_data()
    # ... training code ...

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(
        num_workers=8,      # 8 distributed workers
        use_gpu=True,       # 1 GPU per worker
        resources_per_worker={"GPU": 1}
    )
)
trainer.fit()
```

---

## Current Computational Bottlenecks

### Workload Analysis

From the codebase exploration, here's a detailed breakdown of computational bottlenecks:

| Component | Current Time | Hardware | Bottleneck Type | Parallelization Potential |
|-----------|--------------|----------|-----------------|---------------------------|
| **Embedding Extraction** | 2-3 hours | GPU (RTX 3090) | GPU throughput | ✅ Multi-GPU sharding |
| **Neuron Analysis** | 30 minutes | CPU | Sequential loops | ✅ Embarrassingly parallel |
| **Locality Computation** | 1-2 hours | CPU + RAM | KNN queries | ✅ Distributed KNN |
| **Monotonicity Analysis** | 3-4 hours | CPU | Token counting | ✅ Task parallelism |
| **UMAP Visualization** | 20 minutes | CPU | Dimensionality reduction | ⚠️ Limited parallelism |

### Detailed Bottleneck Breakdown

#### 1. Embedding Extraction (GPU-Bound)

**Current Implementation:**
```python
def generate_all_embeddings(sentences_df, model, tokenizer, device, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(sentences_df), batch_size)):
        batch = sentences_df.iloc[i:i+batch_size]
        # Tokenize and forward pass
        # ...
    return np.vstack(embeddings)
```

**Bottleneck:**
- Single GPU processes ~20,000 batches sequentially
- Each batch: 32 sentences × 512 tokens × BERT forward pass
- 625K sentences take ~2-3 hours on RTX 3090

**Why It's Slow:**
- Only one GPU utilized (if multiple GPUs available)
- No pipeline parallelism
- Synchronous processing

#### 2. Neuron Analysis (CPU-Bound, Embarrassingly Parallel)

**Current Pattern:**
```python
for neuron_idx in range(25):
    for dataset_name in datasets:
        activations = embeddings @ neuron_directions[neuron_idx]
        top_indices = np.argsort(activations)[-10:]
        # Store results
```

**Bottleneck:**
- 100 independent tasks (25 neurons × 4 datasets)
- Each task takes ~18 seconds
- Runs sequentially: ~30 minutes total

**Why It's Slow:**
- No parallelization across neurons or datasets
- Perfect candidate for task parallelism

#### 3. Locality Computation (Memory-Bound)

**Current Pattern:**
```python
for each neuron-dataset pair:
    top_sentences = get_top_activating(neuron, dataset)
    for sentence in top_sentences:
        neighbors = find_knn(sentence, embeddings, k=10)
        compute_dot_products(sentence, neighbors)
        # Histogram construction
```

**Bottleneck:**
- KNN queries on 625K × 768 embedding space
- 100 neuron-dataset pairs × 10 top sentences × 10 neighbors
- ~1-2 hours on 32-core CPU

**Why It's Slow:**
- KNN queries are memory-intensive
- Sequential processing of neuron-dataset pairs

#### 4. Monotonicity Analysis (CPU-Bound)

**Current Pattern:**
```python
for neuron in range(25):
    for dataset in range(4):
        for token in common_tokens (915):
            activations = get_activations(neuron, dataset)
            quintiles = split_into_quintiles(activations)
            counts = count_token_frequency_per_quintile(token, quintiles)
            check_monotonicity(counts)
```

**Bottleneck:**
- 91,500 token-neuron-dataset combinations
- Each requires tokenization + counting (~10ms)
- ~15 minutes of pure computation, 3-4 hours with overhead

**Why It's Slow:**
- Triple nested loop with no parallelization
- Tokenization repeated for each combination

---

## Ray Architecture Proposal

### Overall System Design

```
┌─────────────────────────────────────────────────────────────┐
│                       Ray Cluster                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  Head Node                              │ │
│  │  - Orchestration                                       │ │
│  │  - Result aggregation                                  │ │
│  │  - Dashboard/monitoring                                │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│  ┌─────────────────────────┼────────────────────────────┐  │
│  │                         │                             │  │
│  ▼                         ▼                             ▼  │
│  ┌──────────┐      ┌──────────┐                ┌──────────┐│
│  │ GPU Node │      │ GPU Node │      ...       │ GPU Node ││
│  │          │      │          │                │          ││
│  │ - BERT   │      │ - BERT   │                │ - BERT   ││
│  │   Actor  │      │   Actor  │                │   Actor  ││
│  │ - 1 GPU  │      │ - 1 GPU  │                │ - 1 GPU  ││
│  └──────────┘      └──────────┘                └──────────┘│
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              CPU Worker Pool (32-64 cores)           │  │
│  │  - Neuron analysis tasks                            │  │
│  │  - Locality computation                             │  │
│  │  - Monotonicity analysis                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Ray Object Store (Shared Memory)           │  │
│  │  - Embeddings (300MB × 4 datasets)                  │  │
│  │  - BERT model weights (440MB, shared)               │  │
│  │  - Intermediate results                             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Mapping to Ray Primitives

| Experimental Component | Ray Primitive | Resource Spec | Concurrency |
|------------------------|---------------|---------------|-------------|
| **Embedding Extraction** | Ray Actors (stateful) | `num_gpus=1` | 8 actors |
| **Neuron Analysis** | Ray Tasks | `num_cpus=1` | 100 tasks |
| **Locality Computation** | Ray Tasks + Object Store | `num_cpus=2` | 50 tasks |
| **Monotonicity Analysis** | Ray Tasks | `num_cpus=1` | 100 tasks |
| **Result Aggregation** | Ray Tasks | `num_cpus=4` | 1 task |

---

## Implementation Details by Component

### 1. Distributed Embedding Extraction

**Goal:** Extract 625K embeddings using multiple GPUs in parallel.

**Strategy:** Use Ray Actors to create persistent GPU workers that avoid repeated model loading.

#### Implementation

```python
import ray
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from typing import List, Dict
from itertools import cycle

@ray.remote(num_gpus=1)
class BERTEmbedder:
    """Stateful BERT embedding extractor that runs on a single GPU.

    The model is loaded once during initialization and reused across
    multiple batch inference calls, avoiding expensive re-initialization.
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """Initialize BERT model and tokenizer on GPU."""
        self.device = torch.device("cuda")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode

        print(f"BERTEmbedder initialized on {self.device}")

    def embed_batch(self, sentences: List[str]) -> np.ndarray:
        """Extract [CLS] embeddings for a batch of sentences.

        Args:
            sentences: List of input sentences

        Returns:
            numpy array of shape (batch_size, 768) containing embeddings
        """
        # Tokenize
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Move to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass (no gradient computation needed)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract [CLS] token embeddings (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return cls_embeddings

    def get_device_info(self) -> Dict:
        """Return GPU device information for debugging."""
        return {
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated": torch.cuda.memory_allocated(0) / 1e9,
            "gpu_memory_reserved": torch.cuda.memory_reserved(0) / 1e9
        }


def extract_all_embeddings_distributed(
    datasets: Dict[str, List[str]],
    num_gpus: int = 8,
    batch_size: int = 32,
    output_dir: str = "./embeddings/"
) -> Dict[str, np.ndarray]:
    """Extract embeddings for all datasets using distributed GPU workers.

    Args:
        datasets: Dictionary mapping dataset names to lists of sentences
        num_gpus: Number of GPU actors to create
        batch_size: Batch size for inference
        output_dir: Directory to save embeddings

    Returns:
        Dictionary mapping dataset names to embedding arrays
    """
    import os
    from tqdm import tqdm

    # Initialize Ray (if not already initialized)
    if not ray.is_initialized():
        ray.init()

    # Create GPU actors (one per GPU)
    print(f"Initializing {num_gpus} BERT embedding actors...")
    embedders = [BERTEmbedder.remote() for _ in range(num_gpus)]

    # Verify GPU allocation
    device_infos = ray.get([actor.get_device_info.remote() for actor in embedders])
    for i, info in enumerate(device_infos):
        print(f"Actor {i}: {info}")

    all_embeddings = {}

    # Process each dataset
    for dataset_name, sentences in datasets.items():
        print(f"\nProcessing {dataset_name}: {len(sentences)} sentences")

        # Divide sentences into batches
        batches = [
            sentences[i:i+batch_size]
            for i in range(0, len(sentences), batch_size)
        ]

        # Distribute batches across GPU actors using round-robin
        futures = []
        for batch_idx, batch in enumerate(batches):
            # Cycle through actors to distribute load evenly
            actor = embedders[batch_idx % num_gpus]
            future = actor.embed_batch.remote(batch)
            futures.append(future)

        # Collect results with progress bar
        print(f"Collecting embeddings from {len(futures)} batches...")
        batch_embeddings = []
        for future in tqdm(futures):
            batch_embeddings.append(ray.get(future))

        # Concatenate all batches
        dataset_embeddings = np.vstack(batch_embeddings)
        all_embeddings[dataset_name] = dataset_embeddings

        # Save to disk
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{dataset_name}_embeddings.npy")
        np.save(output_path, dataset_embeddings)
        print(f"Saved {dataset_name} embeddings to {output_path}")
        print(f"Shape: {dataset_embeddings.shape}")

    return all_embeddings


# Example usage
if __name__ == "__main__":
    # Load your datasets (example structure)
    datasets = {
        "QQP": load_qqp_sentences(),      # ~100K sentences
        "QNLI": load_qnli_sentences(),    # ~100K sentences
        "Wiki": load_wiki_sentences(),    # ~200K sentences
        "BookCorpus": load_book_sentences()  # ~200K sentences
    }

    # Extract embeddings using 8 GPUs
    embeddings = extract_all_embeddings_distributed(
        datasets=datasets,
        num_gpus=8,
        batch_size=32,
        output_dir="./embeddings/"
    )
```

#### Performance Analysis

**Baseline (Single GPU):**
- 625K sentences ÷ 32 batch size = 19,531 batches
- ~0.35 seconds per batch (forward pass + transfer)
- Total: ~6,836 seconds ≈ **114 minutes (1.9 hours)**

**Ray Distributed (8 GPUs):**
- 19,531 batches ÷ 8 GPUs = 2,441 batches per GPU
- ~2,441 × 0.35 = 854 seconds ≈ **14 minutes**
- **Speedup: ~8x** (near-linear scaling)

**Memory Requirements:**
- Each actor: ~2GB GPU memory (BERT model + batch)
- Total: 8 GPUs × 2GB = 16GB GPU memory across cluster

---

### 2. Parallel Neuron Analysis

**Goal:** Analyze 25 neurons across 4 datasets to find top-activating sentences.

**Strategy:** Use Ray Tasks for embarrassingly parallel computation.

#### Implementation

```python
import ray
import numpy as np
from typing import Dict, List, Tuple

@ray.remote
def analyze_neuron_on_dataset(
    neuron_idx: int,
    dataset_name: str,
    embeddings: np.ndarray,
    neuron_direction: np.ndarray,
    top_k: int = 10
) -> Dict:
    """Analyze a single neuron on a single dataset.

    Args:
        neuron_idx: Index of the neuron being analyzed
        dataset_name: Name of the dataset
        embeddings: Embedding matrix (N × 768)
        neuron_direction: Direction vector (768,)
        top_k: Number of top activating sentences to return

    Returns:
        Dictionary with analysis results
    """
    # Compute activations (dot product of embeddings with neuron direction)
    activations = embeddings @ neuron_direction

    # Find top-k activating sentence indices
    top_indices = np.argsort(activations)[-top_k:][::-1]
    top_activations = activations[top_indices]

    return {
        "neuron_idx": neuron_idx,
        "dataset_name": dataset_name,
        "top_indices": top_indices.tolist(),
        "top_activations": top_activations.tolist(),
        "mean_activation": float(np.mean(activations)),
        "std_activation": float(np.std(activations))
    }


def run_neuron_analysis_distributed(
    embeddings_dict: Dict[str, np.ndarray],
    neuron_directions: np.ndarray,
    selected_neurons: List[int],
    top_k: int = 10
) -> List[Dict]:
    """Run neuron analysis in parallel across all neuron-dataset pairs.

    Args:
        embeddings_dict: Dictionary mapping dataset names to embeddings
        neuron_directions: Array of neuron direction vectors (768, 768)
        selected_neurons: List of neuron indices to analyze
        top_k: Number of top sentences to extract per neuron-dataset pair

    Returns:
        List of analysis results
    """
    if not ray.is_initialized():
        ray.init()

    # Store embeddings in object store for efficient sharing
    embeddings_refs = {
        name: ray.put(emb) for name, emb in embeddings_dict.items()
    }

    # Create all tasks (25 neurons × 4 datasets = 100 tasks)
    futures = []
    for neuron_idx in selected_neurons:
        neuron_dir = neuron_directions[:, neuron_idx]
        for dataset_name, emb_ref in embeddings_refs.items():
            future = analyze_neuron_on_dataset.remote(
                neuron_idx=neuron_idx,
                dataset_name=dataset_name,
                embeddings=emb_ref,
                neuron_direction=neuron_dir,
                top_k=top_k
            )
            futures.append(future)

    print(f"Launched {len(futures)} parallel analysis tasks")

    # Collect all results
    results = ray.get(futures)

    return results


# Example usage
if __name__ == "__main__":
    # Load embeddings from disk
    embeddings_dict = {
        "QQP": np.load("./embeddings/QQP_embeddings.npy"),
        "QNLI": np.load("./embeddings/QNLI_embeddings.npy"),
        "Wiki": np.load("./embeddings/Wiki_embeddings.npy"),
        "BookCorpus": np.load("./embeddings/BookCorpus_embeddings.npy")
    }

    # Generate random neuron directions
    np.random.seed(42)
    neuron_directions = np.eye(768)  # Use basis vectors or random directions

    # Select 25 random neurons
    selected_neurons = np.random.choice(768, size=25, replace=False)

    # Run distributed analysis
    results = run_neuron_analysis_distributed(
        embeddings_dict=embeddings_dict,
        neuron_directions=neuron_directions,
        selected_neurons=selected_neurons.tolist(),
        top_k=10
    )

    # Save results
    import json
    with open("neuron_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
```

#### Performance Analysis

**Baseline (Sequential):**
- 100 tasks × 18 seconds each = **1,800 seconds (30 minutes)**

**Ray Distributed (32 CPU cores):**
- 100 tasks ÷ 32 cores = ~3 tasks per core
- 3 × 18 seconds = **54 seconds**
- **Speedup: ~33x**

---

### 3. Distributed Locality Score Computation

**Goal:** Compute locality scores for top-activating sentences using KNN.

**Strategy:** Use Ray Tasks with shared embeddings in object store.

#### Implementation

```python
import ray
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List

@ray.remote
def compute_locality_score(
    neuron_idx: int,
    dataset_name: str,
    top_indices: List[int],
    embeddings_ref,  # Ray object reference
    k_neighbors: int = 10,
    num_random: int = 100
) -> Dict:
    """Compute locality score for a neuron-dataset pair.

    Locality score measures whether top-activating sentences cluster
    together in embedding space or are scattered randomly.

    Args:
        neuron_idx: Neuron index
        dataset_name: Dataset name
        top_indices: Indices of top-activating sentences
        embeddings_ref: Ray object reference to embeddings
        k_neighbors: Number of neighbors for KNN
        num_random: Number of random sentences for baseline

    Returns:
        Dictionary with locality metrics
    """
    # Get embeddings from object store
    embeddings = ray.get(embeddings_ref)

    # Extract top-activating sentence embeddings
    top_embeddings = embeddings[top_indices]

    # Fit KNN on all embeddings
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, metric='cosine')
    nbrs.fit(embeddings)

    # Find neighbors of top-activating sentences
    distances, indices = nbrs.kneighbors(top_embeddings)

    # Compute pairwise similarities within top sentences
    top_similarities = []
    for i in range(len(top_indices)):
        for j in range(i+1, len(top_indices)):
            sim = np.dot(top_embeddings[i], top_embeddings[j])
            top_similarities.append(sim)

    # Compute similarities with neighbors
    neighbor_similarities = []
    for i, neighbor_idx in enumerate(indices):
        for j in neighbor_idx[1:]:  # Skip self
            sim = np.dot(top_embeddings[i], embeddings[j])
            neighbor_similarities.append(sim)

    # Compute similarities with random sentences (baseline)
    random_indices = np.random.choice(len(embeddings), size=num_random, replace=False)
    random_embeddings = embeddings[random_indices]
    random_similarities = []
    for i in range(len(top_indices)):
        for j in range(num_random):
            sim = np.dot(top_embeddings[i], random_embeddings[j])
            random_similarities.append(sim)

    # Compute locality score (higher = more clustered)
    locality_score = (
        np.mean(neighbor_similarities) - np.mean(random_similarities)
    ) / np.std(random_similarities)

    return {
        "neuron_idx": neuron_idx,
        "dataset_name": dataset_name,
        "locality_score": float(locality_score),
        "top_similarity_mean": float(np.mean(top_similarities)),
        "neighbor_similarity_mean": float(np.mean(neighbor_similarities)),
        "random_similarity_mean": float(np.mean(random_similarities))
    }


def run_locality_analysis_distributed(
    neuron_analysis_results: List[Dict],
    embeddings_dict: Dict[str, np.ndarray],
    k_neighbors: int = 10
) -> List[Dict]:
    """Run locality analysis in parallel for all neuron-dataset pairs.

    Args:
        neuron_analysis_results: Results from neuron analysis
        embeddings_dict: Dictionary of embeddings per dataset
        k_neighbors: Number of neighbors for KNN

    Returns:
        List of locality score results
    """
    if not ray.is_initialized():
        ray.init()

    # Store embeddings in object store
    embeddings_refs = {
        name: ray.put(emb) for name, emb in embeddings_dict.items()
    }

    # Launch parallel locality computation tasks
    futures = []
    for result in neuron_analysis_results:
        future = compute_locality_score.remote(
            neuron_idx=result["neuron_idx"],
            dataset_name=result["dataset_name"],
            top_indices=result["top_indices"],
            embeddings_ref=embeddings_refs[result["dataset_name"]],
            k_neighbors=k_neighbors
        )
        futures.append(future)

    print(f"Launched {len(futures)} locality computation tasks")

    # Collect results
    locality_results = ray.get(futures)

    return locality_results
```

#### Performance Analysis

**Baseline (Sequential):**
- 100 neuron-dataset pairs × 60 seconds each = **6,000 seconds (100 minutes)**

**Ray Distributed (32 CPU cores):**
- 100 tasks ÷ 32 cores = ~3 tasks per core
- 3 × 60 seconds = **180 seconds (3 minutes)**
- **Speedup: ~33x**

---

### 4. Distributed Monotonicity Analysis

**Goal:** Check monotonicity of token frequencies across activation quintiles.

**Strategy:** Parallelize across neuron-dataset pairs.

#### Implementation

```python
import ray
import numpy as np
from typing import Dict, List
from collections import Counter

@ray.remote
def analyze_monotonicity_for_neuron_dataset(
    neuron_idx: int,
    dataset_name: str,
    embeddings_ref,
    neuron_direction: np.ndarray,
    sentences: List[str],
    common_tokens: List[str],
    tokenizer
) -> Dict:
    """Analyze token monotonicity for a single neuron-dataset pair.

    Args:
        neuron_idx: Neuron index
        dataset_name: Dataset name
        embeddings_ref: Ray object reference to embeddings
        neuron_direction: Neuron direction vector
        sentences: List of sentences
        common_tokens: List of common tokens to analyze
        tokenizer: Tokenizer instance

    Returns:
        Dictionary with monotonicity results
    """
    # Get embeddings
    embeddings = ray.get(embeddings_ref)

    # Compute activations
    activations = embeddings @ neuron_direction

    # Split into quintiles
    quintile_boundaries = np.percentile(activations, [20, 40, 60, 80])
    quintile_indices = [
        np.where(activations <= quintile_boundaries[0])[0],
        np.where((activations > quintile_boundaries[0]) &
                (activations <= quintile_boundaries[1]))[0],
        np.where((activations > quintile_boundaries[1]) &
                (activations <= quintile_boundaries[2]))[0],
        np.where((activations > quintile_boundaries[2]) &
                (activations <= quintile_boundaries[3]))[0],
        np.where(activations > quintile_boundaries[3])[0]
    ]

    # Tokenize sentences in each quintile
    quintile_tokens = []
    for q_indices in quintile_indices:
        q_sentences = [sentences[i] for i in q_indices]
        tokens = []
        for sent in q_sentences:
            tokens.extend(tokenizer.tokenize(sent))
        quintile_tokens.append(Counter(tokens))

    # Check monotonicity for each common token
    monotonic_tokens = []
    for token in common_tokens:
        frequencies = [q_tokens.get(token, 0) for q_tokens in quintile_tokens]

        # Check if monotonically increasing or decreasing
        is_increasing = all(frequencies[i] <= frequencies[i+1]
                          for i in range(len(frequencies)-1))
        is_decreasing = all(frequencies[i] >= frequencies[i+1]
                          for i in range(len(frequencies)-1))

        if is_increasing or is_decreasing:
            monotonic_tokens.append({
                "token": token,
                "frequencies": frequencies,
                "direction": "increasing" if is_increasing else "decreasing"
            })

    return {
        "neuron_idx": neuron_idx,
        "dataset_name": dataset_name,
        "num_monotonic_tokens": len(monotonic_tokens),
        "monotonic_tokens": monotonic_tokens,
        "total_tokens_analyzed": len(common_tokens)
    }


def run_monotonicity_analysis_distributed(
    neuron_directions: np.ndarray,
    selected_neurons: List[int],
    embeddings_dict: Dict[str, np.ndarray],
    sentences_dict: Dict[str, List[str]],
    common_tokens: List[str],
    tokenizer
) -> List[Dict]:
    """Run monotonicity analysis in parallel across all neuron-dataset pairs.

    Args:
        neuron_directions: Neuron direction vectors
        selected_neurons: List of neuron indices
        embeddings_dict: Dictionary of embeddings
        sentences_dict: Dictionary of sentences
        common_tokens: List of common tokens to analyze
        tokenizer: Tokenizer instance

    Returns:
        List of monotonicity results
    """
    if not ray.is_initialized():
        ray.init()

    # Store embeddings in object store
    embeddings_refs = {name: ray.put(emb) for name, emb in embeddings_dict.items()}

    # Store tokenizer in object store (if serializable)
    tokenizer_ref = ray.put(tokenizer)

    # Launch parallel tasks
    futures = []
    for neuron_idx in selected_neurons:
        neuron_dir = neuron_directions[:, neuron_idx]
        for dataset_name in embeddings_dict.keys():
            future = analyze_monotonicity_for_neuron_dataset.remote(
                neuron_idx=neuron_idx,
                dataset_name=dataset_name,
                embeddings_ref=embeddings_refs[dataset_name],
                neuron_direction=neuron_dir,
                sentences=sentences_dict[dataset_name],
                common_tokens=common_tokens,
                tokenizer=tokenizer_ref
            )
            futures.append(future)

    print(f"Launched {len(futures)} monotonicity analysis tasks")

    # Collect results
    results = ray.get(futures)

    return results
```

#### Performance Analysis

**Baseline (Sequential):**
- 100 neuron-dataset pairs × 120 seconds each = **12,000 seconds (200 minutes)**

**Ray Distributed (32 CPU cores):**
- 100 tasks ÷ 32 cores = ~3 tasks per core
- 3 × 120 seconds = **360 seconds (6 minutes)**
- **Speedup: ~33x**

---

### 5. Using Ray Data for End-to-End Pipeline

**Alternative approach:** Use Ray Data for a more declarative pipeline.

#### Implementation

```python
import ray
from ray.data import Dataset
from transformers import AutoTokenizer

# Initialize Ray
ray.init()

# Create dataset from all sentences
all_data = []
for dataset_name, sentences in datasets_dict.items():
    for idx, sentence in enumerate(sentences):
        all_data.append({
            "text": sentence,
            "dataset": dataset_name,
            "idx": idx
        })

ds = ray.data.from_items(all_data)

# Define BERT inference class
class BERTEmbedding:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model.eval()

    def __call__(self, batch):
        import torch
        texts = batch["text"]

        # Tokenize
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract CLS embeddings
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        batch["embedding"] = embeddings

        return batch

# Apply BERT inference with GPU batching
ds_with_embeddings = ds.map_batches(
    BERTEmbedding,
    batch_size=32,
    num_gpus=1,
    concurrency=8  # 8 parallel GPU workers
)

# Save embeddings
ds_with_embeddings.write_parquet("./embeddings/")

print("Embeddings saved successfully!")
```

**Benefits of Ray Data approach:**
- More declarative and less boilerplate
- Automatic data pipelining and batching
- Built-in fault tolerance
- Seamless integration with distributed storage (S3, HDFS)

---

## Performance Optimization Strategy

### 1. Resource Allocation

**GPU Allocation:**
```python
# Allocate 1 GPU per embedding actor
@ray.remote(num_gpus=1)
class BERTEmbedder:
    pass

# Allocate fractional GPUs if running multiple models per GPU
@ray.remote(num_gpus=0.5)
class LightweightModel:
    pass
```

**CPU Allocation:**
```python
# Allocate 2 CPUs for memory-intensive tasks
@ray.remote(num_cpus=2, memory=8*1024*1024*1024)  # 8GB RAM
def compute_knn(embeddings):
    pass
```

### 2. Object Store Management

**Best Practices:**
```python
# Store large objects once
embeddings_ref = ray.put(large_embeddings)  # 300MB

# All tasks can access without copying
futures = [process_task.remote(embeddings_ref) for _ in range(100)]

# Avoid repeated ray.put() in loops
for i in range(1000):
    # BAD: puts object 1000 times
    result = task.remote(ray.put(data))

    # GOOD: put once, reuse
    data_ref = ray.put(data)
    for i in range(1000):
        result = task.remote(data_ref)
```

### 3. Batching and Pipelining

**Optimal Batch Sizes:**
- **GPU inference:** 32-128 (depends on GPU memory)
- **CPU tasks:** 100-1000 (depends on task granularity)

**Pipelining:**
```python
# Pipeline: Process next batch while current batch is running
from ray.util.queue import Queue

result_queue = Queue()

# Producer: Submit batches
for batch in batches:
    future = process_batch.remote(batch)
    result_queue.put(future)

# Consumer: Process results as they arrive
while not result_queue.empty():
    future = result_queue.get()
    result = ray.get(future)
    save_result(result)
```

### 4. Checkpointing and Fault Tolerance

**Periodic Checkpointing:**
```python
import os

def save_checkpoint(results, checkpoint_dir, step):
    """Save intermediate results to enable recovery."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Checkpoint saved: {checkpoint_path}")

# In main loop
for i, batch in enumerate(batches):
    result = process_batch.remote(batch)
    results.append(ray.get(result))

    # Checkpoint every 100 batches
    if (i + 1) % 100 == 0:
        save_checkpoint(results, "./checkpoints", step=i+1)
```

### 5. Monitoring and Profiling

**Enable Ray Dashboard:**
```python
ray.init(dashboard_host="0.0.0.0", dashboard_port=8265)
# Access at http://localhost:8265
```

**Profile Task Execution:**
```python
import time

@ray.remote
def profiled_task(data):
    start = time.time()
    result = expensive_computation(data)
    duration = time.time() - start
    print(f"Task completed in {duration:.2f}s")
    return result
```

---

## Migration Path

### Phase 1: Single-Component Migration (Week 1)

**Goal:** Migrate embedding extraction to Ray as a proof-of-concept.

**Tasks:**
1. Set up Ray cluster (single-node or multi-node)
2. Implement `BERTEmbedder` actor
3. Test with small dataset subset (10K sentences)
4. Validate embedding quality (compare with baseline)
5. Benchmark performance (single GPU vs. multi-GPU)

**Success Metrics:**
- Embeddings match baseline (cosine similarity > 0.9999)
- 8x speedup on 8-GPU cluster

### Phase 2: Expand to Analysis Tasks (Week 2)

**Goal:** Migrate neuron analysis and locality computation.

**Tasks:**
1. Implement parallel neuron analysis with Ray Tasks
2. Implement locality score computation
3. Validate analysis results against baseline
4. Optimize object store usage

**Success Metrics:**
- Results match baseline
- 20-30x speedup on 32-core CPU cluster

### Phase 3: Full Pipeline Integration (Week 3)

**Goal:** Integrate all components into end-to-end Ray pipeline.

**Tasks:**
1. Migrate monotonicity analysis
2. Implement result aggregation
3. Add checkpointing and fault tolerance
4. Create Ray Data alternative implementation
5. Performance tuning and optimization

**Success Metrics:**
- Full pipeline runs end-to-end
- 5-8x overall speedup
- Automatic recovery from failures

### Phase 4: Production Hardening (Week 4)

**Goal:** Prepare for production use.

**Tasks:**
1. Add comprehensive logging and monitoring
2. Create deployment documentation
3. Optimize resource allocation
4. Add unit tests for Ray components
5. Create cost analysis and optimization guide

---

## Best Practices and Considerations

### 1. When to Use Ray vs. Alternatives

**Use Ray when:**
- ✅ You have embarrassingly parallel workloads (neuron analysis)
- ✅ You need to scale across multiple machines
- ✅ You have heterogeneous resources (GPUs + CPUs)
- ✅ You want to avoid infrastructure complexity (vs. Spark, Dask)

**Consider alternatives when:**
- ❌ Workload is not parallelizable (inherently sequential)
- ❌ Single machine is sufficient (premature optimization)
- ❌ Already using Spark/Dask ecosystem extensively

### 2. Object Store Best Practices

**DO:**
- ✅ Use `ray.put()` for large objects shared across tasks
- ✅ Store read-only data (embeddings, model weights)
- ✅ Monitor object store memory usage

**DON'T:**
- ❌ Store mutable objects (can cause inconsistencies)
- ❌ Put small objects (<1KB) - overhead not worth it
- ❌ Exceed object store capacity (causes eviction/spilling)

### 3. Resource Management

**GPU Management:**
```python
# Specify exact GPU allocation
@ray.remote(num_gpus=1)
class Model:
    def __init__(self):
        # Ray sets CUDA_VISIBLE_DEVICES automatically
        self.device = torch.device("cuda:0")
```

**Memory Management:**
```python
# Specify memory requirements
@ray.remote(memory=16*1024*1024*1024)  # 16GB
def memory_intensive_task(data):
    pass
```

### 4. Debugging Strategies

**Enable Verbose Logging:**
```python
import logging
logging.basicConfig(level=logging.INFO)

ray.init(logging_level=logging.INFO)
```

**Local Mode for Debugging:**
```python
# Run Ray in single-process mode for easier debugging
ray.init(local_mode=True)
```

**Inspect Task Failures:**
```python
try:
    result = ray.get(future)
except ray.exceptions.RayTaskError as e:
    print(f"Task failed: {e}")
    # Inspect stack trace
```

### 5. Cost Optimization

**Auto-Scaling Clusters:**
- Use Ray Autoscaler to automatically add/remove nodes
- Define min/max nodes based on workload
- Automatically terminate idle workers

**Spot Instances:**
- Use cloud spot instances for cost savings (70-90% cheaper)
- Ray handles preemption gracefully with task retries

**Resource Pooling:**
- Share Ray cluster across multiple experiments
- Use namespaces to isolate experiments

---

## Resource Requirements

### Minimum Setup (Proof-of-Concept)

**Single Node:**
- **GPUs:** 1 × RTX 3090 (or equivalent)
- **CPUs:** 16 cores
- **RAM:** 64GB
- **Storage:** 200GB SSD
- **Network:** 1 Gbps

**Expected Performance:**
- Embedding extraction: ~2 hours (no speedup, single GPU)
- Analysis tasks: ~2x speedup (16 cores vs. sequential)

### Recommended Setup (Production)

**Multi-Node Cluster:**

**Head Node:**
- **GPUs:** 0 (orchestration only)
- **CPUs:** 8 cores
- **RAM:** 32GB
- **Storage:** 100GB SSD

**GPU Worker Nodes (2 nodes × 4 GPUs):**
- **GPUs:** 8 × A100 40GB (or 8 × RTX 3090)
- **CPUs:** 32 cores per node
- **RAM:** 128GB per node
- **Storage:** 500GB SSD per node
- **Network:** 10 Gbps

**CPU Worker Nodes (1 node):**
- **GPUs:** 0
- **CPUs:** 64 cores
- **RAM:** 256GB
- **Storage:** 500GB SSD

**Expected Performance:**
- Embedding extraction: ~15 minutes (8 GPUs)
- Analysis tasks: ~10 minutes (64 cores + 32 cores from GPU nodes)
- **Total:** ~30 minutes (vs. 10 hours baseline) = **20x speedup**

### Cloud Cost Estimate

**AWS Configuration (On-Demand):**
- Head node: c5.2xlarge ($0.34/hour)
- GPU nodes: 2 × p3.8xlarge (4 × V100 per node, $12.24/hour each)
- CPU node: r5.16xlarge ($4.03/hour)
- **Total:** ~$28.85/hour

**Cost per Full Experiment:**
- Baseline (single machine): 10 hours × $5/hour ≈ **$50**
- Ray cluster: 0.5 hours × $28.85/hour ≈ **$14.50**
- **Savings:** $35.50 per run (70% reduction)

**With Spot Instances:**
- 0.5 hours × $8/hour ≈ **$4** (92% reduction vs. baseline)

---

## Monitoring and Debugging

### 1. Ray Dashboard

**Access:** `http://<head-node-ip>:8265`

**Key Metrics:**
- **Task timeline:** Visualize task execution and dependencies
- **Resource usage:** CPU, GPU, memory, disk utilization per node
- **Object store:** Memory usage, object sizes, evictions
- **Actor status:** Active actors, restarted actors, failed actors
- **Logs:** Centralized logging from all workers

**Screenshot of Dashboard:**
```
┌────────────────────────────────────────────────────────────────┐
│  Ray Dashboard                                                 │
├────────────────────────────────────────────────────────────────┤
│  Cluster Status: ● Running                Nodes: 4             │
│  CPUs: 128 / 128 (100%)                   GPUs: 8 / 8 (100%)  │
│  Tasks: 2,456 / 10,000 completed          Actors: 8 running   │
│  Object Store: 12.5 GB / 50 GB (25%)                          │
├────────────────────────────────────────────────────────────────┤
│  [Task Timeline] ████████████▓░░░░░░░░░░░  65% complete        │
│  [GPU Utilization] Node 1: ████████████████ 98%                │
│                    Node 2: ████████████████ 97%                │
│  [Memory] ███████░░░░░░░░░ 45% used                           │
└────────────────────────────────────────────────────────────────┘
```

### 2. Custom Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@ray.remote
def logged_task(data):
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {len(data)} items")
    result = process(data)
    logger.info(f"Completed with {len(result)} outputs")
    return result
```

### 3. Progress Tracking

```python
from tqdm import tqdm

futures = [task.remote(data) for data in batches]

# Track progress as tasks complete
results = []
for future in tqdm(futures, desc="Processing tasks"):
    result = ray.get(future)
    results.append(result)
```

### 4. Error Handling

```python
from ray.exceptions import RayTaskError, RayActorError

@ray.remote(max_retries=3)
def resilient_task(data):
    """Task with automatic retry on failure."""
    try:
        return process(data)
    except Exception as e:
        logging.error(f"Task failed: {e}")
        raise

# Handle failures gracefully
try:
    results = ray.get(futures)
except RayTaskError as e:
    print(f"Task failed after retries: {e}")
    # Fallback logic or alerting
```

---

## References

### Ray Documentation
- [Ray Core Walkthrough](https://docs.ray.io/en/latest/ray-core/walkthrough.html)
- [Ray Data Documentation](https://docs.ray.io/en/latest/data/data.html)
- [Ray Train for PyTorch](https://docs.ray.io/en/latest/train/getting-started-pytorch.html)
- [Configuring GPUs](https://docs.ray.io/en/latest/train/user-guides/using-gpus.html)
- [Batch Inference Guide](https://docs.ray.io/en/latest/data/batch_inference.html)

### Research Papers
- [Ray: A Distributed Framework for Emerging AI Applications](https://arxiv.org/abs/1712.05889) (Moritz et al., 2018)
- [An Interpretability Illusion for BERT](https://arxiv.org/abs/2104.07143) (Bolukbasi et al., 2021)

### Anyscale Resources
- [Anyscale Documentation](https://docs.anyscale.com/)
- [Anyscale Academy Tutorials](https://github.com/anyscale/academy) - Comprehensive Ray tutorials
- [What is Ray? (Anyscale Guide)](https://docs.anyscale.com/get-started/what-is-ray)
- [Ray Basics Tutorial](https://docs.anyscale.com/get-started/ray-basics)
- [Production Best Practices](https://docs.anyscale.com/platform/services/production-best-practices/)
- [Best Practices for Ray in Production (Webinar)](https://www.anyscale.com/events/2025/04/30/best-practices-ray-production)

### Community Resources
- [Ray GitHub Repository](https://github.com/ray-project/ray)
- [Ray Discourse Forum](https://discuss.ray.io/)
- [Ray Educational Materials](https://github.com/ray-project/ray-educational-materials)

### Related Tools
- **Dask:** Python parallel computing library (alternative for CPU-only workloads)
- **Apache Spark:** Distributed data processing (more heavyweight than Ray)
- **Horovod:** Distributed deep learning framework (training-focused)
- **DeepSpeed:** Microsoft's distributed training library (training-focused)

---

## Next Steps

1. **Set up Ray Cluster:**
   - Install Ray: `pip install ray[default]`
   - Initialize cluster: `ray start --head` (single node) or use Ray cluster launcher

2. **Start with Proof-of-Concept:**
   - Implement embedding extraction with 1-2 GPUs
   - Compare performance and validate results

3. **Iterate and Expand:**
   - Add analysis tasks incrementally
   - Monitor performance and optimize bottlenecks

4. **Document and Share:**
   - Record performance metrics
   - Share learnings with the team
   - Update this document with production insights

---

**Questions or Issues?**
- Open an issue in the repository
- Consult Ray documentation at https://docs.ray.io/
- Join Ray Slack community for support

**Happy Distributing! 🚀**
