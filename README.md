# Qwen-VL Setup and PoC on Windows

This repository provides instructions and a proof-of-concept (PoC) script for running the Qwen-VL model on a Windows system to describe images in exactly 5 words and rename them accordingly.

## Prerequisites
1. **Anaconda** (optional but recommended for managing dependencies).
2. **Python 3.10 or higher**.
3. A CUDA-enabled GPU (optional but speeds up processing).

## Installation

### 1. Set up the Environment
- Install [Anaconda](https://www.anaconda.com/) (optional).
- Open Anaconda Prompt and create a virtual environment:
  ```bash
  conda create -n qwen python=3.10 -y
  conda activate qwen
  ```

### 2. Install Required Packages
Run the following commands to install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/huggingface/accelerate
pip install qwen-vl-utils
pip install pillow
```

## Running the Script

1. Place `.jpg` images in the `pics` folder (create this folder if it doesn’t exist).
2. Run the script `qwen_vl_poc.py`:
   ```bash
   python qwen_vl_poc.py
   ```
3. The script will process images, generate a 5-word description for each, and rename the files accordingly.

### Folder Structure
Ensure the following structure:
```
C:.
└───pics
    └─── (Place your images here)
```

## Notes
- The script supports GPU and CPU. It will automatically detect and use GPU if available.
- Images are resized and processed to optimize memory usage.

For more details, see the comments in `qwen_vl_poc.py`.