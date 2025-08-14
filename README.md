
## Prerequisites

- Python 3.10
- CUDA 12.4
- pytorch 2.4.1

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Rearank
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Generate Training Data

Use the `produce_data.py` script to generate training data for the ranking task:

```bash
python produce_data.py
```

This script will:
- Load the base dataset
- Apply elimination sort instruction formatting
- Generate training data in parquet format
- Save the processed data to the `data/` directory

### Step 2: Extract Test Samples

Extract a small subset of data (2 samples) for testing purposes:

```bash
python extract_data.py 
```

### Step 3: Test with Small Dataset

Navigate to the VERL directory and run the test script with the extracted samples:

```bash
cd verl
bash examples/grpo_trainer/test.sh
```

**Important**: Before running, you need to modify the following paths in `test.sh`:
- `data.train_files`: Update to point to your generated test data file
- `data.val_files`: Update to point to your generated test data file
- `custom_reward_function.path`: Update to point to your local `reward_func.py` path

Example modifications in `test.sh`:
```bash
# Change these lines:
data.train_files=/path/to/your/rearank_12k__default__train__elimination_sort_small_2.parquet \
data.val_files=/path/to/your/rearank_12k__default__train__elimination_sort_small_2.parquet \
custom_reward_function.path=/path/to/your/reward_func.py \
```

### Step 4: Full Training

After successful testing, run the full training pipeline:

```bash
bash examples/grpo_trainer/rerank.sh
```

**Important**: Before running, you need to modify the following paths in `rerank.sh`:
- `data.train_files`: Update to point to your full training data file
- `data.val_files`: Update to point to your validation data file
- `custom_reward_function.path`: Update to point to your local `reward_func.py` path

Example modifications in `rerank.sh`:
```bash
# Change these lines:
data.train_files=/path/to/your/full_training_data.parquet \
data.val_files=/path/to/your/validation_data.parquet \
custom_reward_function.path=/path/to/your/reward_func.py \
```

## Configuration


- **Reward Function**:
  - `custom_reward_function.path`: Path to the custom reward function
  - `custom_reward_function.name`: Function name (compute_score)

### Custom Reward Function

The `reward_func.py` file contains the custom reward function that evaluates the quality of document rankings. It includes:

- Format validation for elimination sort responses
- Relevance scoring based on ranking quality
- Penalty mechanisms for incorrect formats

## File Structure

```
Rearank/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── produce_data.py          # Data generation script
├── extract_data.py          # Data extraction script
├── reward_func.py           # Custom reward function
├── data/                    # Data directory
│   ├── combined_qrels.txt   # Query relevance labels
│   └── rearank_12k/         # Generated training data
├── verl/                    # VERL framework
│   └── examples/
│       └── grpo_trainer/
│           ├── test.sh      # Testing script
│           └── rerank.sh    # Training script
└── assets/                  # Project assets
```
