# PneumoniaMNIST experiment commands

See the [task documentation](../../problems/image_classification/README.md) for the dataset, search procedure, Ubuntu setup, endpoint configuration, and runnable examples.

| Entry point | Purpose |
| --- | --- |
| `setup_ubuntu.sh` | Install dependencies into the task-specific environment |
| `fetch_upstream.py` | Download or verify the pinned external engine |
| `prepare_data.py` | Prepare and verify fixed image subsets |
| `run_experiment.py` | Run or resume one search and its paired test evaluation |
| `run_matrix.py` | Run the model/resolution/seed matrix sequentially |
| `evaluate.py` | Evaluate a saved candidate or a train-selected pair |
| `summarize_results.py` | Calculate per-search metrics and condition-level mean/sample SD |
| `smoke_offline.py` | Verify the native workflow using local mock model responses |

The runtime adapter is in `med_evolve/`; task definitions are in `problems/image_classification/`. The engine uses disk storage. A Redis service is not required by these entry points.
