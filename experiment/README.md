# Experiment Scripts

Experimental scripts for testing memory building and QA systems, with token usage and latency statistics.

## Script Description

- `run_locomo_experiment.py` - **Run experiments for all conversations in parallel**
- `test_memory_building.py` - Test memory building process for a single conversation
- `test_qa.py` - Test QA system for a single conversation

## Quick Start

### Run all experiments in parallel, logs will be output to logs directory

```bash
# Run memory building and QA tests for all conversations (0-9)
nohup python run_locomo_experiment.py --dataset yourpath &

# Specify model and parallelism
nohup python run_locomo_experiment.py --model gpt-4.1-mini --max-workers 5 --dataset yourpath &

# Only run memory building, skip QA tests
nohup python run_locomo_experiment.py --skip-qa --dataset yourpath &

# Run conversations in specified range
nohup python run_locomo_experiment.py --start 0 --end 4 --dataset yourpath &
```

**Parameters:**
- `--start, -s` - Starting conversation index (default: 0)
- `--end, -e` - Ending conversation index (default: 9)
- `--dataset, -d` - Dataset path (default: locomo/data/locomo10.json in project root)
- `--model, -m` - LLM model name (if not specified, uses default value from Config)
- `--max-workers, -w` - Maximum parallelism (default: 3)
- `--skip-qa` - Skip QA tests, only run memory building
- `--log-dir` - Log directory (default: experiment/logs)

**Output:**
- Log files: `experiment/logs/conv_{conversation_idx}.log` - Detailed logs for each conversation
- Summary file: `experiment/logs/summary.json` - Contains result statistics for all conversations

**Features:**
- ✅ Parallel processing of multiple conversations for improved efficiency
- ✅ Each conversation uses independent namespace, no interference
- ✅ Detailed logging and summary statistics

## Run Scripts Individually

If you need to test a specific conversation individually, you can use the following scripts:

### Memory Building Test

```bash
# Basic usage
python test_memory_building.py <conversation_idx> [options]

# Examples
python test_memory_building.py 0
python test_memory_building.py 0 --model gpt-4.1-mini
python test_memory_building.py 0 --dataset /path/to/dataset.json --model deepseek-chat
```

**Parameters:**
- `conversation_idx` - Conversation index (starting from 0, required)
- `--dataset, -d` - Dataset path (default: locomo/data/locomo10.json in project root)
- `--model, -m` - LLM model name, such as gpt-4o-mini, deepseek-chat, etc. (if not specified, uses default value from Config)
- `--skip-storage` - Skip Neo4j storage (test mode only)

### QA Test

```bash
# Basic usage
python test_qa.py <conversation_idx> [options]

# Examples
python test_qa.py 1
python test_qa.py 1 --question-idx 0 --model gpt-4.1-mini
python test_qa.py 1 --dataset /path/to/dataset.json --model deepseek-chat
python test_qa.py 1 --namespace custom_namespace
```

**Parameters:**
- `conversation_idx` - Conversation index (starting from 0, required)
- `--question-idx, -q` - Question index (if not specified, test all questions)
- `--dataset, -d` - Dataset path (default: locomo/data/locomo10.json in project root)
- `--model, -m` - LLM model name, such as gpt-4o-mini, deepseek-chat, etc. (if not specified, uses default value from Config)
- `--namespace, -n` - Namespace (default: locomo_conv_<conversation_idx>)

## Notes

1. Before running QA tests, you need to run memory building tests first to generate corresponding memory data
2. Model configuration: Need to configure `LLM_API_KEY` and `LLM_BASE_URL` in `.env` file, or set via environment variables
3. If `--model` parameter is not specified, will use default model from Config (read from environment variables or `.env` file)
