# Experiment Scripts

实验脚本用于测试记忆构建和QA系统，并统计token使用量和时延。

## 脚本说明

- `run_locomo_experiment.py` - **并行运行所有conversation的实验**
- `test_memory_building.py` - 测试单个conversation的记忆构建流程
- `test_qa.py` - 测试单个conversation的QA系统

## 快速开始

### 并行运行所有实验，日志会输出到logs目录下

```bash
# 运行所有conversation (0-9) 的记忆构建和QA测试
nohup python run_locomo_experiment.py --dataset yourpath &

# 指定模型和并行数
nohup python run_locomo_experiment.py --model gpt-4.1-mini --max-workers 5 --dataset yourpath &

# 只运行记忆构建，跳过QA测试
nohup python run_locomo_experiment.py --skip-qa --dataset yourpath &

# 运行指定范围的conversation
nohup python run_locomo_experiment.py --start 0 --end 4 --dataset yourpath &
```

**参数：**
- `--start, -s` - 起始conversation索引（默认: 0）
- `--end, -e` - 结束conversation索引（默认: 9）
- `--dataset, -d` - 数据集路径（默认：项目根目录下的locomo/data/locomo10.json）
- `--model, -m` - LLM模型名称（如果未指定，使用Config中的默认值）
- `--max-workers, -w` - 最大并行数（默认: 3）
- `--skip-qa` - 跳过QA测试，只运行记忆构建
- `--log-dir` - 日志目录（默认: experiment/logs）

**输出：**
- 日志文件：`experiment/logs/conv_{conversation_idx}.log` - 每个conversation的详细日志
- 汇总文件：`experiment/logs/summary.json` - 包含所有conversation的结果统计

**特性：**
- ✅ 并行处理多个conversation，提高效率
- ✅ 每个conversation使用独立的namespace，互不干扰
- ✅ 详细的日志记录和汇总统计

## 单独运行脚本

如果需要单独测试某个conversation，可以使用以下脚本：

### 记忆构建测试

```bash
# 基本用法
python test_memory_building.py <conversation_idx> [选项]

# 示例
python test_memory_building.py 0
python test_memory_building.py 0 --model gpt-4.1-mini
python test_memory_building.py 0 --dataset /path/to/dataset.json --model deepseek-chat
```

**参数：**
- `conversation_idx` - conversation索引（从0开始，必需）
- `--dataset, -d` - 数据集路径（默认：项目根目录下的locomo/data/locomo10.json）
- `--model, -m` - LLM模型名称，如 gpt-4o-mini, deepseek-chat 等（如果未指定，使用Config中的默认值）
- `--skip-storage` - 跳过Neo4j存储（仅测试模式）

### QA测试

```bash
# 基本用法
python test_qa.py <conversation_idx> [选项]

# 示例
python test_qa.py 1
python test_qa.py 1 --question-idx 0 --model gpt-4.1-mini
python test_qa.py 1 --dataset /path/to/dataset.json --model deepseek-chat
python test_qa.py 1 --namespace custom_namespace
```

**参数：**
- `conversation_idx` - conversation索引（从0开始，必需）
- `--question-idx, -q` - 问题索引（未指定则测试所有问题）
- `--dataset, -d` - 数据集路径（默认：项目根目录下的locomo/data/locomo10.json）
- `--model, -m` - LLM模型名称，如 gpt-4o-mini, deepseek-chat 等（如果未指定，使用Config中的默认值）
- `--namespace, -n` - 命名空间（默认：locomo_conv_<conversation_idx>）

## 注意事项

1. 运行QA测试前，需要先运行记忆构建测试生成对应的记忆数据
2. 模型配置：需要在 `.env` 文件中配置 `LLM_API_KEY` 和 `LLM_BASE_URL`，或通过环境变量设置
3. 如果不指定 `--model` 参数，将使用Config中的默认模型（从环境变量或 `.env` 文件读取）

