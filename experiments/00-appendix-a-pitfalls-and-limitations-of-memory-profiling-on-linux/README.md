# Experiment 00: Pitfalls and Limitations of Memory Profiling on Linux

This experiment investigates the challenges and limitations of accurately measuring memory consumption in Python
applications running on Linux systems. It serves as Appendix A of the Memory-Aware Chunking thesis and provides a
comprehensive analysis of different memory profiling techniques and their reliability under various conditions.

## 🎯 Objective

The primary goal is to systematically evaluate different memory profiling approaches for Python applications,
particularly focusing on:

- **Accuracy comparison** between internal Python profilers and external system-level tools
- **Reliability assessment** under memory pressure and resource constraints
- **Identification of pitfalls** in memory measurement that can lead to incorrect conclusions
- **Best practices** for memory profiling in scientific computing workflows

## 🔬 Methodology

The experiment uses a controlled approach with synthetic seismic data processing as the computational workload. It
employs multiple memory profiling techniques simultaneously to cross-validate measurements and identify discrepancies.

### Core Components

1. **Synthetic Data Generation**: Creates seismic datasets of varying sizes to simulate real-world computational
   workloads
2. **Memory Profiling**: Implements multiple profiling backends (psutil, resource, tracemalloc, kernel-level monitoring)
3. **Isolated Execution**: Uses Docker containers with controlled resource limits to ensure reproducible measurements
4. **Multi-level Monitoring**: Combines Python-level and system-level memory tracking

### Profiling Techniques Evaluated

| Technique         | Type     | Scope          | Advantages            | Limitations             |
|-------------------|----------|----------------|-----------------------|-------------------------|
| `tracemalloc`     | Internal | Python objects | Fine-grained tracking | Python-only allocations |
| `psutil`          | External | Process-level  | Cross-platform        | OS-dependent accuracy   |
| `resource`        | External | Process-level  | POSIX standard        | Limited granularity     |
| Kernel monitoring | External | System-level   | Most accurate         | Linux-specific          |

## 🏗️ Architecture

The experiment is designed with a modular architecture that separates concerns:

```
experiment/
├── main.py           # CLI interface and command routing
├── actions.py        # High-level experiment actions
├── data.py          # Synthetic data generation
├── profilers.py     # Memory profiling orchestration
├── operators/       # Computational workloads
│   └── envelope.py  # Seismic envelope calculation
└── interfaces.py    # Type definitions
```

### Execution Environment

The experiment runs in a containerized environment with:

- **Supervisor daemon** for process orchestration
- **Multiple monitoring scripts** running concurrently
- **Controlled resource limits** via cgroups
- **Isolated filesystem** to prevent interference

## 🚀 Usage

### Prerequisites

- Docker with BuildKit support
- Python 3.13+ (for local development)
- Linux system (for kernel-level monitoring)

### Quick Start

1. **Generate synthetic data**:

```bash
python experiment/main.py generate-data --inlines 100 --xlines 100 --samples 1000 --output-dir ./data
```

2. **Run memory profiling experiment**:

```bash
python experiment/main.py operate envelope \
    --segy-path ./data/100-100-1000.segy \
    --memory-profiler tracemalloc \
    --memory-profile-output-dir ./results \
    --memory-profile-session-id experiment-001
```

3. **Run in containerized environment** (recommended):

```bash
docker build -t memory-profiling-experiment .
docker run --privileged \
    -v $(pwd)/out:/app/out \
    memory-profiling-experiment \
    operate envelope --segy-path /app/data/sample.segy --memory-profiler kernel
```

### Available Commands

#### Data Generation

```bash
python experiment/main.py generate-data [OPTIONS]
```

**Options:**

- `--inlines INT`: Number of inline traces (default: 100)
- `--xlines INT`: Number of crossline traces (default: 100)
- `--samples INT`: Number of time samples (default: 100)
- `--output-dir PATH`: Output directory for SEGY files
- `--prefix STR`: Filename prefix

#### Memory Profiling

```bash
python experiment/main.py operate OPERATOR [OPTIONS]
```

**Operators:**

- `envelope`: Seismic envelope calculation using Hilbert transform

**Options:**

- `--segy-path PATH`: Path to input SEGY file (required)
- `--memory-profiler {none,psutil,resource,tracemalloc,kernel}`: Profiling method
- `--memory-profile-output-dir PATH`: Output directory for profiling results
- `--memory-profile-session-id STR`: Session identifier for output files

## 📊 Output and Analysis

### Generated Data

The experiment produces several types of output:

1. **Memory usage logs**: Time-series data of memory consumption
2. **Profiling reports**: Detailed memory allocation traces
3. **System metrics**: CPU usage, page faults, memory pressure indicators
4. **Performance metrics**: Execution time and peak memory usage

### Log Files

When running in containerized mode, the following logs are generated:

- `/app/logs/memory-usage.log`: System-level memory consumption over time
- `/app/logs/page-faults.log`: Page fault statistics
- `/app/logs/memory-pressure.log`: Memory pressure indicators
- `/app/logs/main.log`: Application execution logs
- `/app/logs/supervisord.log`: Process orchestration logs

### Analysis Notebooks

The `notebooks/` directory contains Jupyter notebooks for analyzing results:

- **01-data-generation.ipynb**: Validates synthetic data generation
- **02-execution-environment.ipynb**: Tests execution environment setup
- **03-memory-profiling.ipynb**: Analyzes memory profiling accuracy
- **04-memory-pressure.ipynb**: Studies behavior under memory constraints

## 🐳 Docker Environment

The experiment includes a sophisticated Docker setup for isolated execution:

### Multi-stage Build

- **Base stage**: Uses TraceQ profiling framework
- **Builder stage**: Installs Python dependencies
- **Final stage**: Configures monitoring and execution environment

### Resource Control

- **Memory limits**: Configurable via `MEMORY_LIMIT_MB` environment variable
- **CPU allocation**: Single CPU core allocation for consistent results
- **cgroup monitoring**: Direct access to kernel memory statistics

### Monitoring Stack

- **Supervisor**: Orchestrates multiple monitoring processes
- **Memory usage monitor**: Tracks cgroup memory consumption
- **Page fault monitor**: Records memory access patterns
- **Memory pressure monitor**: Detects memory stress conditions

## 🔧 Configuration

### Environment Variables

Key configuration options:

```bash
# Memory monitoring
MEMORY_USAGE_SAMPLING_PRECISION=2          # Sampling frequency (10^-2 seconds)
MEMORY_LIMIT_MB=1024                       # Container memory limit

# Page fault monitoring
PAGE_FAULTS_MONITORED_PROCESS_NAME="operate envelope"
PAGE_FAULTS_SAMPLING_PRECISION=2

# Memory pressure monitoring
MEMORY_PRESSURE_SAMPLING_PRECISION=2
```

### Supervisor Configuration

The `supervisord.conf` file orchestrates multiple monitoring processes:

- Memory usage monitoring (priority 1)
- Page fault monitoring (priority 2)
- Memory pressure monitoring (priority 3)
- Memory limit enforcement (priority 4)
- Main experiment execution (priority 5)

## 📈 Key Findings

This experiment reveals several important insights about memory profiling:

1. **Tool-specific discrepancies**: Different profiling tools can report significantly different memory usage values
2. **Memory pressure effects**: System memory pressure affects profiling accuracy
3. **Allocation vs. usage**: Distinction between allocated and actually used memory varies by tool
4. **Timing sensitivity**: Memory measurements are highly sensitive to sampling frequency

## 🔗 Dependencies

Core dependencies (see `requirements.txt`):

- **traceq**: Advanced memory profiling framework
- **numpy**: Numerical computing
- **scipy**: Scientific computing (Hilbert transform)
- **segyio**: Seismic data I/O
- **psutil**: System and process monitoring
- **loguru**: Structured logging

## 📚 Related Work

This experiment supports the theoretical framework presented in the Memory-Aware Chunking thesis, specifically:

- Chapter 3: Memory Management in Scientific Computing
- Appendix A: Memory Profiling Methodology
- Appendix B: Experimental Validation Framework

## 🤝 Contributing

When modifying this experiment:

1. **Maintain reproducibility**: Ensure all changes preserve deterministic behavior
2. **Update documentation**: Modify this README and relevant notebooks
3. **Test thoroughly**: Validate changes across different system configurations
4. **Follow conventions**: Use existing code style and naming patterns

## 📄 License

This experiment is part of the Memory-Aware Chunking thesis research project. Please refer to the main repository
license for usage terms.