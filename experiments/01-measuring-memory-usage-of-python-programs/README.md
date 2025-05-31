# Experiment 01: Measuring Memory Usage of Python Programs

This experiment provides a comprehensive comparison of different memory profiling techniques for Python applications,
focusing on the accuracy and reliability of various measurement tools. It serves as the primary empirical study for
evaluating memory profiling approaches in the Memory-Aware Chunking thesis.

## 🎯 Objective

The primary goal is to systematically compare memory profiling tools and techniques, specifically:

- **Cross-validation** of memory measurements between different profiling tools
- **Accuracy assessment** of internal vs. external memory profiling approaches
- **Performance impact analysis** of profiling overhead on application execution
- **TraceQ framework evaluation** as a unified profiling solution
- **Best practices identification** for memory profiling in scientific computing

## 🔬 Methodology

This experiment uses a controlled comparative approach where the same computational workload (seismic envelope
calculation) is executed multiple times using different memory profiling techniques. The results are then analyzed to
identify discrepancies, patterns, and reliability characteristics.

### Core Components

1. **Synthetic Data Generation**: Creates consistent seismic datasets for reproducible testing
2. **Multi-tool Profiling**: Implements 8 different profiling approaches (4 direct + 4 via TraceQ)
3. **Statistical Analysis**: Performs comprehensive statistical comparison of profiling results
4. **Visualization Suite**: Generates detailed charts and reports for result interpretation

### Profiling Tools Evaluated

| Tool            | Type     | Implementation  | Advantages                    | Limitations             |
|-----------------|----------|-----------------|-------------------------------|-------------------------|
| **psutil**      | External | Direct + TraceQ | Cross-platform, process-level | OS-dependent accuracy   |
| **resource**    | External | Direct + TraceQ | POSIX standard                | Limited granularity     |
| **tracemalloc** | Internal | Direct + TraceQ | Fine-grained Python tracking  | Python-only allocations |
| **kernel**      | External | Direct + TraceQ | Most accurate system view     | Linux-specific          |

### Experimental Design

- **Multiple runs**: Each profiling tool executes 5 independent runs for statistical validity
- **Controlled environment**: Docker containers with fixed CPU allocation and isolated execution
- **Consistent workload**: Same synthetic seismic data processed by all tools
- **Comprehensive metrics**: Memory usage over time, peak consumption, and execution statistics

## 🏗️ Architecture

The experiment follows a modular pipeline architecture:

```
experiment/
├── generate_data.py           # Synthetic seismic data generation
├── measure_with_psutil.py     # Direct psutil profiling
├── measure_with_resource.py   # Direct resource module profiling
├── measure_with_tracemalloc.py # Direct tracemalloc profiling
├── measure_with_kernel.py     # Direct kernel-level profiling
├── measure_with_traceq.py     # Unified TraceQ profiling
├── collect_results.py         # Data aggregation and preprocessing
└── analyze_results.py         # Statistical analysis and visualization
```

### Execution Pipeline

1. **Data Generation**: Creates synthetic seismic datasets with configurable dimensions
2. **Profiling Phase**: Executes each tool multiple times with identical inputs
3. **Collection Phase**: Aggregates results from all profiling runs into unified datasets
4. **Analysis Phase**: Performs statistical analysis and generates comprehensive visualizations

## 🚀 Usage

### Prerequisites

- Docker with BuildKit support
- Linux system (recommended for kernel-level profiling)
- Sufficient disk space for results (varies with dataset size)

### Quick Start

Run the complete experiment pipeline:

```bash
cd experiments/01-measuring-memory-usage-of-python-programs
./scripts/experiment.sh
```

### Configuration

Key environment variables for customization:

```bash
# Dataset configuration
export DATASET_INLINES=600      # Number of inline traces
export DATASET_XLINES=600       # Number of crossline traces
export DATASET_SAMPLES=600      # Number of time samples

# Experiment configuration
export EXPERIMENT_N_RUNS=5      # Number of runs per tool
export CPUSET_CPUS=0            # CPU core allocation

# Output configuration
export OUTPUT_DIR="./out/results/$(date +%Y%m%d%H%M%S)"
```

### Individual Components

#### 1. Data Generation

```bash
python experiment/generate_data.py
```

Environment variables:

- `OUTPUT_DIR`: Output directory for generated data (default: `./out/inputs`)
- `DATASET_INLINES`: Number of inline traces (default: 100)
- `DATASET_XLINES`: Number of crossline traces (default: 100)
- `DATASET_SAMPLES`: Number of time samples (default: 100)

#### 2. Memory Profiling

**Direct tool profiling:**

```bash
# psutil profiling
python experiment/measure_with_psutil.py

# resource module profiling
python experiment/measure_with_resource.py

# tracemalloc profiling
python experiment/measure_with_tracemalloc.py

# kernel-level profiling
python experiment/measure_with_kernel.py
```

**TraceQ unified profiling:**

```bash
# TraceQ with different backends
python experiment/measure_with_traceq.py
```

Environment variables for profiling:

- `SEGY_FILEPATH`: Path to input seismic data file
- `OUTPUT_RESULT_PATH`: Output file for profiling results
- `POLL_INTERVAL`: Sampling interval in seconds (default: 0.05)
- `APPEND_TIMESTAMP`: Whether to append timestamp to output files
- `TRACEQ_BACKEND`: Backend for TraceQ (psutil, resource, tracemalloc, kernel)
- `SESSION_ID`: Session identifier for TraceQ profiling

#### 3. Results Collection

```bash
python experiment/collect_results.py
```

Aggregates individual profiling results into unified CSV files:

- `profiles_detail.csv`: Time-series memory usage data
- `profiles_summary.csv`: Peak memory usage statistics

#### 4. Analysis and Visualization

```bash
python experiment/analyze_results.py
```

Generates comprehensive analysis including:

- Memory usage timeline charts
- Statistical comparison tables
- Distribution analysis (KDE, CDF)
- Tool comparison visualizations

## 📊 Output and Analysis

### Generated Artifacts

The experiment produces a comprehensive set of outputs organized in the following structure:

```
out/
├── inputs/                    # Generated synthetic data
│   └── {inlines}-{xlines}-{samples}.segy
├── profiles/                  # Raw profiling results
│   ├── psutil-{timestamp}.txt
│   ├── resource-{timestamp}.txt
│   ├── tracemalloc-{timestamp}.txt
│   ├── kernel-{timestamp}.txt
│   ├── traceq_psutil-{run_id}.prof
│   ├── traceq_resource-{run_id}.prof
│   ├── traceq_tracemalloc-{run_id}.prof
│   └── traceq_kernel-{run_id}.prof
├── results/                   # Aggregated data
│   ├── profiles_detail.csv
│   └── profiles_summary.csv
└── analysis/                  # Visualizations and reports
    ├── memory_history_no_traceq.pdf
    ├── compare_orig_vs_traceq_{tool}.pdf
    ├── kde_base_vs_traceq_{tool}.pdf
    ├── cdf_base_vs_traceq_{tool}.pdf
    ├── max_memory_orig_vs_traceq.pdf
    ├── boxplot_base_vs_traceq.pdf
    ├── points_collected.pdf
    ├── phase_comparison.csv
    └── traceq_comparison.csv
```

### Key Analysis Reports

#### 1. Memory Usage Timeline Analysis

- **Individual tool charts**: Memory consumption over time for each profiling tool
- **Comparative charts**: Side-by-side comparison of original tools vs. TraceQ implementations
- **Aggregate timeline**: Combined view of all non-TraceQ tools

#### 2. Statistical Distribution Analysis

- **Kernel Density Estimation (KDE)**: Probability density of memory usage values
- **Cumulative Distribution Function (CDF)**: Cumulative probability distributions
- **Box plots**: Distribution comparison with quartiles and outliers

#### 3. Comparative Analysis Tables

- **TraceQ comparison**: Detailed comparison of original tools vs. TraceQ implementations
- **Phase analysis**: Memory usage statistics across different execution phases
- **Peak memory analysis**: Maximum memory consumption comparison

#### 4. Performance Metrics

- **Data collection efficiency**: Number of measurement points collected per tool
- **Profiling overhead**: Impact of profiling on execution performance
- **Measurement consistency**: Variance analysis across multiple runs

## 🐳 Docker Environment

The experiment uses a sophisticated Docker-based execution environment to ensure reproducibility and isolation:

### Multi-stage Build Process

- **Builder stage**: Compiles TraceQ (Rust components) and installs Python dependencies
- **Final stage**: Creates minimal runtime environment with user permissions

### Execution Architecture

- **Docker-in-Docker (DinD)**: Provides isolated container execution for each profiling run
- **Volume management**: Persistent storage for results across container lifecycles
- **Resource constraints**: Fixed CPU allocation and memory limits for consistent measurements

### Container Features

- **User permission mapping**: Maintains host user permissions for output files
- **Build context sharing**: Efficient sharing of common libraries (TraceQ, common utilities)
- **Environment isolation**: Each profiling run executes in a fresh container instance

## 📈 Key Findings

This experiment reveals important insights about memory profiling tool accuracy and reliability:

### Tool Accuracy Comparison

1. **Kernel-level monitoring** provides the most accurate system-wide memory measurements
2. **psutil** shows good correlation with kernel measurements but with some overhead
3. **tracemalloc** captures Python-specific allocations but misses C-library memory usage
4. **resource module** provides limited granularity but consistent peak measurements

### TraceQ Framework Validation

1. **Measurement consistency**: TraceQ implementations show high correlation with direct tool usage
2. **Overhead analysis**: TraceQ introduces minimal additional profiling overhead
3. **Unified interface**: Provides consistent API across different profiling backends
4. **Data quality**: Maintains measurement accuracy while improving data collection efficiency

### Profiling Best Practices

1. **Multiple tool validation**: Cross-validation between tools improves measurement confidence
2. **Statistical significance**: Multiple runs essential for reliable memory profiling
3. **Environment control**: Containerized execution critical for reproducible results
4. **Sampling frequency**: Higher sampling rates improve temporal resolution but increase overhead

## 🔗 Dependencies

Core dependencies (see `requirements.txt`):

- **matplotlib**: Visualization and chart generation
- **pandas**: Data manipulation and analysis
- **psutil**: System and process monitoring
- **seaborn**: Statistical data visualization
- **traceq**: Unified memory profiling framework (built from source)
- **common**: Shared utilities for seismic data processing (built from source)

## 📚 Related Work

This experiment supports the theoretical framework presented in the Memory-Aware Chunking thesis:

- **Chapter 2**: Memory Profiling Techniques and Tools
- **Chapter 3**: Empirical Evaluation of Memory Measurement Approaches
- **Appendix A**: Detailed Memory Profiling Methodology
- **Appendix B**: Statistical Analysis of Profiling Tool Accuracy

## 🔧 Advanced Configuration

### Custom Dataset Generation

Create datasets with specific characteristics:

```bash
# Large dataset for stress testing
export DATASET_INLINES=1000
export DATASET_XLINES=1000
export DATASET_SAMPLES=1000

# Small dataset for quick testing
export DATASET_INLINES=100
export DATASET_XLINES=100
export DATASET_SAMPLES=100
```

### Profiling Customization

Adjust profiling parameters:

```bash
# High-frequency sampling
export POLL_INTERVAL=0.01

# Extended profiling runs
export EXPERIMENT_N_RUNS=10

# Custom output location
export OUTPUT_DIR="/path/to/custom/output"
```

### Docker Resource Control

Configure container resources:

```bash
# CPU allocation
export CPUSET_CPUS="0,1"  # Use specific CPU cores

# Memory limits (handled by Docker)
export MEMORY_LIMIT="4g"
```

## 🤝 Contributing

When modifying this experiment:

1. **Maintain statistical validity**: Ensure changes preserve the ability to perform meaningful statistical analysis
2. **Update documentation**: Modify this README and any relevant analysis scripts
3. **Test thoroughly**: Validate changes across different system configurations and dataset sizes
4. **Follow conventions**: Use existing code style and naming patterns
5. **Preserve reproducibility**: Ensure all changes maintain deterministic behavior

### Adding New Profiling Tools

To add a new profiling tool:

1. Create `measure_with_{tool}.py` following existing patterns
2. Update `collect_results.py` to handle the new tool's output format
3. Modify `analyze_results.py` to include the new tool in comparisons
4. Update the experiment shell script to include the new tool in the pipeline

## 📄 License

This experiment is part of the Memory-Aware Chunking thesis research project. Please refer to the main repository
license for usage terms.