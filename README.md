# Memory-Aware Chunking for Seismic Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)

This repository contains the complete research framework for **Memory-Aware Chunking**, a novel approach to optimizing memory utilization in distributed seismic data processing workflows. The work demonstrates how machine learning-based memory prediction can significantly improve data parallelism performance while reducing out-of-memory failures.

## 🎯 Overview

Memory-Aware Chunking addresses the critical challenge of optimal data partitioning in memory-constrained distributed computing environments. By leveraging machine learning models to predict memory consumption patterns, this approach enables intelligent chunk sizing that maximizes computational efficiency while preventing memory exhaustion.

### Key Contributions

- **🔬 Comprehensive Memory Profiling Framework**: Systematic evaluation of memory measurement techniques for Python applications
- **🤖 Predictive Memory Models**: Machine learning models that accurately predict memory consumption from input data dimensions
- **⚡ Intelligent Chunking Algorithm**: Memory-aware data partitioning strategy for distributed computing frameworks
- **📊 Performance Validation**: Empirical demonstration of 15-40% execution time improvements and 60-80% reduction in out-of-memory failures
- **🛠️ Production-Ready Tools**: TraceQ profiling framework and common utilities for seismic data processing

## 🏗️ Repository Structure

```
memory-aware-chunking/
├── experiments/                    # Comprehensive experimental framework
│   ├── 00-appendix-a-pitfalls-and-limitations-of-memory-profiling-on-linux/
│   ├── 01-measuring-memory-usage-of-python-programs/
│   ├── 02-predicting-memory-consumption-from-input-shapes/
│   ├── 03-improving-data-parallelism-using-memory-aware-chunking/
│   └── README.md                  # Experiments overview and navigation
├── libs/                          # Reusable libraries and frameworks
│   ├── common/                    # Shared utilities for seismic processing
│   └── traceq/                    # Advanced memory profiling framework
├── thesis/                        # Complete thesis documentation
│   ├── sections/                  # Individual thesis chapters
│   ├── assets/                    # Images, templates, and macros
│   ├── bibliography.bib           # Research references
│   ├── main.tex                   # Main thesis document
│   └── out/main.pdf              # Compiled thesis PDF
├── LICENSE                        # MIT License
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- **Docker** with BuildKit support
- **Linux system** (recommended for optimal performance)
- **8+ GB RAM** (varies by experiment)
- **Git** for repository management

### Running Experiments

Each experiment is self-contained and can be executed independently:

```bash
# Clone the repository
git clone https://github.com/discovery-unicamp/memory-aware-chunking.git
cd memory-aware-chunking

# Navigate to any experiment
cd experiments/01-measuring-memory-usage-of-python-programs

# Run the complete experiment pipeline
./scripts/experiment.sh
```

### Installing Libraries

The repository includes two reusable Python libraries:

```bash
# Install TraceQ profiling framework
cd libs/traceq
pip install .

# Install common seismic processing utilities
cd libs/common
pip install .
```

## 📊 Experimental Framework

The research follows a systematic four-phase approach:

### [🔍 Experiment 00: Memory Profiling Foundations](./experiments/00-appendix-a-pitfalls-and-limitations-of-memory-profiling-on-linux)
**Objective**: Investigate challenges and limitations of memory measurement in Python applications on Linux systems.

**Key Technologies**: Multiple profiling backends, Docker containers, supervisor-orchestrated monitoring

**Findings**: Tool-specific measurement discrepancies, memory pressure effects, timing sensitivity in profiling

### [📏 Experiment 01: Memory Profiling Tool Validation](./experiments/01-measuring-memory-usage-of-python-programs)
**Objective**: Comprehensive comparison of memory profiling techniques with statistical validation.

**Key Technologies**: 8 profiling approaches, Docker-in-Docker execution, TraceQ framework validation

**Findings**: Kernel-level monitoring accuracy, TraceQ framework effectiveness, statistical significance requirements

### [🤖 Experiment 02: Predictive Memory Modeling](./experiments/02-predicting-memory-consumption-from-input-shapes)
**Objective**: Develop machine learning models to predict memory consumption from input data dimensions.

**Key Technologies**: 8 regression algorithms, advanced feature engineering, Optuna hyperparameter optimization

**Findings**: Ensemble method superiority, feature importance insights, algorithm-specific modeling requirements

### [⚡ Experiment 03: Memory-Aware Chunking Validation](./experiments/03-improving-data-parallelism-using-memory-aware-chunking)
**Objective**: Demonstrate practical performance improvements through memory-aware chunking in distributed computing.

**Key Technologies**: Dask distributed computing, real-time memory monitoring, Docker-in-Docker architecture

**Findings**: 15-40% execution time reduction, 60-80% fewer OOM failures, improved scaling efficiency

## 🛠️ Libraries and Tools

### TraceQ: Advanced Memory Profiling Framework

[TraceQ](./libs/traceq) is a specialized profiling tool designed for accurate memory measurements in Python applications:

**Features**:
- High-accuracy profiling using Linux `/proc` filesystem
- Multiple backend support (psutil, tracemalloc, kernel-level)
- Granular memory usage analysis
- Optimized for data processing tasks
- HPC environment compatibility

**Usage**:
```python
from traceq import profile

@profile
def memory_intensive_task(data):
    # Your computation here
    pass
```

### Common: Seismic Processing Utilities

[Common](./libs/common) provides shared utilities for seismic data processing across all experiments:

**Components**:
- **Builders**: Synthetic seismic data generation
- **Loaders**: SEGY file I/O operations
- **Operators**: Seismic processing algorithms (Envelope, GST3D, Gaussian Filter)
- **Transformers**: Data transformation utilities
- **Runners**: Execution orchestration tools

## 📚 Thesis Documentation

The [thesis](./thesis) directory contains the complete academic documentation:

**Structure**:
- **Chapters 1-3**: Introduction, fundamental concepts, and related work
- **Chapter 4**: Memory profiling methodology and TraceQ framework
- **Chapter 5**: Predictive memory modeling with machine learning
- **Chapter 6**: Memory-aware chunking algorithm and implementation
- **Chapter 7**: Experimental validation and performance analysis
- **Appendix A**: Detailed memory profiling challenges and solutions

**Compilation**:
```bash
cd thesis
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 📈 Key Results

### Performance Improvements
- **15-40% reduction** in execution time for memory-constrained scenarios
- **60-80% fewer** out-of-memory failures in distributed processing
- **Improved scaling efficiency** with increasing worker count
- **Better resource utilization** without performance degradation

### Memory Optimization
- **Accurate prediction** of memory consumption from input dimensions
- **Intelligent chunk sizing** based on ML-predicted memory patterns
- **Dynamic adaptation** to available memory per worker
- **Predictable resource usage** for better allocation planning

### Tool Validation
- **Comprehensive profiling accuracy** assessment across multiple tools
- **TraceQ framework validation** for production-ready memory monitoring
- **Statistical significance** requirements for reliable memory profiling
- **Best practices** for memory measurement in scientific computing

## 🔬 Research Methodology

### Systematic Experimental Design

The research employs a rigorous experimental methodology:

1. **Controlled Environments**: Docker-based isolation ensures reproducible results
2. **Statistical Validation**: Multiple runs with statistical analysis for reliability
3. **Cross-Validation**: Independent validation across different tools and approaches
4. **Scalability Testing**: Evaluation across multiple worker configurations and data sizes
5. **Real-World Validation**: Practical application using industry-standard seismic processing algorithms

### Seismic Processing Algorithms

The framework validates memory-aware chunking using three representative seismic processing operations:

| Algorithm | Type | Computational Pattern | Memory Characteristics |
|-----------|------|----------------------|----------------------|
| **Envelope** | Signal Processing | Hilbert transform on traces | Linear scaling with volume |
| **GST3D** | Structural Analysis | 3D gradient structure tensor | Cubic scaling with dimensions |
| **Gaussian Filter** | Smoothing | 3D convolution filtering | Quadratic scaling with kernel |

## 🌟 Applications and Use Cases

### High-Performance Computing (HPC)
- **Resource optimization** in memory-constrained cluster environments
- **Job scheduling** with accurate memory requirement prediction
- **Cost reduction** through improved resource utilization

### Cloud Computing
- **Instance sizing** optimization for memory-intensive workloads
- **Auto-scaling** based on predicted memory requirements
- **Cost optimization** through right-sizing of cloud resources

### Seismic Data Processing
- **Large-scale surveys** with intelligent data partitioning
- **Real-time processing** with memory-aware chunk sizing
- **Distributed workflows** with optimized memory utilization

### Scientific Computing
- **Memory-intensive simulations** with predictive resource planning
- **Data processing pipelines** with intelligent chunking strategies
- **Distributed computing frameworks** with memory-aware optimization

## 🔧 Advanced Configuration

### Environment Variables

Global configuration options across all experiments:

```bash
# Resource allocation
export CPUSET_CPUS="0,1,2,3"        # CPU core assignment
export MEMORY_LIMIT_GB=32            # Memory limit per experiment

# Dataset configuration
export DATASET_FINAL_SIZE=800        # Maximum data dimensions
export DATASET_STEP_SIZE=200         # Increment between sizes

# Experiment parameters
export EXPERIMENT_N_RUNS=10          # Statistical sample size
export SAFETY_FACTOR=0.8             # Memory safety margin
```

### Custom Profiling Configuration

TraceQ framework customization:

```toml
# traceq.toml
output_dir = "./custom_reports"

[profiler]
enabled_metrics = "memory_usage,execution_time"
precision = "4"

[profiler.memory_usage]
enabled_backends = "kernel,psutil,tracemalloc"
sampling_interval = "0.01"
```

### Docker Resource Management

Container resource control:

```bash
# Memory-constrained testing
docker run --memory=4g --cpuset-cpus="0,1" experiment-image

# High-performance execution
docker run --memory=32g --cpuset-cpus="0-7" experiment-image
```

## 🤝 Contributing

We welcome contributions to the Memory-Aware Chunking framework! Here's how you can help:

### Development Guidelines

1. **Follow existing patterns**: Use established code structure and naming conventions
2. **Maintain reproducibility**: Ensure all changes preserve deterministic behavior
3. **Document thoroughly**: Update README files and inline documentation
4. **Test comprehensively**: Validate changes across different configurations
5. **Preserve compatibility**: Maintain backward compatibility with existing experiments

### Adding New Experiments

To contribute a new experiment:

1. **Create experiment directory**: Follow naming pattern `04-new-experiment-name`
2. **Include standard structure**: `experiment/`, `scripts/`, `notebooks/`, `requirements.txt`, `Dockerfile`, `README.md`
3. **Document dependencies**: Clearly specify relationships to existing experiments
4. **Update overview**: Add experiment description to main experiments README

### Extending Libraries

To enhance TraceQ or Common libraries:

1. **Follow library conventions**: Maintain existing API patterns
2. **Add comprehensive tests**: Include unit and integration tests
3. **Update documentation**: Modify README and inline documentation
4. **Version appropriately**: Follow semantic versioning principles

### Reporting Issues

When reporting issues:

1. **Provide context**: Include system information and experiment details
2. **Include logs**: Attach relevant error messages and execution logs
3. **Describe reproduction**: Provide steps to reproduce the issue
4. **Suggest solutions**: If possible, propose potential fixes

## 📖 Publications and Citations

This work has been developed as part of academic research. If you use this framework in your research, please cite:

```bibtex
@mastersthesis{fonseca2024memory,
  title={Memory-Aware Chunking for Seismic Processing},
  author={Fonseca, Daniel De Lucca},
  year={2024},
  school={University of Campinas},
  type={Master's Thesis}
}
```

### Related Publications

- **TraceQ Framework**: Advanced memory profiling for Python applications
- **Memory Prediction Models**: Machine learning approaches to memory consumption estimation
- **Distributed Computing Optimization**: Memory-aware strategies for data parallelism

## 🙏 Acknowledgments

This research was conducted at the **Institute of Computing, University of Campinas (Unicamp)**, Brazil, under the supervision of **Prof. Edson Borin**.

**Special Thanks**:
- **Petrobras** for collaboration and support in seismic processing research
- **Discovery Research Group** for providing the research infrastructure
- **Open Source Community** for the foundational tools and libraries used in this work

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

The MIT License allows for:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

## 🔗 Links and Resources

- **Repository**: [https://github.com/discovery-unicamp/memory-aware-chunking](https://github.com/discovery-unicamp/memory-aware-chunking)
- **TraceQ Library**: [./libs/traceq](./libs/traceq)
- **Experiments Overview**: [./experiments](./experiments)
- **Thesis Documentation**: [./thesis](./thesis)
- **Institute of Computing, Unicamp**: [https://www.ic.unicamp.br/](https://www.ic.unicamp.br/)
- **Discovery Research Group**: [https://www.discovery.ic.unicamp.br/](https://www.discovery.ic.unicamp.br/)

---

**Memory-Aware Chunking** represents a significant advancement in intelligent memory management for distributed computing, providing both theoretical insights and practical tools for optimizing memory utilization in data-intensive applications.