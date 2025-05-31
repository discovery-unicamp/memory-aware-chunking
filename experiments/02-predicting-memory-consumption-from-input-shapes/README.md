# Experiment 02: Predicting Memory Consumption from Input Shapes

This experiment develops and evaluates machine learning models to predict memory consumption of seismic processing
algorithms based on input data dimensions. It serves as the core predictive modeling component of the Memory-Aware
Chunking thesis, enabling intelligent memory-aware workload optimization.

## 🎯 Objective

The primary goal is to build accurate predictive models that can estimate memory consumption before algorithm execution,
specifically:

- **Predictive Model Development**: Train multiple regression models to predict peak memory usage from input shapes
- **Algorithm-Specific Modeling**: Create specialized models for different seismic processing operators
- **Feature Engineering**: Develop effective features from input dimensions (inlines, xlines, samples)
- **Model Optimization**: Perform hyperparameter tuning and feature selection for optimal performance
- **Practical Validation**: Evaluate model accuracy and reliability for real-world memory-aware chunking applications

## 🔬 Methodology

This experiment employs a comprehensive machine learning pipeline that combines systematic data generation, memory
profiling, feature engineering, and multi-model evaluation to develop robust memory consumption predictors.

### Core Components

1. **Systematic Data Generation**: Creates datasets with varying input dimensions using combinatorial approach
2. **Multi-Algorithm Profiling**: Profiles memory usage across different seismic processing operators
3. **Advanced Feature Engineering**: Generates geometric, logarithmic, and ratio-based features from input shapes
4. **Multi-Model Evaluation**: Trains and compares 8 different regression algorithms with hyperparameter optimization
5. **Model Selection & Optimization**: Uses Optuna for automated hyperparameter tuning and model selection

### Seismic Processing Algorithms

| Algorithm           | Type                | Computational Characteristics | Memory Pattern                |
|---------------------|---------------------|-------------------------------|-------------------------------|
| **Envelope**        | Signal Processing   | Hilbert transform on traces   | Linear with volume            |
| **GST3D**           | Structural Analysis | 3D gradient structure tensor  | Cubic with spatial dimensions |
| **Gaussian Filter** | Smoothing           | 3D convolution filtering      | Quadratic with kernel size    |

### Machine Learning Pipeline

1. **Data Collection**: Memory profiling across input dimension combinations (100×100×100 to 800×800×800)
2. **Feature Engineering**: 20+ derived features including volume, ratios, logarithmic transforms, and geometric
   properties
3. **Model Training**: 8 regression algorithms with automated hyperparameter optimization
4. **Model Selection**: Multi-objective optimization balancing accuracy, RMSE, MAE, and R²
5. **Validation**: Data reduction and feature selection analysis for robustness assessment

## 🏗️ Architecture

The experiment follows a sophisticated ML pipeline architecture:

```
experiment/
├── generate_data.py           # Systematic dataset generation
├── collect_memory_profile.py  # TraceQ-based memory profiling
├── collect_results.py         # ML pipeline & model training
└── analyze_results.py         # Comprehensive analysis & visualization
```

### Execution Pipeline

1. **Data Generation**: Creates synthetic seismic datasets across input dimension combinations
2. **Memory Profiling**: Profiles each algorithm on each dataset using TraceQ framework
3. **Feature Engineering**: Extracts 20+ features from input dimensions and memory profiles
4. **Model Training**: Trains 8 regression models with hyperparameter optimization
5. **Model Selection**: Uses Optuna to find optimal model weighting and selection criteria
6. **Validation**: Performs data reduction and feature selection analysis
7. **Analysis**: Generates comprehensive visualizations and performance reports

## 🚀 Usage

### Prerequisites

- Docker with BuildKit support
- Sufficient computational resources (experiment can be resource-intensive)
- Linux system (recommended for optimal performance)

### Quick Start

Run the complete experiment pipeline:

```bash
cd experiments/02-predicting-memory-consumption-from-input-shapes
./scripts/experiment.sh
```

### Configuration

Key environment variables for customization:

```bash
# Dataset configuration
export DATASET_FINAL_SIZE=800    # Maximum dimension size
export DATASET_STEP_SIZE=100     # Increment between sizes

# Experiment configuration
export EXPERIMENT_N_RUNS=5       # Number of profiling runs per configuration
export CPUSET_CPUS=0             # CPU core allocation

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
- `INITIAL_SIZE`: Starting dimension size (default: 100)
- `FINAL_SIZE`: Maximum dimension size (default: 600)
- `STEP_SIZE`: Increment between sizes (default: 100)

#### 2. Memory Profiling

```bash
python experiment/collect_memory_profile.py
```

Environment variables:

- `ALGORITHM`: Algorithm to profile (envelope, gst3d, gaussian_filter)
- `INPUT_PATH`: Path to input SEGY file
- `OUTPUT_DIR`: Output directory for profiles (default: `./out/profiles`)
- `SESSION_ID`: Unique session identifier
- `PROFILER`: Profiling backend (default: kernel)

#### 3. Model Training & Analysis

```bash
python experiment/collect_results.py
```

Environment variables:

- `OUTPUT_DIR`: Base output directory (default: `./out`)
- `PROFILES_DIR`: Directory containing profile files
- `TEST_SIZE`: Train/test split ratio (default: 0.2)
- `ACCURACY_THRESHOLD`: Accuracy threshold for model evaluation (default: 0.05)
- `MODELS_TO_EVALUATE`: Comma-separated list of models to train
- `OPTUNA_TRIALS`: Number of optimization trials (default: 50)
- `K_FOLDS`: Cross-validation folds (default: 3)

#### 4. Results Analysis

```bash
python experiment/analyze_results.py
```

Generates comprehensive visualizations and analysis reports for all trained models and operators.

## 📊 Machine Learning Models

The experiment evaluates 8 different regression algorithms, each optimized through hyperparameter tuning:

### Model Portfolio

| Model                         | Type               | Strengths                          | Hyperparameters Tuned                  |
|-------------------------------|--------------------|------------------------------------|----------------------------------------|
| **Linear Regression**         | Linear             | Simple, interpretable              | None (baseline)                        |
| **Polynomial Regression**     | Non-linear         | Captures polynomial relationships  | Degree (2-4)                           |
| **Decision Tree**             | Tree-based         | Non-linear, interpretable          | Max depth, min samples split           |
| **Random Forest**             | Ensemble           | Robust, handles overfitting        | N estimators, max depth                |
| **Gradient Boosting**         | Ensemble           | High accuracy, sequential learning | N estimators, learning rate, max depth |
| **XGBoost**                   | Ensemble           | State-of-the-art gradient boosting | N estimators, learning rate, max depth |
| **Support Vector Regression** | Kernel-based       | Effective in high dimensions       | C, epsilon, kernel                     |
| **Elastic Net**               | Regularized Linear | Feature selection, regularization  | Alpha, L1 ratio                        |

### Feature Engineering

The experiment generates 20+ features from basic input dimensions:

#### Basic Features

- **Volume**: `inlines × xlines × samples`
- **Surface Area**: `2 × (inlines×xlines + inlines×samples + xlines×samples)`
- **Diagonal Length**: `√(inlines² + xlines² + samples²)`

#### Interaction Features

- **Pairwise Products**: `inlines×xlines`, `inlines×samples`, `xlines×samples`
- **Logarithmic Transforms**: `log₂(inlines)`, `log₂(xlines)`, `log₂(samples)`, `log₂(volume)`

#### Ratio Features

- **Dimension Ratios**: `inlines/xlines`, `inlines/samples`, `xlines/samples`
- **Proportion Features**: Each dimension as fraction of total sum

#### Advanced Features

- **Quadratic Interactions**: `volume²`
- **Log Combinations**: `log(volume) × log(diagonal)`
- **Statistical Features**: Mean and standard deviation of dimensions

### Model Selection Strategy

The experiment uses a sophisticated multi-objective optimization approach:

1. **Hyperparameter Optimization**: Optuna-based search for each model
2. **Cross-Validation**: K-fold validation during hyperparameter tuning
3. **Multi-Metric Evaluation**: Combines accuracy, RMSE, MAE, and R² scores
4. **Weighted Scoring**: Optuna optimizes metric weights for final model selection
5. **Robustness Testing**: Data reduction and feature selection validation

## 📈 Output and Analysis

### Generated Artifacts

The experiment produces extensive outputs organized by algorithm and analysis type:

```
out/
├── inputs/                           # Generated synthetic datasets
│   └── {inlines}-{xlines}-{samples}.segy
├── profiles/                         # Memory profiling results
│   ├── envelope-{dimensions}-{session}.prof
│   ├── gst3d-{dimensions}-{session}.prof
│   └── gaussian-filter-{dimensions}-{session}.prof
├── results/
│   ├── data/
│   │   ├── dataset.csv              # Aggregated training data
│   │   └── features.csv             # Engineered features
│   ├── operators/
│   │   ├── envelope/
│   │   │   ├── models/              # Individual model results
│   │   │   │   ├── linear_regression/
│   │   │   │   ├── random_forest/
│   │   │   │   └── ...
│   │   │   └── results/
│   │   │       ├── model_metrics.csv
│   │   │       ├── model_tuning.csv
│   │   │       ├── data_reduction.csv
│   │   │       ├── feature_selection.csv
│   │   │       ├── profile_summary.csv
│   │   │       └── profile_history.csv
│   │   ├── gst3d/
│   │   └── gaussian_filter/
│   ├── model_tuning.csv             # Global hyperparameter results
│   └── best_models/                 # Final optimized models
└── charts/                          # Comprehensive visualizations
    ├── profile/                     # Memory profiling analysis
    ├── model/                       # Model performance analysis
    ├── data_reduction/              # Data efficiency analysis
    ├── feature_selection/           # Feature importance analysis
    └── cross_operator/              # Cross-algorithm comparisons
```

### Key Analysis Reports

#### 1. Memory Profiling Analysis

- **Peak Memory vs. Volume**: Memory consumption scaling patterns
- **Memory Distribution**: Statistical distribution analysis across configurations
- **3D Memory Heatmaps**: Spatial visualization of memory usage patterns
- **Execution Time Analysis**: Performance characteristics across input sizes

#### 2. Model Performance Analysis

- **Accuracy Comparison**: Model performance across different metrics
- **Residual Analysis**: Prediction error patterns and distributions
- **Feature Importance**: Contribution analysis of engineered features
- **Hyperparameter Impact**: Effect of tuning on model performance

#### 3. Robustness Analysis

- **Data Reduction**: Model performance with reduced training data
- **Feature Selection**: Impact of feature reduction on accuracy
- **Cross-Validation**: Stability analysis across different data splits
- **Generalization**: Performance on unseen data configurations

#### 4. Cross-Algorithm Analysis

- **Memory Pattern Comparison**: Different algorithms' memory characteristics
- **Model Transferability**: Cross-algorithm model performance
- **Computational Complexity**: Scaling behavior comparison
- **Prediction Reliability**: Confidence intervals and uncertainty quantification

## 🐳 Docker Environment

The experiment uses a sophisticated containerized execution environment for reproducibility and scalability:

### Multi-stage Build Process

- **Builder stage**: Compiles Rust components (TraceQ) and installs ML dependencies
- **Final stage**: Creates optimized runtime environment with user permissions

### Execution Architecture

- **Docker-in-Docker (DinD)**: Isolated execution for each profiling run
- **Volume management**: Persistent storage across container lifecycles
- **Resource control**: CPU allocation and memory limits for consistent measurements

### Container Features

- **ML Dependencies**: Complete scikit-learn, XGBoost, Optuna stack
- **Visualization Tools**: Matplotlib, Seaborn for comprehensive plotting
- **User permission mapping**: Maintains host permissions for output files

## 📈 Key Findings

This experiment reveals important insights about memory prediction and algorithm characteristics:

### Memory Consumption Patterns

1. **Linear Scaling**: Envelope algorithm shows near-linear memory scaling with volume
2. **Cubic Complexity**: GST3D exhibits cubic scaling due to 3D tensor operations
3. **Kernel Dependencies**: Gaussian Filter memory usage depends on filter kernel size
4. **Dimension Interactions**: Non-linear interactions between inlines, xlines, and samples

### Model Performance Insights

1. **Ensemble Superiority**: Random Forest and XGBoost consistently outperform linear models
2. **Feature Importance**: Volume and logarithmic transforms are most predictive
3. **Algorithm Specificity**: Different algorithms require specialized feature sets
4. **Hyperparameter Impact**: Proper tuning improves accuracy by 15-30%

### Practical Applications

1. **Memory-Aware Chunking**: Models enable intelligent workload partitioning
2. **Resource Planning**: Accurate memory prediction for HPC job scheduling
3. **Algorithm Selection**: Performance-memory trade-off optimization
4. **Scalability Analysis**: Prediction of memory requirements for larger datasets

## 🔗 Dependencies

Core dependencies (see `requirements.txt`):

- **scikit-learn**: Machine learning algorithms and evaluation metrics
- **xgboost**: Gradient boosting framework
- **optuna**: Hyperparameter optimization
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Visualization and plotting
- **seaborn**: Statistical data visualization
- **shap**: Model interpretability and feature importance
- **docker**: Container management for isolated execution

## 📚 Related Work

This experiment supports the theoretical framework presented in the Memory-Aware Chunking thesis:

- **Chapter 4**: Predictive Memory Modeling for Seismic Processing
- **Chapter 5**: Machine Learning Approaches to Memory Estimation
- **Chapter 6**: Memory-Aware Chunking Algorithm Design
- **Appendix C**: Comprehensive Model Evaluation and Validation

## 🔧 Advanced Configuration

### Custom Model Selection

Specify which models to evaluate:

```bash
export MODELS_TO_EVALUATE="random_forest,xgboost,gradient_boosting"
```

### Hyperparameter Tuning

Adjust optimization parameters:

```bash
export OPTUNA_TRIALS=100          # More thorough search
export K_FOLDS=5                  # More robust validation
export RANDOM_STATE=42            # Reproducible results
```

### Performance Thresholds

Configure acceptance criteria:

```bash
export ACCURACY_THRESHOLD=0.05    # 5% accuracy threshold
export SCORE_ACCEPTANCE_THRESHOLD=0.1  # 10% score acceptance
```

### Dataset Scaling

Control experiment scope:

```bash
# Large-scale experiment
export DATASET_FINAL_SIZE=1200
export DATASET_STEP_SIZE=200

# Quick testing
export DATASET_FINAL_SIZE=400
export DATASET_STEP_SIZE=200
```

## 🤝 Contributing

When modifying this experiment:

1. **Maintain ML rigor**: Ensure changes preserve statistical validity and reproducibility
2. **Update documentation**: Modify this README and analysis scripts accordingly
3. **Test thoroughly**: Validate changes across different algorithms and dataset sizes
4. **Follow conventions**: Use existing code style and naming patterns
5. **Preserve model artifacts**: Ensure all trained models and results are properly saved

### Adding New Algorithms

To add a new seismic processing algorithm:

1. Implement the algorithm in the `common.operators` module
2. Add algorithm option to `collect_memory_profile.py`
3. Update the experiment shell script to include profiling runs
4. Modify analysis scripts to handle the new algorithm's characteristics

### Adding New Models

To add a new regression model:

1. Add model constructor to `MODEL_CONSTRUCTORS_HASHMAP`
2. Implement hyperparameter tuning in `tune_model_hyperparams()`
3. Add model building logic in `build_model_from_params()`
4. Update model evaluation lists and documentation

## 📄 License

This experiment is part of the Memory-Aware Chunking thesis research project. Please refer to the main repository
license for usage terms.