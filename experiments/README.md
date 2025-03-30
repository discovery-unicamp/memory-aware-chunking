# Experiments Directory

This directory contains a collection of experiments that explore different aspects of Memory-Aware Chunking.
Each folder is an independent experiment aimed at understanding a specific aspect and answer specific questions.

## How to Run the Experiments

TODO

## Available Experiments

### [`02-predicting-memory-consumption-from-input-shapes`](./02-predicting-memory-consumption-from-input-shapes)

**Objective**:
This experiment aims to predict the memory consumption of Python programs based on the input shape.
The goal is to develop a model that can predict the memory consumption of a Python program based on features extracted
from the input data.

**Key Questions**:

- What features are most relevant for predicting memory consumption?
- What models are best suited for predicting memory consumption?
- How accurate are the predictions made by the model?
- How does the model perform on unseen data?
- What are the limitations of the model?

### [
`03-improving-data-parallelism-using-memory-aware-chunking`](./03-improving-data-parallelism-using-memory-aware-chunking)

**Objective**:
This experiment aims to improve data parallelism in Python programs using Memory-Aware Chunking.
The goal is to use the memory-usage prediction model (developed
on [experiment 02](./02-predicting-memory-consumption-from-input-shapes)) and define the chunk size based on the
predicted memory consumption of the input data.

**Key Questions**:

- How does the chunk size affect the performance of the program?
- Does using Memory-Aware Chunking improve the performance of the program?
- How close Memory-Aware Chunking gets from the optimal chunk size?