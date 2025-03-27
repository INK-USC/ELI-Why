# ELI-Why
Benchmark and Code for ELI-Why

The `ELI-Why` benchmark is stored as `ELI_Why.jsonl`.

## Project Structure

- **RQ1/**  
- **Individual Metrics/**
Contains scripts for running individual metrics and analyses. To run individual metrics, navigate to the `RQ1/` directory and execute:
```bash
python3 individual_metric_readability_analysis.py --model GPT4o --set full
```
Alternatively, to run all combinations, use:
```bash
python3 individual_metric_readability_analysis.py --all
```
- **TESDIFF(Similarity) Metric/**
Contains scripts for running TESDIFF metric and analyses. To run it, in the `RQ1/` directory and execute:
```bash
python3 tesdiff.py --all
```
