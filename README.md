# memorization-in-llms
A compression-based approach to defining and measuring memorization with LLMs.

This repository contains the code needed to measure memorization in LLMs using input-output compression. It was developed collaboratively by Avi Schwarzschild, Zhili Feng, and Pratyush Maini at Carnegie Mellon University in 2024. This code is particularly useful for reproducing the resutls in our paper on the topic.


## Getting Started

### Requirements
This code was developed and tested with Python 3.10.4.

To install requirements:

```$ pip install -r requirements.txt```

### Optimizing Prompts

```
% python example_script.py
```

### Memorization Measurements

```
% python promptmin-main.py
```

### Logging Style and Data Analysis

```
outputs
└── happy-Melissa
        ├── .hydra
        │   ├── config.yaml
        │   ├── hydra.yaml
        │   └── overrides.yaml
        ├── results.json
        └── log.log
```

These output folders can be parsed and analyzed as a DataFrame using Pandas.
Open the [analyze_results notebook](analyze_results.ipynb) to process experiments.

## Contributing

We encourage anyone using the code to reachout to us directly and open issues and pull requests with questions and improvements!

## Citing Our Work


