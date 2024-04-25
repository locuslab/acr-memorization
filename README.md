# Rethinking LLM Memorization through the Lens of Adversarial Compression

A compression-based approach to defining and measuring memorization with LLMs. 

This repository contains the code needed to measure memorization in LLMs using input-output compression. This method is presented in [our paper](https://arxiv.org/pdf/2404.15146). This repo was developed collaboratively by Avi Schwarzschild, Zhili Feng, and Pratyush Maini at Carnegie Mellon University in 2024. This code is particularly useful for reproducing the results in our paper on the topic.


## Getting Started

### Requirements
This code was developed and tested with Python 3.10.4. After cloning the repository, you can install the requirements and run our experiments.

To install requirements:

```$ pip install -r requirements.txt```

### Memorization Measurements

Try computing the compression ratio of the first sample in the [Famous Quotes](datasets/famous_quotes.json) dataset with the following command.  
```
% python prompt-minimization-main.py dataset=famous_quotes data_idx=0
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
Open the [analyze_results notebook](analyze_results.ipynb) to process experiments or run [make_table_of_results.py](make_table_of_results.py) to see parse the output folder. The notebook will load all the results into a Pandas DataFrame and then it can be edited (for example by adding cells) to to whatever analysis is needed. The script is a short Python script that will show you the set of experiment names, a table with every entry, and a summary table aggregating across (model, dataset, optimizer) groups. It can also be used with the flag `--experiment_name <experiment-name-0> <experiment-name-1>...` to aggregate results from any number of experiments.

### Optimizing Prompts
We include a simple script of optimizing input tokens to elicit a targeted output from an LLM. This is only one step in finding minimal prompts, but it may be helpful to see how prompt optimization can be done in general.
```
% python example_script.py
```

## Contributing

We encourage anyone using the code to reach out to us directly and open issues and pull requests with questions and improvements!

## Citing Our Work

```
@misc{schwarzschild2024rethinking,
      title={Rethinking LLM Memorization through the Lens of Adversarial Compression}, 
      author={Avi Schwarzschild and Zhili Feng and Pratyush Maini and Zachary C. Lipton and J. Zico Kolter},
      year={2024},
      eprint={2404.15146},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
