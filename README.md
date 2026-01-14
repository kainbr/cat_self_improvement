# Self-Improvement for Computerized Adaptive Testing

Code for _Self-Improvement for Computerized Adaptive Testing_ (accepted at ECML 2025, [Springer Nature Link](https://link.springer.com/chapter/10.1007/978-3-032-05981-9_5)).

Yannick Rudolph*, Kai Neubauer*, and Ulf Brefeld (* equal contribution)

> Computerized adaptive testing (CAT) allows for assessing latent traits and abilities of students with fewer items and in less time due to an individualized item selection algorithm based on previous responses. Following recent machine learning solutions to CAT, we study learning both the underlying response model for cognitive diagnosis and a policy for the item selection algorithm jointly from offline training data. While the task of the response model is to predict performances on all unseen items for a user, the goal of the policy is to select the subset of items which maximizes information for the response model. Since subset selection is a combinatorial problem, we propose to leverage an iterative self-improvement approach to policy learning from the field of neural combinatorial optimization while accounting for interdependencies between response model and policy. We specifically focus on the generalization capabilities of transformer-based models and, in contrast to related work, do not rely on optimization of local variables during inference. We report on empirical results.

Cite this work:
```bibtex
@inproceedings{rudolph2025self,
  title={Self-improvement for Computerized Adaptive Testing},
  author={Rudolph, Yannick and Neubauer, Kai and Brefeld, Ulf},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  year={2025}
}
```

## Installation

We use [conda](https://pip.pypa.io/en/stable/) to manage dependencies. 
To create a new conda environment with the necessary dependencies installed, execute the following command:

```bash
conda env create --file env.yaml
```

By default, this will create a conda environment named `cat`. To activate this environment, use:

```bash
conda activate cat
```

In our experiments, we use [EduCAT](https://github.com/bigdata-ustc/EduCAT) for CAT-related baselines. 
However, to ensure a fair comparison with our experimental setup, we had to align the evaluation.
Consequently, the code is copied in this repo (see folder `educat`, repo cloned from hash '0f191de', license: MIT), and relevant modifications are highlighted within the code.

## Data

The CAT experiments use the Eedi dataset from the [NeurIPS 2020 Education challenge](https://eedi.com/projects/neurips-education-challenge). 
Simply download the dataset and extract it to a location of your choice (should contain the subdirectories `images`, `metadata`, `test_data` and `train_data`). 
This code base assumes that the data is located at `/home/knowledge-tracing/data/neurips_education_challenge`.
You can change this either by editing the data configs in `pkg/config/data` or providing the respective command-line arguments (see below).

## Run experiments

We use [hydra](http://hydra.cc) for experiment configuration and batch/parallel execution.
The corresponding config files are located at `pkg/config`.

Please refer to the `scripts` folder for notebooks containing experiment evaluations and code for generating the tables and figures.

### CAT

#### EduCAT baselines

```bash
python educat_reproduce.py -m +exps=educat_baselines
```

#### Random

```bash
python cat.py -m +exps=random
```

#### Uncertainty

```bash
python cat.py -m +exps=uncertainty
```

#### Self-improvement

```bash
python cat.py -m +exps=self_improvement
```


### Synthetic data

```bash
python cat.py -m +exps=synthetic_1
```

### Knowledge tracing

The code for the knowledge tracing implementation of our model, along with baselines and experiments, is contained in a separate repository. 
Currently, the code in that repository is undergoing review. 
Once it’s published, we’ll provide a link to the repository here. 
