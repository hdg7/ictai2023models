# ICTAI 2023 Models Generation and Evaluation

This repo is to generate the models for the following paper:
```
@incollection{menendez2022measuring,
  title={Measuring Machine Learning Robustness in front of Static and Dynamic Adversaries},
  author={Menendez, Hector D.},
  booktitle={Measuring Machine Learning Robustness in front of Static and Dynamic Adversaries},
  year={2022},
  publisher={IEEE 34rd International Conference on Tools with Artificial Intelligence (ICTAI)}
}
```
Please, cite the paper if you use the tool.

The code uses MLighter, if you want to use the tool, please visit:
https://github.com/hdg7/mlighter/

## The files

The files from the repo just generate the models and apply the robustness metrics to them. The files are:

* classifiers.py: creates the models and applies basic noise to them.
* geneticAlgorithm.py: applies the GA to the models.
* classifiersNames.txt: to select the classifiers that will be evaluated.
