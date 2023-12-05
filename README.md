# Neuro-fuzzy Logic for Optimal Citrus Leaf Nutrient Norms
A novel approach for finding non-linear agricultural
macronutrient protocols for Citrus x sinensis Valencia
oranges in South Africa
  * [Abstract](#abstract)
  * [Repository structure](#repository-structure)
    + [Overview](#overview)
      - [model_pkg](#model-pkg)
      - [neurofuzzy_pkg](#neurofuzzy-pkg)
      - [results](#results)
      - [tests](#tests)
  * [Getting Started](#getting-started)
  * [About](#about)
  * [Acknowledgements](#acknowledgements)
  * [Obligatory Meme](#obligatory-meme)



<a name="Abstract"/>
<a name="Repository structure"/>
<a name="model_pkg"/>
<a name="neurofuzzy_pkg"/>
<a name="results"/>
<a name="tests"/>
<a name="Running instructions"/>
<a name="About"/>
<a name="Acknowledgements"/>
<a name="Obligatory Meme"/>


## Abstract 

COPY PASTE ABSTRACT HERE 

## Repository structure
### Overview
```bash
.
├── data
├── main.ipynb
├── main.py
├── model_pkg
├── neurofuzzy_pkg
├── README.md
├── results
└── tests_pkg
```
We split up the structure of our project into packages. To run the whole project either clone the Repo and run main.py locally on your machine or download main.ipynb and upload the file to Google Colab. 

#### model_pkg
```bash
├── model_pkg
│   ├── __init__.py
│   ├── DataAugmentation.py
│   ├── DataPipeline.py
│   ├── DataWrangling.py
│   ├── MLP.py
│   ├── Model.py
│   └── Trainer.py
```
This is the base package for any Neural Network Model. 

[`Model.py`](https://github.com/juelha/IANNWTF_FINAL/blob/main/model_pkg/Model.py)is the main class of the package where the Data, Architecture and Trainer come together. 

[`DataPipeline.py`](https://github.com/juelha/IANNWTF_FINAL/blob/main/model_pkg/DataPipeline.py/) is the script containing all functions needed for preparing a panda dataset for the model. 

[`MLP.py`](https://github.com/juelha/IANNWTF_FINAL/blob/main/model_pkg/MLP.py/) is the class containing a simple feed-forward architecture, as well as its forward- and inverse-pass.

[`Trainer.py`](https://github.com/juelha/IANNWTF_FINAL/blob/main/model_pkg/Trainer.py/) is the class responsible for training the model with **Backpropagation**.

Finally we added the scripts used for data wrangling and data augmentation. They do not need to be run, since their results are already saved in github but are added for completion's sake. 


#### neurofuzzy_pkg
```bash
├── neurofuzzy_pkg
│   ├── __init__.py
│   ├── fuzzyLayers
│   │   ├── __init__.py
│   │   ├── DefuzzificationLayer.py
│   │   ├── FuzzificationLayer.py
│   │   ├── RuleAntecedentLayer.py
│   │   ├── RuleConsequentLayer.py
│   │   └── RuleStengthNorm.py
│   ├── MamdaniArc.py
│   └── utils
│       ├── math_funcs.py
│       ├── MFs.py
│       ├── neurofuzzyTrainer.py
│       └── ruleExtractor.py
```

[`/fuzzyLayers`](https://github.com/juelha/IANNWTF_FINAL/tree/main/neurofuzzy_pkg/fuzzyLayers) contains all layers needed for the neuro-fuzzy model.

[`MamdaniArc.py`](https://github.com/juelha/IANNWTF_FINAL/blob/main/neurofuzzy_pkg/MamdaniArc.py) describes the architecture of our neuro-fuzzy network and the instructions for a forward pass of an input vector.

[`/utils`](https://github.com/juelha/IANNWTF_FINAL/tree/main/neurofuzzy_pkg/utils) contains 

- [`math_funcs`](https://github.com/juelha/IANNWTF_FINAL/blob/main/neurofuzzy_pkg/utils/math_funcs.py) is a collection for all math functions needed, to reduce the amount of required packages,
- [`MFs.py`](https://github.com/juelha/IANNWTF_FINAL/blob/main/neurofuzzy_pkg/utils/MFs.py), a collection of Membership functions and the initialization of their parameters, as well as their visualisation,
- [`neurofuzzyTrainer.py`](https://github.com/juelha/IANNWTF_FINAL/blob/main/neurofuzzy_pkg/utils/neurofuzzyTrainer.py), the Trainer of the neuro-fuzzy network subclassed from [`Trainer.py`](https://github.com/juelha/IANNWTF_FINAL/blob/main/model_pkg/Trainer.py/) and rewritten to tune the parameters of the Membership Functions,
- [`ruleExtractor.py`](https://github.com/juelha/IANNWTF_FINAL/blob/main/neurofuzzy_pkg/utils/ruleExtractor.py), the class used for extracting the fuzzy rules from the neuro-fuzzy network and validating them with the MLP.


#### results
```bash
├── results
│   ├── bad_yield.csv
│   ├── figs
│   └── good_yield.csv
```
This directory contains our results that are saved automatically during training. 
[`bad_yield.csv`] and [`good_yield.csv`] contain the final rule base for either yield.
The folder [`figs`] () is the destination of the figures generated during training.


#### tests
```bash
└── tests_pkg
    ├── __init__.py
    ├── test_mlp.py
    ├── test_neurofuzzy.py
    └── test_rule_extract.py

```
Package for testing the main parts of the program: the MLP, the neurofuzzy model and the rule extraction.

## Getting Started

### Used Packages
- Tensorflow 2.7.0
- keras 2.7.0  
- Numpy 1.21.4 
- Python 3.9.7 
- pandas  1.3.4
- matplotlib 3.4.3

### How to Install and Run the Project

#### Option: run online
- To run this project online we suggest using Google Colab.
- Download the file [`main.ipynb`](https://github.com/juelha/IANNWTF_FINAL/blob/main/main.ipynb) 
- Upload it to Google Colab, the whole project will be automatically cloned in the first cell
- run

#### Option: run locally
- Clone the Repository or Download the ZIP File and extract.
- Follow instructions from /set_up/cheatsheet.txt
- Open the main.py either by navigating with the terminal or by using your preferred IDE 
- Make sure the environment that includes the required packages is activated
- run

## About
Final project for the course Artificial Neural Networks with TensorFlow at University of Osnabrück by J. Hattendorf, S. Scholle, T. Schuchort

## Acknowledgements 
Special thanks to [`CRI`](https://www.citrusres.com/) for providing us with the datasets.


## Obligatory Meme
<img width="500" alt="java 8 and prio java 8  array review example" src="https://external-preview.redd.it/vH65sWEO5o9xGrUtbu_EzYLKh8cvzU86nOKCsCqZLJo.jpg?width=640&crop=smart&auto=webp&s=22ce61e77ce054375a21fbd8e1fda594674b60aa">

[`source`](https://www.reddit.com/r/ProgrammerHumor/comments/5smlfq/fuzzy_logic_example_cat/)
