# Neuro-Fuzzy Rule Miner

## About
Implementation belonging to the Bachelorsthesis at University of Osnabrück by Julia Hattendorf

Publication: TBA

You can listen to the podcast episode about the topic [here](https://cogsci-journal.uni-osnabrueck.de/podcast/9-fuzzy-neural-networks-brain-to-brain-with-julia-hattendorf/)

## Getting Started

### Used Packages
- Numpy 1.21.4 
- Python 3.9.7 
- pandas  1.3.4
- matplotlib 3.4.3

### How to Install and Run the Project

- Clone the Repository or Download the ZIP File and extract.
- Follow instructions from /set_up/cheatsheet.txt
- Open the main.py either by navigating with the terminal or by using your preferred IDE 
- Make sure the environment that includes the required packages is activated
- run

## Explanations

### About Neuro-Fuzziness <a class="anchor" id="neuro-fuzzy"></a>

#### What?
- neuro-fuzzy = models that adjust fuzzy sets and rules according to neural networks tuning techniques
- neuro-fuzzy system = when the model can act as a fuzzy system once it is trained

#### Why?
- Knowledge acquisition directly from data
- White box neural nets

#### How?

<img align="right" width="400" src="https://github.com/juelha/NeuroFuzzyRuleMiner/blob/main/doc/figures/nf_annot.svg" hspace="50">

The architecture is comprised of three layers: for fuzzification, for the IF-part of a rule and for the THEN-part of a rule.
The propagation of the input through the architecture is computed as follows:


**Fuzzification-Layer:**

The input vector $\vec{x}$ is fuzzified. Each input $x_i$ has its own set of membership functions $MF_i$, for example: $MF_i = (MF_{low}, MF_{medium}, MF_{high})$.  The outputs of this layer can be referred to as the degrees of membership, $\mu_{ij}$, of an input $x_i$ to a function $MF_{ij}$.
$$ \mu_{ij} = MF_{ij}(x_i) $$


**IF-Layer:**

The fuzzified inputs are combined via a T-norm operation, here multiplication. This represents the if-part of the fuzzy rule. The outputs of this layer is referred to as the rule strengths $\vec{R} = (r_1, r_2, ..., r_{m^n})$, where $\rho$ = (1, ..., $m^n$) for $n$ inputs and $m$ membership functions.
$$ R_\rho = \text{T-norm} (\mu_{ij}, \mu_{(i+1)j}, ..., \mu_{nm}) = \mu_{ij} * \mu_{(i+1)j} * ... * \mu_{nm} $$

**THEN-Layer:**

The rule strengths are mapped to classes by one-hot encoded class weights.
$$ y_\rho = R_\rho * class weight_\rho $$


### Dataset <a id="data"></a>

<img align="right" width="600" src="https://github.com/juelha/NeuroFuzzyRuleMiner/blob/main/doc/figures/iris_scatter.png" hspace="10">

The Iris dataset is a popular classification problem where four features were measured for the three iris species setosa, versicolor, virginica.

On the right is a scatter plot of the Iris Dataset. The first 50 samples belong to the species setosa, the samples from 50 to 100 to versicolor, and the samples from 100 to 150 to virginica.


### Running the model

<img align="right" width="500" src="https://github.com/juelha/NeuroFuzzyRuleMiner/blob/main/doc/figures/repo_struct.svg" hspace="10">


Each operation that is performed on the neuro-fuzzy architecture is implemented in a script, i.e. Builder, Trainer, Classifier, and Rule Miner.

**Builder:**
- initializes the free parameters: parameters of the membership functions, and
class weights

**Trainer:**
- trains the parameters of the membership functions with gradient descent

**Rule Miner:**
- extracts the IF-THEN rules from a trained neuro-fuzzy model  

**Classifier:**
- propagates a sample through the model and outputs the class with the highest activation
