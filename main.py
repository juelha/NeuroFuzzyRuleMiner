# custom packagess
from model_pkg import *
from neurofuzzy_pkg import * 
from tests_pkg import *
from neurofuzzy_pkg import utils


def main():
    """
    Main Program of the project:
    1. Train NeuroFuzzy model
        - The NeuroFuzzy model will be trained as classifier (output is either good|bad yield)
        - The trained weights (parameters of membership functions) will give us the rules
    2. Train MLP model 
        - Trained as classifier to validate rules from NeuroFuzzy model
    3. Extract neuro fuzzy rules 
        - ruleExtractor will extract rules from Neurofuzzy model and check with mlp if they are "valid" 
    """
    
    print(
"███╗░░██╗███████╗██╗░░░██╗██████╗░░█████╗░░░░░░░███████╗██╗░░░██╗███████╗███████╗██╗░░░██╗░░░░░░███╗░░░███╗░█████╗░██████╗░███████╗██╗░░░░░" + ("\n") +
"████╗░██║██╔════╝██║░░░██║██╔══██╗██╔══██╗░░░░░░██╔════╝██║░░░██║╚════██║╚════██║╚██╗░██╔╝░░░░░░████╗░████║██╔══██╗██╔══██╗██╔════╝██║░░░░░" + ("\n") +
"██╔██╗██║█████╗░░██║░░░██║██████╔╝██║░░██║█████╗█████╗░░██║░░░██║░░███╔═╝░░███╔═╝░╚████╔╝░█████╗██╔████╔██║██║░░██║██║░░██║█████╗░░██║░░░░░" + ("\n") +
"██║╚████║██╔══╝░░██║░░░██║██╔══██╗██║░░██║╚════╝██╔══╝░░██║░░░██║██╔══╝░░██╔══╝░░░░╚██╔╝░░╚════╝██║╚██╔╝██║██║░░██║██║░░██║██╔══╝░░██║░░░░░" + ("\n") +
"██║░╚███║███████╗╚██████╔╝██║░░██║╚█████╔╝░░░░░░██║░░░░░╚██████╔╝███████╗███████╗░░░██║░░░░░░░░░██║░╚═╝░██║╚█████╔╝██████╔╝███████╗███████╗" + ("\n") +
"╚═╝░░╚══╝╚══════╝░╚═════╝░╚═╝░░╚═╝░╚════╝░░░░░░░╚═╝░░░░░░╚═════╝░╚══════╝╚══════╝░░░╚═╝░░░░░░░░░╚═╝░░░░░╚═╝░╚════╝░╚═════╝░╚══════╝╚══════╝")


    ## Model based on Mamdani Inference <- WORKS!!!
   # MamdaniModel = Model(DataPipeline(), MamdaniArc(), neurofuzzyTrainer())
  # MamdaniModel.run()

    ## Model based on Sugeno Inference
  #  SugenoModel = Model(DataPipeline(), SugenoArc(), neurofuzzyTrainer())
   # SugenoModel.run()
  
    # My Model
    MyModel = Model(DataPipeline(), MyArc(), neurofuzzyTrainer())
    MyModel.build_MyArc()

    ## Model with MLP arc
    # MLPModel = Model(DataPipeline(),  MLP((4,32),2),  Trainer())

    # MLPModel.train()
    # print(MLPModel.summary()) 


    # rules = ruleExtractor(MamdaniModel, MLPModel)
    # rules.print_results()
    
    return 0


if __name__ == "__main__":

    main()
