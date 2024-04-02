# custom packagess
from model_pkg import *
from neurofuzzy_pkg import * 
from neurofuzzy_pkg import utils
import time


def main():
    """
    Main Program of the project:
    1. Train NeuroFuzzy model
        - The NeuroFuzzy model will be trained as a classifier 
        - The trained weights (parameters of membership functions and classweights) will give us the rules
    3. Extract neuro fuzzy rules 
        - RuleMiner will extract rules from Neurofuzzy model 
    """
    
    print(
"███╗░░██╗███████╗██╗░░░██╗██████╗░░█████╗░░░░░░░███████╗██╗░░░██╗███████╗███████╗██╗░░░██╗░░░░░░███╗░░░███╗░█████╗░██████╗░███████╗██╗░░░░░" + ("\n") +
"████╗░██║██╔════╝██║░░░██║██╔══██╗██╔══██╗░░░░░░██╔════╝██║░░░██║╚════██║╚════██║╚██╗░██╔╝░░░░░░████╗░████║██╔══██╗██╔══██╗██╔════╝██║░░░░░" + ("\n") +
"██╔██╗██║█████╗░░██║░░░██║██████╔╝██║░░██║█████╗█████╗░░██║░░░██║░░███╔═╝░░███╔═╝░╚████╔╝░█████╗██╔████╔██║██║░░██║██║░░██║█████╗░░██║░░░░░" + ("\n") +
"██║╚████║██╔══╝░░██║░░░██║██╔══██╗██║░░██║╚════╝██╔══╝░░██║░░░██║██╔══╝░░██╔══╝░░░░╚██╔╝░░╚════╝██║╚██╔╝██║██║░░██║██║░░██║██╔══╝░░██║░░░░░" + ("\n") +
"██║░╚███║███████╗╚██████╔╝██║░░██║╚█████╔╝░░░░░░██║░░░░░╚██████╔╝███████╗███████╗░░░██║░░░░░░░░░██║░╚═╝░██║╚█████╔╝██████╔╝███████╗███████╗" + ("\n") +
"╚═╝░░╚══╝╚══════╝░╚═════╝░╚═╝░░╚═╝░╚════╝░░░░░░░╚═╝░░░░░░╚═════╝░╚══════╝╚══════╝░░░╚═╝░░░░░░░░░╚═╝░░░░░╚═╝░╚════╝░╚═════╝░╚══════╝╚══════╝")

    learning_rate = 1
    n_epochs = 1

    # parameters for running iris dataset
    df_name = "iris"
    n_participants = 4
    fuzzy_labels = ["small" , "medium","high"] 
    lingusitic_output = ["Setosa", "Versicolour", "Virginica"]

    # parameters for running xor dataset
    # df_name = "xor"
    # fuzzy_labels = ["false", "true"]
    # n_participants = 2
    # lingusitic_output = ["false","true"]


    MyModel = Model(DataPipeline(df_name), 
                     MyArc(fuzzy_labels, n_participants, len(lingusitic_output)), 
                     MyArcTrainer(n_epochs=n_epochs, learning_rate=learning_rate),
                     Builder(),
                     Classifier())
   # MyModel.build_MyArc() 
    MyModel.load_MyArc()
    MyModel.trainMyArc()
    print(MyModel.class_acc()) 



    MyRules = RuleMiner(MyModel, df_name, fuzzy_labels, lingusitic_output)
    MyRules.get_best_rules(inputs=MyModel.data.inputs)
    

    return 0


if __name__ == "__main__":
    main()

