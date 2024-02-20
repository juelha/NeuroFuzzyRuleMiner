# custom packagess
from model_pkg import *
from neurofuzzy_pkg import * 
from tests_pkg import *
from neurofuzzy_pkg import utils
import time


def main():
    """
    Main Program of the project:
    1. Train NeuroFuzzy model
        - The NeuroFuzzy model will be trained as a classifier 
        - The trained weights (parameters of membership functions and classweights) will give us the rules
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


    # My Model
    learning_rate = 1
    n_epochs = 5
    df_name = "iris"
    n_participants = 4
    fuzzy_labels = ["small", "medium", "high"]
    lingusitic_output = ["Setosa", "Versicolour", "Virginica"]

    df_name = "dummy2"
    n_participants = 2
    fuzzy_labels = ["low", "medium", "high"]
    lingusitic_output = ["low","high"]


    # df_name = "xor"
    # fuzzy_labels = ["false", "true"]
    # n_participants = 2
    # lingusitic_output = ["false","true"]


    MyModel = Model(DataPipeline(df_name), 
                     MyArc(fuzzy_labels, n_participants, len(lingusitic_output)), 
                     MyArcTrainer(n_epochs=n_epochs, learning_rate=learning_rate),
                     Builder(),
                     Classifier())
    MyModel.build_MyArc() 
    MyModel.trainMyArc()
    print(MyModel.class_acc()) # when arc is not trained -> 0.688

    # for i in range(1):
    #     MyModel.trainMyArc()
    #     MyModel.build_MyArc_CW()
    #     MyModel.arc.RuleConsequentLayer.save_weights(df_name)


    # print(MLPModel.summary()) 


    MyRules = RuleMiner(MyModel, df_name, fuzzy_labels, lingusitic_output)
    MyRules.get_best_rules(inputs=MyModel.data.inputs)
   # MyRules.print_results()
    
    return 0


if __name__ == "__main__":
   # start_time = time.time()
    main()
  #  print("--- %s seconds ---" % (time.time() - start_time))
