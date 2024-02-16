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
    #MamdaniModel.run()

    ## Model based on Sugeno Inference
  #  SugenoModel = Model(DataPipeline(), SugenoArc(), neurofuzzyTrainer())
   # SugenoModel.run()
  
    # My Model
    batch_size = 100
    learning_rate = 1
    n_epochs = 10
    df_name = "dummy2"

    
    
        # Model with MLP arc
    # MLPModel = Model(DataPipeline(df_name),  
    #                  MLP((2,6),2),  
    #                  Trainer(n_epochs=n_epochs))
   # MLPModel.train()


    MyModel = Model(DataPipeline(df_name, batch_size=batch_size), 
                     MyArc(), 
                     MyArcTrainer(n_epochs=n_epochs, learning_rate=learning_rate),
                     Builder(),
                     Classifier())
    MyModel.build_MyArc() 
    MyModel.trainMyArc()
    print(MyModel.class_acc()) # when arc is not trained -> 0.688
  #  MyModel.arc.RuleConsequentLayer.save_weights(df_name)

    #MyModel.build_MyArc_CW()
    # MyModel.build_MyArc() # works 
   # MyModel.build_MyArc_MF()
    # for i in range(1):
    #     MyModel.trainMyArc()
    #     MyModel.build_MyArc_CW()
    #     MyModel.arc.RuleConsequentLayer.save_weights(df_name)


    # print(MLPModel.summary()) 


   # rules = ruleExtractor(MyModel, MLPModel, df_name)
    # rules.print_results()
    
    return 0


if __name__ == "__main__":
   # start_time = time.time()
    main()
  #  print("--- %s seconds ---" % (time.time() - start_time))
