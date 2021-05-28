# Import Module
import os


os.getcwd()

# Change the directory
os.chdir("/Users/sherry/Desktop/processed")
i=0
# iterate through all file
for file in os.listdir():
	# Check whether file is in text format or not
	if file.endswith(".JSON"):
                print(i+1)
                i= i+1
            #command = "./eld /Users/sherry/Desktop/extractable-three-fold-lin+nonlin/train_data/" + file +"-extractPredicates -labelSimpleGeneratedPredicates -getHornGraph:monoDirectionLayerGraph -abstract -solvabilityTimeout:300 -t:1200"
           
            #os.system(command)
    #os.system("")


