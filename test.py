import glob
import json
import os
from typing import Any, Dict
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import seaborn
import tensorflow as tf
import tf2_gnn
from tf2_gnn.cli_utils.training_utils import train, log_line, make_run_id
from tf2_gnn.data import DataFold, HornGraphSample, HornGraphDataset
from tf2_gnn.models import InvariantArgumentSelectionTask, InvariantNodeIdentifyTask

from Miscellaneous import pickleWrite, pickleRead, drawBinaryLabelPieChart
from dotToGraphInfo import parseArgumentsFromJson
from utils import plot_confusion_matrix,get_recall_and_precision,plot_ROC,assemble_name,my_round_fun

def train_on_graphs(benchmark_name="unknown",label="rank",force_read=False,train_n_times=1,path="../",file_type=".smt2",json_type=".JSON",form_label=False,GPU=False,pickle=True,hyper_parameters={}):
    gathered_nodes_binary_classification_task = ["predicate_occurrence_in_SCG", "argument_lower_bound_existence",
                                                 "argument_upper_bound_existence", "argument_occurrence_binary",
                                                 "template_relevance", "clause_occurrence_in_counter_examples_binary"]

    graph_type=json_type[1:json_type.find(".JSON")]
    print("graph_type",graph_type)
    nodeFeatureDim = hyper_parameters["nodeFeatureDim"] #64
    parameters = tf2_gnn.GNN.get_default_hyperparameters()
    parameters['graph_type'] = graph_type  # hyperEdgeHornGraph or layerHornGraph
    #parameters["message_calculation_class"]="rgcn"#rgcn,ggnn,rgat
    #parameters['num_heads'] = 2
    #parameters["global_exchange_dropout_rate"]=0
    #parameters["layer_input_dropout_rate"]=0
    #parameters["gnn_layer_input_dropout_rate"]=0
    #parameters["graph_aggregation_dropout_rate"]=0
    #parameters["regression_mlp_dropout"]=0
    #parameters["scoring_mlp_dropout_rate"]=0
    #parameters["residual_every_num_layers"]=10000000
    parameters['hidden_dim'] = nodeFeatureDim #64
    #parameters["num_edge_MLP_hidden_layers"]
    parameters['num_layers'] = hyper_parameters["num_layers"]
    parameters['node_label_embedding_size'] = nodeFeatureDim
    parameters['max_nodes_per_batch']=hyper_parameters["max_nodes_per_batch"] #todo: _batch_would_be_too_full(), need to extend _finalise_batch() to deal with hyper-edge
    parameters['regression_hidden_layer_size'] = hyper_parameters["regression_hidden_layer_size"]
    parameters["benchmark"]=benchmark_name
    parameters["label_type"]=label
    parameters ["gathered_nodes_binary_classification_task"]=gathered_nodes_binary_classification_task
    parameters["threshold"]=hyper_parameters["threshold"]
    max_epochs = 20
    patience = 20
    # parameters["add_self_loop_edges"]=False
    # parameters["tie_fwd_bkwd_edges"]=True

    these_hypers: Dict[str, Any] = {
        "optimizer": "Adam",  # One of "SGD", "RMSProp", "Adam"
        "learning_rate": 0.001,
        "learning_rate_decay": 0.98,
        "momentum": 0.85,
        "gradient_clip_value": 1, #1
        "use_intermediate_gnn_results": False,
    }
    parameters.update(these_hypers)
    #get dataset
    dataset=HornGraphDataset(parameters)
    #dataset._read_from_pickle = pickle
    if pickle==True:
        if force_read==True:
            write_graph_to_pickle(benchmark=benchmark_name,  data_fold=["train", "valid", "test"],
                                  label=label,path=path,
                                  file_type=file_type,max_nodes_per_batch=parameters['max_nodes_per_batch'],graph_type=graph_type)
        else:
            print("Use pickle data for training")
        #if form_label == True and not os.path.isfile("../pickleData/" + label + "-" + benchmark_name + "-gnnInput_train_data.txt"):
        if form_label == True:
            form_GNN_inputs_and_labels(label=label, datafold=["train", "valid", "test"], benchmark=benchmark_name,graph_type=graph_type,gathered_nodes_binary_classification_task=gathered_nodes_binary_classification_task)
        else:
            print("Use label in pickle data for training")
    elif pickle==False:
        dataset._path=path
        dataset._json_type=json_type
    if GPU==True:
        dataset._use_worker_threads=False #solve Failed setting context: CUDA_ERROR_NOT_INITIALIZED: initialization error
    dataset.load_data([DataFold.TRAIN,DataFold.VALIDATION,DataFold.TEST])
    parameters["node_vocab_size"]=dataset._node_vocab_size
    parameters["class_weight"]= {}
    parameters["class_weight_fold"] = {}
    def log(msg):
        log_line(log_file, msg)

    train_loss_list_average = []
    valid_loss_list_average = []
    test_loss_list_average = []
    mean_loss_list_average = []

    train_loss_average = []
    valid_loss_average = []
    test_loss_average = []
    best_valid_epoch_average = []
    accuracy_average=[]
    trained_model_path=None
    error_loaded_model_average = []
    error_memory_model_average=[]

    for n in range(train_n_times): # train n time to get average performance, default is one
        # initial different models by different training task
        if label == "argument_identify" or label == "control_location_identify" or label == "argument_identify_no_batchs": #all nodes binary classification task
            model = InvariantNodeIdentifyTask(parameters, dataset)
        elif label=="predicate_occurrence_in_clauses" or label=="argument_lower_bound" or label=="argument_upper_bound":#gathered nodes single output regression task
            model = InvariantArgumentSelectionTask(parameters, dataset)
        elif label in gathered_nodes_binary_classification_task: #gathered nodes binary classification task
            model = InvariantNodeIdentifyTask(parameters, dataset)
        elif label=="argument_bound": #gathered nodes two outputs regression task
            patience=max_epochs
            model = InvariantArgumentSelectionTask(parameters, dataset)
        else:
            model = InvariantArgumentSelectionTask(parameters, dataset)

        #train
        quiet=False
        model_name="GNN"
        task_name="Argument_selection"
        run_id = make_run_id(model_name, task_name)
        save_dir=os.path.abspath("trained_model")
        log_file = os.path.join(save_dir, f"{run_id}.log")
        # import multiprocessing
        # process_train = multiprocessing.Process(train, args=(model,dataset,log,run_id,200,20,save_dir,quiet,None))
        # process_train.start()
        # process_train.join()

        trained_model_path,train_loss_list,valid_loss_list,best_valid_epoch,train_metric_list,valid_metric_list,weights_list_train ,weights_list_valid = train(
            model=model,
            dataset=dataset,
            log_fun=log,
            run_id=run_id,
            max_epochs=max_epochs,
            patience=patience,
            save_dir=save_dir,
            quiet=quiet,
            aml_run=None)
        #predict
        print("trained_model_path", trained_model_path)

        test_data = dataset.get_tensorflow_dataset(DataFold.TEST)
        weights_dict_dataset_saved = {} #loaded weights from saved file
        # use model in memory
        _, _, test_results,weights_dict_dataset = model.run_one_epoch(test_data, training=False, quiet=quiet)
        test_metric, test_metric_string = model.compute_epoch_metrics(test_results)
        predicted_Y_loaded_model_from_memory = model.predict(test_data)
        rounded_predicted_Y_loaded_model_from_memory=my_round_fun(predicted_Y_loaded_model_from_memory,threshold=hyper_parameters["threshold"])
        print("test_metric_string model from memory", test_metric_string)
        print("test_metric model from memory", test_metric)
        print("predicted_Y_loaded_model_from_memory",tf.math.round(predicted_Y_loaded_model_from_memory))
        #load model from files
        loaded_model = tf2_gnn.cli_utils.model_utils.load_model_for_prediction(trained_model_path, dataset)
        _, _, test_results,weights_dict_dataset_saved = loaded_model.run_one_epoch(test_data, training=False, quiet=quiet)
        test_metric, test_metric_string = loaded_model.compute_epoch_metrics(test_results)
        predicted_Y_loaded_model = loaded_model.predict(test_data)
        rounded_predicted_Y_loaded_model=my_round_fun(predicted_Y_loaded_model,threshold=hyper_parameters["threshold"])
        print("predicted_Y_loaded_model",tf.math.round(predicted_Y_loaded_model))

        print("test_metric_string",test_metric_string)
        print("test_metric",test_metric)

        true_Y=[]
        for data in iter(test_data):
            #print("wodedata debug",data[0]) #input
            true_Y.extend(np.array(data[1]["node_labels"]))



        #print("true_Y", true_Y)
        #print("predicted_Y_loaded_model_from_memory", predicted_Y_loaded_model_from_memory)
        #print("predicted_Y_loaded_model", predicted_Y_loaded_model)
        error_loaded_model = (lambda : tf.keras.losses.MSE(true_Y, predicted_Y_loaded_model) \
            if label not in gathered_nodes_binary_classification_task else tf.keras.losses.binary_crossentropy(true_Y, predicted_Y_loaded_model))()
        print("\n error of loaded_model", error_loaded_model)

        error_memory_model = (lambda : tf.keras.losses.MSE(true_Y, predicted_Y_loaded_model) \
            if label not in gathered_nodes_binary_classification_task else tf.keras.losses.binary_crossentropy(true_Y, predicted_Y_loaded_model_from_memory))()
        print("\n error of loaded_model", error_loaded_model)

        mse_mean = tf.keras.losses.MSE([np.mean(true_Y)]*len(true_Y), true_Y)
        print("\n mse_mean_Y_and_True_Y", mse_mean)
        mean_loss_list=mse_mean
        num_correct = tf.reduce_sum(tf.cast(tf.math.equal(true_Y, tf.math.round(predicted_Y_loaded_model)),tf.int32))
        accuracy = num_correct / len(predicted_Y_loaded_model)
        accuracy_average.append(accuracy)

        test_loss_list_average.append(predicted_Y_loaded_model_from_memory)
        mean_loss_list_average.append(mean_loss_list)
        error_loaded_model_average.append(error_loaded_model)
        error_memory_model_average.append(error_memory_model)
        test_loss_average.append(predicted_Y_loaded_model[-1])

        train_loss_list_average.append(train_loss_list)
        valid_loss_list_average.append(valid_loss_list)
        train_loss_average.append(train_loss_list[-1])
        valid_loss_average.append(valid_loss_list[-1])
        best_valid_epoch_average.append(best_valid_epoch)


       

    # get aberage training performance
    train_loss_list_average = np.mean(train_loss_list_average, axis=0)
    valid_loss_list_average = np.mean(valid_loss_list_average, axis=0)
    test_loss_list_average = np.mean(test_loss_list_average, axis=0)
    mean_loss_list_average = np.mean(mean_loss_list)
    error_loaded_model_average = np.mean(error_loaded_model_average)
    error_memory_model_average = np.mean(error_memory_model_average)
    train_loss_average = np.mean(train_loss_average)
    valid_loss_average = np.mean(valid_loss_average)
    best_valid_epoch_average = np.mean(best_valid_epoch_average)
    mean_accuracy = np.mean(accuracy_average)
    write_accuracy_to_log(label, benchmark_name, accuracy_average, best_valid_epoch_average, graph_type)
    # visualize results
    draw_training_results(train_loss_list_average, valid_loss_list_average,
                          mean_loss_list_average,
                          error_memory_model_average, true_Y, predicted_Y_loaded_model, label,
                          benchmark_name, graph_type,gathered_nodes_binary_classification_task,hyper_parameters)
    draw_weights_results(label,benchmark_name, graph_type,gathered_nodes_binary_classification_task,hyper_parameters,weights_list_train)                      
    write_train_results_to_log(dataset, predicted_Y_loaded_model, train_loss_average,
                               valid_loss_average, error_loaded_model, mean_loss_list, accuracy_average,
                               best_valid_epoch_average,hyper_parameters,
                               benchmark=benchmark_name, label=label, graph_type=graph_type)

    pickleWrite(parameters, benchmark_name+"-"+label+"-parameters","/Users/sherry/Downloads/Systematic-Predicate-Abstraction-using-Machine-Learning-master/Heuristic_selection/src/trained_model/")

    return trained_model_path


def write_accuracy_to_log(label, benchmark, accuracy_list, best_valid_epoch_list, graph_type):
    mean_accuracy = np.mean(accuracy_list)
    best_valid_epoch_average = np.mean(best_valid_epoch_list)
    with open("trained_model/" + label + "-" + graph_type + "-" + benchmark + ".log", 'w') as out_file:
        out_file.write("accuracy_list:" + str(accuracy_list) + "\n")
        out_file.write("accuracy mean:" + str(mean_accuracy) + "\n")
        out_file.write("best_valid_epoch_list:" + str(best_valid_epoch_list) + "\n")
        out_file.write("best_valid_epoch_average:" + str(best_valid_epoch_average) + "\n")


def draw_training_results(train_loss_list_average, valid_loss_list_average,
                          mean_loss_list_average,
                          mse_loaded_model_average, true_Y, predicted_Y_loaded_model, label,
                          benchmark_name, graph_type,gathered_nodes_binary_classification_task,hyper_parameters):
    # mse on train, validation,test,mean
    plt.plot(train_loss_list_average, color="blue")
    plt.plot(valid_loss_list_average, color="green")
    plt.plot([mean_loss_list_average] * len(train_loss_list_average), color="red")
    plt.plot([mse_loaded_model_average] * len(train_loss_list_average), color="black")
    y_range=[0,max(max(train_loss_list_average),max(valid_loss_list_average))]
    upper_bound=1#max(y_range)
    grid=upper_bound/10
    plt.ylim([min(y_range), upper_bound])
    plt.yticks(np.arange(min(y_range), upper_bound, grid))
    #plt.yscale('log')
    #plt.ylim(bottom=0, top=15)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    train_loss_legend = mpatches.Patch(color='blue', label='train_loss')
    valid_loss_legend = mpatches.Patch(color='green', label='valid_loss')
    mean_loss_legend = mpatches.Patch(color='red', label='mean_loss')
    test_loss_legend = mpatches.Patch(color='black', label='test_loss')
    plt.legend(handles=[train_loss_legend, valid_loss_legend, mean_loss_legend, test_loss_legend])
    plot_name=assemble_name(label,graph_type,benchmark_name,"nodeFeatureDim",str(hyper_parameters["nodeFeatureDim"]),"num_layers",str(hyper_parameters["num_layers"]),"regression_hidden_layer_size",str(hyper_parameters["regression_hidden_layer_size"]),"threshold",str(hyper_parameters["threshold"]))
    plt.savefig("trained_model/" + plot_name + ".png")
    plt.clf()
    # plt.show()

    if label in gathered_nodes_binary_classification_task: # confusion matrix on true y and predicted y

        saving_path_confusion_matrix="trained_model/" + plot_name+ "-confusion_matrix.png"
        saving_path_roc = "trained_model/" + plot_name + "-ROC.png"
        recall,precision,f1_score,false_positive_rate=get_recall_and_precision(true_Y,my_round_fun(predicted_Y_loaded_model,threshold=hyper_parameters["threshold"]),verbose=True)
        plot_confusion_matrix(predicted_Y_loaded_model,true_Y,saving_path_confusion_matrix,recall=recall,precision=precision,f1_score=f1_score)
        plot_ROC(false_positive_rate,recall,saving_path_roc)


    else:
        # scatter on true y and predicted y
        a = plt.axes(aspect='equal')
        plt.scatter(true_Y, predicted_Y_loaded_model)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        lims = [0, np.max([np.max(true_Y), np.max(predicted_Y_loaded_model)])]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
        plt.savefig("trained_model/" + plot_name + "-scatter.png")
        plt.clf()

    # error distribution on true y and predicted y
    if np.min(predicted_Y_loaded_model) == float("-inf") or np.max(predicted_Y_loaded_model) == float("inf") or np.min(
            true_Y) == float("-inf") or np.max(true_Y) == float("inf"):
        pass
    else:
        error = np.array(predicted_Y_loaded_model) - np.array(true_Y)
        plt.hist(error, bins=25)
        plt.xlabel("Prediction Error [occurence]")
        _ = plt.ylabel("Count")
        plt.savefig("trained_model/" + plot_name+ "-error-distribution.png")
        plt.clf()

    
def draw_weights_results(label,benchmark_name, graph_type,gathered_nodes_binary_classification_task,hyper_parameters,weights_list_train):
  
    draw_train_weights = [i["embedded_weights"] for i in weights_list_train]
    draw_train_weights2 = [j["regression_weights"] for j in weights_list_train]
    print("weights_list_train", draw_train_weights)
    print("weights_list_train2", draw_train_weights2)
    plt.plot(draw_train_weights, color="blue")
    plt.plot(draw_train_weights2, color="red")
    y_range=[-0.1,0.1]
    plt.ylim(y_range)
    plt.yticks(np.arange(min(y_range), max(y_range), 0.010))
    plt.ylabel('Weights')
    plt.xlabel('epochs')
    embedded_weights_legend = mpatches.Patch(color='blue', label='embedded_weights')
    egression_weights_legend = mpatches.Patch(color='red', label='regression_weights')

    plt.legend(handles=[egression_weights_legend, embedded_weights_legend])
    #regression_weights_legend = mpatches.Patch(color='green', label='regression_weights')
    #plt.legend(handles=[embedded_weights_legend, regression_weights_legend])
    plot_name=assemble_name(label,graph_type,benchmark_name,"Weights Graph",str(hyper_parameters["nodeFeatureDim"]),"num_layers",str(hyper_parameters["num_layers"]),"regression_hidden_layer_size",str(hyper_parameters["regression_hidden_layer_size"]),"threshold",str(hyper_parameters["threshold"]))
    plt.savefig("trained_model/" + plot_name + ".png")
    plt.clf()

def draw_pixel_graph(label,benchmark_name, graph_type,gathered_nodes_binary_classification_task,hyper_parameters,weights_list_train):
    draw_train_weights2 = [j["regression_weights"] for j in weights_list_train]
    
def write_train_results_to_log(dataset, predicted_Y_loaded_model, train_loss, valid_loss, mse_loaded_model_list,
                               mean_loss_list, accuracy_list,best_valid_epoch, hyper_parameters,benchmark="unknown", label="rank", graph_type="hyperEdgeHornGraph"):
    mean_loss_list_average = np.mean(mean_loss_list)
    mse_loaded_model_average = np.mean(mse_loaded_model_list)
    mean_accuracy = np.mean(accuracy_list)
    log_name = assemble_name(label, graph_type, benchmark, "nodeFeatureDim",
                              str(hyper_parameters["nodeFeatureDim"]), "num_layers",
                              str(hyper_parameters["num_layers"]), "regression_hidden_layer_size",
                              str(hyper_parameters["regression_hidden_layer_size"]))
    with open("trained_model/" + log_name+ ".log", 'a') as out_file:
        out_file.write("best_valid_epoch:" + str(best_valid_epoch) + "\n")
        out_file.write("train loss:" + str(train_loss) + "\n")
        out_file.write("valid loss:" + str(valid_loss) + "\n")
        out_file.write("test loss list:" + str(mse_loaded_model_list) + "\n")
        out_file.write("mean test loss:" + str(mse_loaded_model_average) + "\n")

        out_file.write("mean loss list:" + str(mean_loss_list) + "\n")
        out_file.write("mean mean loss:" + str(mean_loss_list_average) + "\n")

        out_file.write("accuracy list:" + str(accuracy_list) + "\n")
        out_file.write("mean accuracy:" + str(mean_accuracy) + "\n")

        predicted_argument_lists = get_predicted_label_list_divided_by_file(dataset, predicted_Y_loaded_model)
        mse_list = []
        for predicted_label, arguments, file_name in zip(predicted_argument_lists, dataset._label_list["test"],
                                                         dataset._file_list["test"]):
            out_file.write("-------" + "\n")
            out_file.write(file_name + "\n")
            out_file.write("true label:" + str(arguments) + "\n")
            out_file.write("true label rank:" + str(ss.rankdata(arguments, method="dense")) + "\n")
            out_file.write("predicted label:" + str(predicted_label) + "\n")
            out_file.write("rounded label:" + str(tf.math.round(predicted_label)) + "\n")
            out_file.write(
                "predicted label rank:" + str(ss.rankdata(tf.math.round(predicted_label), method="dense")) + "\n")
            mse = tf.keras.losses.MSE(arguments, predicted_label)
            out_file.write("mse:" + str(mse) + "\n")
            mse_list.append(mse)

        out_file.write("-------" + "\n")
        out_file.write("mean(mse_list):" + str(np.mean(mse_list)) + "\n")
        plt.xlabel('graph number')
        plt.ylabel('mse of predicted label')
        plt.plot(mse_list, label="predicted_data_mse")
        plt.plot([mean_loss_list_average] * len(mse_list), label="mean_mse")
        plt.legend()
        plt.savefig("trained_model/" + log_name+ "-test-mse.png")
        plt.clf()


class raw_graph_inputs():
    def __init__(self,num_edge_types,total_number_of_nodes):
        self._num_edge_types=num_edge_types
        self._total_number_of_nodes=total_number_of_nodes
        self._node_number_per_edge_type=[]
        self.final_graphs=None
        self.argument_scores=[]
        self.labels=[]
        self.ranked_argument_scores=[]
        self.file_names=[]
        self.argument_identify=[]
        self.control_location_identify=[]
        self.label_size=0
        self.vocabulary_set=set()
        self.token_map={}
        self.class_weight={}

class load_data_graph_inputs():
    def __init__(self,num_edge_types,total_number_of_nodes):
        self._num_edge_types=num_edge_types
        self._total_number_of_nodes=total_number_of_nodes
        self._node_number_per_edge_type=[]
        self.final_graphs=None
        self.argument_scores=[]
        self.labels=[]
        self.ranked_argument_scores=[]
        self.file_names=[]
        self.argument_identify=[]
        self.control_location_identify=[]
        self.label_size=0
        self.vocabulary_set=set()
        self.token_map={}
        self.class_weight={}

class parsed_graph_data:
    def __init__(self,graphs_node_label_ids,graphs_argument_indices,graphs_adjacency_lists,
                 graphs_argument_scores,total_number_of_node,graph_control_location_indices,file_name_list,skipped_file_list,parsed_arguments,
                 graphs_node_symbols,graphs_label_indices,graphs_learning_labels,vocabulary_set, token_map):
        self.graphs_node_label_ids=graphs_node_label_ids
        self.graphs_argument_indices=graphs_argument_indices
        self.graphs_adjacency_lists=graphs_adjacency_lists
        self.graphs_argument_scores=graphs_argument_scores
        self.total_number_of_node=total_number_of_node
        self.graphs_control_location_indices=graph_control_location_indices
        self.file_name_list=file_name_list
        self.parsed_arguments=parsed_arguments
        self.graphs_node_symbols=graphs_node_symbols
        self.vocabulary_set=vocabulary_set
        self.token_map=token_map
        self.graphs_label_indices=graphs_label_indices
        self.graphs_learning_labels=graphs_learning_labels
        self.skipped_file_list=skipped_file_list

def write_graph_to_pickle(benchmark,  data_fold=["train", "valid", "test"], label="rank",path="/Users/sherry/Downloads/Systematic-Predicate-Abstraction-using-Machine-Learning-master/Heuristic_selection/",
                          file_type=".smt2",max_nodes_per_batch=10000,
                          graph_type="hyperEdgeHornGraph",vocabulary_name="",file_list=[]):
    json_type="."+graph_type+".JSON"
    if len(data_fold)==1:
        voc=pickleRead(vocabulary_name + "-" + label + "-vocabulary" ,"/Users/sherry/Downloads/Systematic-Predicate-Abstraction-using-Machine-Learning-master/Heuristic_selection/src/trained_model/")
        vocabulary_set, token_map = voc[0],voc[1]
    else:
        vocabulary_set, token_map = build_vocabulary(datafold=["train", "valid", "test"], path=path,json_type=json_type)
        pickleWrite([vocabulary_set, token_map], benchmark + "-" + label + "-vocabulary" , "/Users/sherry/Downloads/Systematic-Predicate-Abstraction-using-Machine-Learning-master/Heuristic_selection/src/trained_model/")
    benchmark_name = benchmark.replace("/", "-")
    print("debugging write graph")
    for df in data_fold:
        print("write data_fold to pickle data:", df)
        graphs_node_label_ids = [] 
        graphs_node_symbols = []
        graphs_argument_indices = []
        graphs_adjacency_lists = []
        graphs_argument_scores = []
        parsed_arguments = []
        graphs_control_location_indices = []
        graphs_label_indices = []
        graphs_learning_labels = []
        total_number_of_node = 0
        file_type = file_type
        file_name_list = []
        skipped_file_list=[]

        # for fileGraph, fileArgument in zip(sorted(glob.glob(path +df+"_data/"+ '*' + file_type + json_type)),
        #                                    sorted(glob.glob(path +df+"_data/"+ '*' + file_type + '.arguments'))):
        files_from_benchmark=set(sorted(glob.glob(path +df+"_data/"+ '*' + file_type + json_type)))
        file_set=(lambda : [f+json_type for f in file_list] if len(file_list)>0 else files_from_benchmark)()
        #print("file set debug", file_set)
        # print("file from benchmark debug", files_from_benchmark)
        # print("path is:", path)
        # print("file type debug", file_type)
        # print("json type debug", json_type)
        #fileSet=set(sorted(glob.glob(path +df+"_data/"+ '*' + file_type + json_type))) #all .smt2.hyperedge.JSON file
        for fileGraph in file_set:
            fileName = fileGraph[:fileGraph.find(file_type + json_type) + len(file_type)]
            fileName = fileName[fileName.rindex("/") + 1:]
            # read graph
            print("read graph from",fileGraph)
            print("file_name",fileName)
            print("max node",max_nodes_per_batch )
            with open(fileGraph) as f:
                loaded_graph = json.load(f)
                #debug check all field if equal to empty
                #print("Debuuuuug",loaded_graph["nodeIds"])
                if len(loaded_graph["nodeIds"]) == 0:
                    print("nodeIds==0"," skip ",fileName)
                    skipped_file_list.append(fileName)
                    # for f in glob.glob(path+df+"_data/"+fileName + "*"):
                    #     shutil.copy(f, "../benchmarks/problem_cases/")
                    #     os.remove(f)
                elif len(loaded_graph["nodeIds"]) >= max_nodes_per_batch: #
                    #print("more than " + str(max_nodes_per_batch) + " nodes","skip",fileName)
                    skipped_file_list.append(fileName)
                # if len(loaded_graph["argumentEdges"]) == 0:
                #     print("argumentEdges==0",fileName)
                # if len(loaded_graph["guardASTEdges"]) == 0:
                #     print("guardASTEdges==0",fileName)
                # if len(loaded_graph["dataFlowASTEdges"]) == 0:
                #     print("dataFlowASTEdges==0",fileName)
                # if len(loaded_graph["binaryAdjacentList"]) == 0:
                #     print("binaryAdjacentList==0",fileName)
                # if len(loaded_graph["ternaryAdjacencyList"]) == 0:
                #     print("ternaryAdjacencyList==0",fileName)
                # if len(loaded_graph["controlFlowHyperEdges"]) == 0:
                #     print("controlFlowHyperEdges==0",fileName)
                # if len(loaded_graph["dataFlowHyperEdges"]) == 0:
                #     print("dataFlowHyperEdges==0",fileName)
                else:
                    file_name_list.append(fileGraph[:fileGraph.find(json_type)])
                    graphs_node_label_ids.append(loaded_graph["nodeIds"])
                    graphs_node_symbols.append(loaded_graph["nodeSymbolList"])
                    #read label
                    if label=="predicate_occurrence_in_clauses":
                        graphs_label_indices.append(loaded_graph["predicateIndices"])
                        graphs_learning_labels.append(loaded_graph["predicateOccurrenceInClause"])
                    elif label=="predicate_occurrence_in_SCG":
                        graphs_label_indices.append(loaded_graph["predicateIndices"])
                        graphs_learning_labels.append(loaded_graph["predicateStrongConnectedComponent"])
                    elif label=="argument_bound" or label=="argument_lower_bound_existence" or label=="argument_upper_bound_existence" or label=="argument_lower_bound" or label=="argument_upper_bound":
                        graphs_argument_indices.append(loaded_graph["argumentIndices"])
                        graphs_learning_labels.append(loaded_graph["argumentBoundList"])
                    elif label=="control_location_identify":
                        graphs_control_location_indices.append(loaded_graph["controlLocationIndices"])
                    elif label=="argument_occurrence_binary":
                        graphs_argument_indices.append(loaded_graph["argumentIndices"])
                        graphs_learning_labels.append(loaded_graph["argumentBinaryOccurrenceList"])
                    elif label=="template_relevance":
                        graphs_label_indices.append(loaded_graph["templateIndices"])
                        graphs_learning_labels.append(loaded_graph["templateRelevanceLabel"])
                    elif label=="clause_occurrence_in_counter_examples_binary":
                        graphs_label_indices.append(loaded_graph["clauseIndices"])
                        graphs_learning_labels.append(loaded_graph["clauseBinaryOccurrenceInCounterExampleList"])

                    else:
                        graphs_argument_indices.append(loaded_graph["argumentIndices"])
                        # read argument from JSON file
                        parsed_arguments = parseArgumentsFromJson(loaded_graph["argumentIDList"],
                                                                  loaded_graph["argumentNameList"],
                                                                  loaded_graph["argumentOccurrence"])
                        graphs_argument_scores.append([int(argument.score) for argument in parsed_arguments])
                        graphs_control_location_indices.append(loaded_graph["controlLocationIndices"])

                    if json_type==".hyperEdgeHornGraph.JSON" or json_type==".equivalent-hyperedgeGraph.JSON" or json_type==".concretized-hyperedgeGraph.JSON": #read adjacency_lists
                        #for hyperedge horn graph
                        #print("debug loaded hyperedge")
                        graphs_adjacency_lists.append([
                            np.array(loaded_graph["argumentEdges"]),
                            np.array(loaded_graph["guardASTEdges"]),
                            np.array(loaded_graph["dataFlowASTEdges"]),
                            np.array(loaded_graph["binaryAdjacentList"]),
                            np.array(loaded_graph["controlFlowHyperEdges"]),
                            np.array(loaded_graph["dataFlowHyperEdges"]),
                            np.array(loaded_graph["ternaryAdjacencyList"])
                        ])
                    else:
                        #print("debug loaded",loaded_graph["predicateArgumentEdges"])
                        #for layer horn graph
                        graphs_adjacency_lists.append([
                            np.array(loaded_graph["predicateArgumentEdges"]),
                            np.array(loaded_graph["predicateInstanceEdges"]),
                            np.array(loaded_graph["argumentInstanceEdges"]),
                            np.array(loaded_graph["controlHeadEdges"]),
                            np.array(loaded_graph["controlBodyEdges"]),
                            np.array(loaded_graph["controlEdges"]),
                            np.array(loaded_graph["controlArgumentEdges"]),
                            np.array(loaded_graph["subTermEdges"]),
                            np.array(loaded_graph["guardEdges"]),
                            np.array(loaded_graph["dataEdges"]),
                            np.array(loaded_graph["predicateInstanceHeadEdges"]),
                            np.array(loaded_graph["predicateInstanceBodyEdges"]),
                            np.array(loaded_graph["controlArgumentHeadEdges"]),
                            np.array(loaded_graph["controlArgumentBodyEdges"]),
                            np.array(loaded_graph["guardConstantEdges"]),
                            np.array(loaded_graph["guardOperatorEdges"]),
                            np.array(loaded_graph["guardScEdges"]),
                            np.array(loaded_graph["dataConstantEdges"]),
                            np.array(loaded_graph["dataOperatorEdges"]),
                            np.array(loaded_graph["dataScEdges"]),
                            np.array(loaded_graph["subTermConstantOperatorEdges"]),
                            np.array(loaded_graph["subTermOperatorOperatorEdges"]),
                            np.array(loaded_graph["subTermScOperatorEdges"]),
                            np.array(loaded_graph["binaryAdjacentList"]),
                            #np.array(loaded_graph["unknownEdges"])
                        ])
                        #print("graphs_adjacency_lists debug:", graphs_adjacency_lists)
                    total_number_of_node += len(loaded_graph["nodeIds"])


        pickle_data=parsed_graph_data(graphs_node_label_ids,graphs_argument_indices,graphs_adjacency_lists,
                                      graphs_argument_scores,total_number_of_node,graphs_control_location_indices,file_name_list,skipped_file_list,
                                      parsed_arguments,graphs_node_symbols,graphs_label_indices,graphs_learning_labels,vocabulary_set, token_map)
        #print("pickledata debug:",pickle_data.graphs_adjacency_lists)                       
        pickleWrite(pickle_data, "train-" +label+"-"+ graph_type +"-"+benchmark_name + "-gnnInput_" + df + "_data")


def form_GNN_inputs_and_labels(label="occurrence", datafold=["train", "valid", "test"], benchmark="",graph_type="hyperEdgeHornGraph",gathered_nodes_binary_classification_task="gathered_nodes_binary_classification_task"):
    print("form labels")
    benchmark_name = benchmark.replace("/", "-")
    for df in datafold:
        parsed_dot_file = pickleRead("train-" +label+"-"+ graph_type +"-"+benchmark_name + "-gnnInput_" + df + "_data")
        #print("train-" +label+"-"+ graph_type +"-"+benchmark_name + "-gnnInput_" + df + "_data")
        #print(parsed_dot_file.graphs_adjacency_lists)
        if label in gathered_nodes_binary_classification_task or label=="predicate_occurrence_in_clauses" or label=="argument_lower_bound" or label=="argument_upper_bound":
            form_predicate_occurrence_related_label_graph_sample(parsed_dot_file.graphs_node_label_ids,
                                                                    parsed_dot_file.graphs_node_symbols,
                                                                    parsed_dot_file.graphs_adjacency_lists,
                                                                    parsed_dot_file.total_number_of_node,
                                                                    parsed_dot_file.vocabulary_set,
                                                                    parsed_dot_file.token_map,
                                                                    parsed_dot_file.file_name_list,
                                                                    parsed_dot_file.skipped_file_list,benchmark, df,
                                                                    parsed_dot_file.graphs_argument_indices,
                                                                    parsed_dot_file.graphs_label_indices,
                                                                    parsed_dot_file.graphs_learning_labels,label,graph_type,gathered_nodes_binary_classification_task)

        else:
            form_horn_graph_samples(parsed_dot_file.graphs_node_label_ids, parsed_dot_file.graphs_node_symbols,
                                    parsed_dot_file.graphs_argument_indices, parsed_dot_file.graphs_adjacency_lists,
                                    parsed_dot_file.graphs_argument_scores, parsed_dot_file.total_number_of_node,
                                    parsed_dot_file.graphs_control_location_indices, parsed_dot_file.file_name_list,parsed_dot_file.skipped_file_list,
                                    parsed_dot_file.graphs_label_indices, parsed_dot_file.graphs_learning_labels,
                                    label, parsed_dot_file.vocabulary_set, parsed_dot_file.token_map, benchmark, df,graph_type)
    print("debug voc set", parsed_dot_file.vocabulary_set)
def get_batch_graph_sample_info(graphs_adjacency_lists,total_number_of_node,vocabulary_set,token_map):
    num_edge_types_list=[]
    for graph_edges in graphs_adjacency_lists:
        number_of_edge_type = 0
        for edge in graph_edges:
            if len(edge)!=0:
                number_of_edge_type+=1
        num_edge_types_list.append(number_of_edge_type)
    raw_data_graph = raw_graph_inputs(max(num_edge_types_list),
                                      total_number_of_node)  # graphs_adjacency_lists[0] means the first graph's adjacency_list
    temp_graph_index = 0
    for i, graphs_adjacency in enumerate(graphs_adjacency_lists):
        temp_count = 0
        for edge_type in graphs_adjacency:
            if len(edge_type) != 0:
                temp_count = temp_count + 1
        if temp_count == raw_data_graph._num_edge_types:#len(graphs_adjacency):
            temp_graph_index = i

    for edge_type in graphs_adjacency_lists[temp_graph_index]:
        if len(edge_type)!=0:
            raw_data_graph._node_number_per_edge_type.append(len(edge_type[0]))

    raw_data_graph.vocabulary_set = vocabulary_set
    raw_data_graph.token_map = token_map
    return raw_data_graph

def form_predicate_occurrence_related_label_graph_sample(graphs_node_label_ids,graphs_node_symbols,
                                                            graphs_adjacency_lists,total_number_of_node,
                                                            vocabulary_set,token_map,file_name_list,skipped_file_list,benchmark,df,
                                                            graphs_argument_indices,graphs_label_indices,graphs_learning_labels,label,graph_type,gathered_nodes_binary_classification_task,pickle=True):
    final_graphs=[]
    raw_data_graph=get_batch_graph_sample_info(graphs_adjacency_lists,total_number_of_node,vocabulary_set,token_map)
    if label=="predicate_occurrence_in_SCG" or label=="predicate_occurrence_in_clauses" or label=="template_relevance" or label=="clause_occurrence_in_counter_examples_binary":
        graphs_node_indices=graphs_label_indices
    elif label=="argument_occurrence_binary":
        graphs_node_indices = graphs_argument_indices
    elif label=="argument_bound":
        graphs_node_indices = graphs_argument_indices
        for one_graph_learning_labels in graphs_learning_labels: #transform "None" to infinity
            for learning_labels in one_graph_learning_labels:
                if isinstance(learning_labels[0],str):
                    learning_labels[0]=(float("-inf"))
                elif isinstance(learning_labels[1],str):
                    learning_labels[1] = (float("inf"))
    elif label=="argument_lower_bound_existence":
        graphs_node_indices = graphs_argument_indices
        graphs_learning_labels_temp=[]
        for one_graph_learning_labels in graphs_learning_labels:
            temp_graph_label=[]
            for learning_labels in one_graph_learning_labels:
                if isinstance(learning_labels[0],str):
                    learning_labels[0]=0
                else:
                    learning_labels[0] = 1
                temp_graph_label.append(learning_labels[0])
            graphs_learning_labels_temp.append(temp_graph_label)
        graphs_learning_labels=graphs_learning_labels_temp
    elif label=="argument_upper_bound_existence":
        graphs_node_indices = graphs_argument_indices
        graphs_learning_labels_temp = []
        for one_graph_learning_labels in graphs_learning_labels:
            temp_graph_label=[]
            for learning_labels in one_graph_learning_labels:
                if isinstance(learning_labels[1],str):
                    learning_labels[1]=0
                else:
                    learning_labels[1] = 1
                temp_graph_label.append(learning_labels[1])
            graphs_learning_labels_temp.append(temp_graph_label)
        graphs_learning_labels = graphs_learning_labels_temp
    elif label=="argument_lower_bound":
        (graphs_node_indices, graphs_learning_labels, graphs_node_label_ids,
        graphs_node_symbols,graphs_adjacency_lists, file_name_list) = form_argument_bound_label(graphs_argument_indices,
                                                                           graphs_learning_labels,
                                                                           graphs_node_label_ids,
                                                                           graphs_node_symbols,
                                                                           graphs_adjacency_lists, file_name_list,label)

    elif label=="argument_upper_bound":
        (graphs_node_indices, graphs_learning_labels, graphs_node_label_ids,
        graphs_node_symbols, graphs_adjacency_lists, file_name_list) = form_argument_bound_label(graphs_argument_indices,
                                                                                                graphs_learning_labels,
                                                                                                graphs_node_label_ids,
                                                                                                graphs_node_symbols,
                                                                                                graphs_adjacency_lists,
                                                                                                file_name_list, label)
    if label in gathered_nodes_binary_classification_task:
        drawBinaryLabelPieChart(graphs_learning_labels, label, graph_type, benchmark,df)
    all_one_label=0
    one_one_label=0
    other_distribution=0
    total_files=len(graphs_node_label_ids)
    for node_label_ids, node_symbols, adjacency_lists,file_name,node_indices,learning_labels in zip(graphs_node_label_ids,graphs_node_symbols,
                                                                                                         graphs_adjacency_lists,
                                                                                                         file_name_list,
                                                                                                         graphs_node_indices,
                                                                                                         graphs_learning_labels):
        raw_data_graph.file_names.append(file_name)
        # node tokenization
        print("debug token_map", token_map)
        #token map contains the same thing as token
        print("debug node_symbols", node_symbols)
        tokenized_node_label_ids=tokenize_symbols(token_map,node_symbols)
        print("debug tokenized_node_label_ids", tokenized_node_label_ids)
        raw_data_graph.labels.append(learning_labels)

        #catch label distribution
        if len(learning_labels) == sum(learning_labels):
            all_one_label = all_one_label + 1
        elif sum(learning_labels) == 1:
            one_one_label = one_one_label + 1
        else:
            other_distribution = other_distribution + 1
        print("debuggung tokenized node",tokenized_node_label_ids)
        print("debugging learning label",learning_labels )
        print("debugging node_indices",node_indices )
        # if(len(tokenized_node_label_ids)>130000):
        #     print("------debug------")
        #     print("file_name",file_name)
        #     print("number of node", len(tokenized_node_label_ids))
        #     print("number of edges per edge type")
        #     for edge_type in adjacency_lists:
        #         print(len(edge_type),end=" ")
        #     print("\n node_indices ", len(node_indices))
        #     print("learning_labels", len(learning_labels))

        # temp_count=0
        # for edge_type in adjacency_lists:
        #     if len(edge_type)==0 :#and len(tokenized_node_label_ids)<50
        #         temp_count+=1
        #
        # if temp_count==2:
        #     print("------debug------")
        #     print("file_name", file_name)
        #     print("number of node", len(tokenized_node_label_ids))
        #     print("number of edges per edge type")
        #     for edge_type in adjacency_lists:
        #         print(len(edge_type), end=" ")
        #     print("\n node_indices ", len(node_indices))
        #     print("learning_labels", len(learning_labels))

        final_graphs.append(
            HornGraphSample(
                adjacency_lists=adjacency_lists,
                node_features=np.array(tokenized_node_label_ids),
                node_indices=np.array(node_indices),
                node_label=np.array(learning_labels)
            )
        )
        raw_data_graph.label_size += len(learning_labels)
    raw_data_graph.final_graphs = final_graphs.copy()

    #print label distribution
    print("-----------label distribution --------- datafold: ",df)
    all_label=[item for sublist in graphs_learning_labels for item in sublist]
    print("total files", str(total_files)+"/"+str(len(skipped_file_list)+total_files))
    print("total label size", raw_data_graph.label_size)
    print("positive label",sum(all_label)/len(all_label))
    print("negative label",1-(sum(all_label)/len(all_label)))
    print("all_one_label", all_one_label, "percentage", all_one_label / total_files)
    print("one_one_label", one_one_label, "percentage", one_one_label / total_files)
    print("other_distribution", other_distribution, "percentage", other_distribution / total_files)
    weight_for_0, weight_for_1 = 1,1
    if df=="train":
        weight_for_0,weight_for_1=get_class_weight(sum(all_label),len(all_label)-sum(all_label),len(all_label))
    print("weight_for_0",weight_for_0)
    print("weight_for_1",weight_for_1)
    raw_data_graph.class_weight[df]={"weight_for_0":weight_for_0,"weight_for_1":weight_for_1}

    if pickle==True:
        pickleWrite(raw_data_graph, label +"-"+graph_type+ "-" + benchmark + "-gnnInput_" + df + "_data")
    return raw_data_graph


def form_argument_bound_label(graphs_argument_indices, graphs_learning_labels, graphs_node_label_ids, graphs_node_symbols,
            graphs_adjacency_lists, file_name_list,label):
    bound_index=0
    if label=="argument_lower_bound":
        bound_index=0
    elif label=="argument_upper_bound":
        bound_index=1
    graphs_node_indices_temp = []
    graphs_learning_labels_temp = []
    node_label_ids_temp = []
    node_symbols_temp = []
    adjacency_lists_temp = []
    file_name_list_temp = []
    for one_graph_indices, one_graph_learning_labels, one_graphs_node_label_ids, one_graphs_node_symbols, one_graphs_adjacency_lists, one_file_name_list in zip(
            graphs_argument_indices, graphs_learning_labels, graphs_node_label_ids, graphs_node_symbols,
            graphs_adjacency_lists, file_name_list):
        temp_indces = []
        temp_labels = []
        for index, learning_label in zip(one_graph_indices, one_graph_learning_labels):
            if not isinstance(learning_label[bound_index], str):
                temp_indces.append(index)
                temp_labels.append(learning_label[bound_index])

        if len(temp_labels) != 0:  # delete this graph
            graphs_node_indices_temp.append(temp_indces)
            graphs_learning_labels_temp.append(temp_labels)
            node_label_ids_temp.append(one_graphs_node_label_ids)
            node_symbols_temp.append(one_graphs_node_symbols)
            adjacency_lists_temp.append(one_graphs_adjacency_lists)
            file_name_list_temp.append(one_file_name_list)
            print("one_graph_learning_labels", one_graph_learning_labels)
            # print("temp_labels", temp_labels)
            # print("temp_indces", temp_indces)
    # if len(graphs_learning_labels_temp)==0:
    return graphs_node_indices_temp,graphs_learning_labels_temp,node_label_ids_temp,node_symbols_temp,adjacency_lists_temp,file_name_list_temp


def form_horn_graph_samples(graphs_node_label_ids,graphs_node_symbols, graphs_argument_indices, graphs_adjacency_lists,
                            graphs_argument_scores, total_number_of_node,graphs_control_location_indices, file_name_list,skipped_file_list,
                            graphs_label_indices,graphs_learning_labels,
                            label,vocabulary_set,token_map,benchmark, df,graph_type,pickle=True,):
    final_graphs_v1 = []
    raw_data_graph = get_batch_graph_sample_info(graphs_adjacency_lists,total_number_of_node,vocabulary_set,token_map)

    total_label=0
    total_nodes=0

    if len(graphs_control_location_indices)==0:
        graphs_control_location_indices=graphs_argument_indices
    # directory_wrong_extracted_cases=file_name_list[0][:file_name_list[0].rfind("/")+1]+"wrong_extracted_cases"
    # if not os.path.exists(directory_wrong_extracted_cases):
    #     os.makedirs(directory_wrong_extracted_cases)
    for node_label_ids, node_symbols, argument_indices, adjacency_lists, argument_scores,control_location_indices,\
        file_name in zip(graphs_node_label_ids,graphs_node_symbols,graphs_argument_indices,graphs_adjacency_lists,
                         graphs_argument_scores,graphs_control_location_indices,file_name_list):
        # convert to rank
        ranked_argument_scores = ss.rankdata(argument_scores, method="dense")
        argument_identify = np.array([0] * len(node_label_ids))
        argument_identify[argument_indices] = 1
        total_nodes+=len(node_label_ids)
        control_location_identify = np.array([0] * len(node_label_ids))
        control_location_identify[control_location_indices]=1

        #If .argument file has different number of argument with JSON file. copy that file to wrong_extracted_cases. and do not append this file
        if len(argument_indices)!=len(argument_scores):
            print("------------------argument_scores != argument_indices-------------------------")
            print("argument_scores", len(argument_scores))
            print("argument_indices", len(argument_indices))
            print(file_name)
            # shutil.copy(file_name, directory_wrong_extracted_cases)
            # from Miscellaneous import remove_list_of_file
            # remove_list_of_file(file_name)
        else:
            raw_data_graph.argument_identify.append(argument_identify)
            raw_data_graph.control_location_identify.append(control_location_identify)
            raw_data_graph.ranked_argument_scores.append(ranked_argument_scores)
            raw_data_graph.argument_scores.append(argument_scores)
            raw_data_graph.file_names.append(file_name)

            #node tokenization
            tokenized_node_label_ids=tokenize_symbols(token_map,node_symbols)

            if label == "rank":
                raw_data_graph.labels.append(argument_scores)
                total_label += len(ranked_argument_scores)
                final_graphs_v1.append(
                    HornGraphSample(
                        adjacency_lists=adjacency_lists,
                        node_features=np.array(tokenized_node_label_ids),
                        node_label=np.array(ranked_argument_scores),
                        node_indices=np.array(argument_indices),
                    )
                )
                raw_data_graph.label_size+=len(ranked_argument_scores)
            elif label == "occurrence":
                raw_data_graph.labels.append(argument_scores)
                total_label += len(argument_scores)
                final_graphs_v1.append(
                    HornGraphSample(
                        adjacency_lists=adjacency_lists,
                        node_features=np.array(tokenized_node_label_ids),
                        node_label=np.array(argument_scores),  # argument_scores
                        node_indices=np.array(argument_indices),
                    )
                )
                raw_data_graph.label_size += len(argument_scores)
            elif label == "argument_identify":
                raw_data_graph.labels.append(argument_identify)
                total_label += len(argument_identify)
                final_graphs_v1.append(
                    HornGraphSample(
                        adjacency_lists=adjacency_lists,
                        node_features=np.array(tokenized_node_label_ids),
                        #node_features=tf.constant(node_label_ids),
                        # node_label=tf.constant(ranked_argument_scores),
                        node_indices=np.array(argument_indices),
                        node_label=np.array(argument_identify)
                        #current_node_index=tf.constant([]),
                        #node_control_location=tf.constant(control_location_indices)
                    )
                )
                raw_data_graph.label_size += len(argument_identify)
            elif label == "control_location_identify":
                raw_data_graph.labels.append(control_location_identify)
                total_label += len(control_location_identify)
                final_graphs_v1.append(
                    HornGraphSample(
                        adjacency_lists=adjacency_lists,
                        node_features=np.array(tokenized_node_label_ids),
                        node_label=np.array(control_location_identify),
                        node_indices=np.array(argument_indices),
                    )
                )
                raw_data_graph.label_size += len(control_location_identify)
            elif label == "argument_identify_no_batchs":
                total_label += len([1])
                raw_data_graph.label_size += len(node_label_ids)
                for i in node_label_ids:
                    if i in argument_indices:
                        final_graphs_v1.append(
                            HornGraphSample(
                                adjacency_lists=adjacency_lists,
                                node_features=np.array(tokenized_node_label_ids),
                                node_label=np.array([1]),
                                node_argument=np.array(argument_indices),
                                current_node_index=np.array([i]),
                                node_control_location=np.array(control_location_indices)
                            )
                        )

                    else:
                        total_label += len([0])
                        final_graphs_v1.append(
                            HornGraphSample(
                                adjacency_lists=tuple(adjacency_lists),
                                node_features=tf.constant(tokenized_node_label_ids),
                                node_label=tf.constant([0]),
                                node_argument=tf.constant(argument_indices),
                                current_node_index=tf.constant([i]),
                                node_control_location=tf.constant(control_location_indices)
                            )
                        )
            else:
                pass
    raw_data_graph.final_graphs = final_graphs_v1.copy()
    if pickle == True:
        pickleWrite(raw_data_graph, label +"-" +graph_type+"-" + benchmark + "-gnnInput_" + df + "_data")
    print("total_label",total_label)
    print("total_nodes",total_nodes)
    return raw_data_graph



def get_predicted_label_list_divided_by_file(dataset,predicted_Y_loaded_model):
    label_number_lists = []
    for labels in dataset._label_list["test"]:
        label_number_lists.append(len(labels))
    predicted_label_lists = []
    for i, n in enumerate(label_number_lists):
        predicted_label_lists.append(
            predicted_Y_loaded_model[sum(label_number_lists[:i]):sum(label_number_lists[:i]) + n])
    return predicted_label_lists

def build_vocabulary(datafold=["train", "valid", "test"], path="",json_type=".layerHornGraph.JSON"):
    vocabulary_set=set(["unknown_node","unknown_predicate","unkown_symblic_constant","unkown_predicate_argument",
                        "unknown_operator","unknown_constant","unknown_predicate_label"])
    print("inside the vocab set is :" , vocabulary_set)
    for fold in datafold:
        for json_file in glob.glob(path+fold+"_data/*"+json_type):
            with open(json_file) as f:
                loaded_graph = json.load(f)
                vocabulary_set.update(loaded_graph["nodeSymbolList"])
    token_map={}
    token_id=0
    vocabulary_set=set([convert_constant_to_category(w) for w in vocabulary_set])
    for word in sorted(vocabulary_set):
        token_map[word]=token_id
        token_id=token_id+1
    print("vocabulary_set",vocabulary_set)
    print("insidetoken is", token_map)
    return vocabulary_set,token_map

def tokenize_symbols(token_map,node_symbols):
    converted_node_symbols=[ convert_constant_to_category(word) for word in node_symbols]
    # node tokenization
    full_operator_list = ["+", "-", "*", "/", ">", ">=", "=", "<", "<=", "==", "===", "!", "+++", "++", "**", "***",
                          "--", "---", "=/=","&","|","EX","and","or"]
    tokenized_node_label_ids = []
    for symbol in converted_node_symbols:
        if symbol in token_map:
            tokenized_node_label_ids.append(token_map[symbol])
        elif "CONTROL" in symbol:
            #print("unknown_predicate_CONTROL", symbol)
            tokenized_node_label_ids.append(token_map["unknown_predicate"])
        elif "predicateArgument" in symbol:
            #print("unkown_predicateArgument", symbol)
            tokenized_node_label_ids.append(token_map["unkown_predicate_argument"])
        elif "template" in symbol:
            #print("unknown_predicate_label", symbol)
            tokenized_node_label_ids.append(token_map["unknown_predicate_label"])
        elif "SYMBOLIC_CONSTANT" in symbol:
            #print("unkown_symblic_constant", symbol)
            tokenized_node_label_ids.append(token_map["unkown_symblic_constant"])
        elif symbol.isnumeric() or symbol[1:].isnumeric():
            #print("unknown_constant", symbol)
            tokenized_node_label_ids.append(token_map["unknown_constant"])
        elif symbol in full_operator_list:
            #print("unknown_operator",symbol)
            tokenized_node_label_ids.append(token_map["unknown_operator"])
        else:
            tokenized_node_label_ids.append(token_map["unknown_node"])
    return tokenized_node_label_ids

def convert_constant_to_category(constant_string):
    converted_string=constant_string
    if constant_string.isdigit() and int(constant_string)>1:
        converted_string="positive_constant"
    elif converted_string[1:].isdigit() and int(constant_string)<-1:
        converted_string="negative_constant"
    return converted_string

def count_paramsters(model):
    from keras import backend as K
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))


def get_class_weight(pos, neg, total):
    weight_for_0 = (1 / neg) * (total) / 2
    weight_for_1 = (1 / pos) * (total) / 2
    return weight_for_0,weight_for_1

def get_test_loss_with_class_weight(class_weight,task_output,labels,from_logits=True):
    #predicted_y = (lambda: task_output if from_logits == False else [logit(x) for x in task_output])()
    #predicted_y=task_output
    predicted_y=[logit(x) for x in task_output]
    # description: implemented by exaggerating inputs
    # weighted_prediction = [y_hat * class_weight["weight_for_1"] if y == 1 else y_hat for y, y_hat in zip(labels, predicted_y)]
    # return tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels,weighted_prediction,from_logits=from_logits))
    # description: implemented by weighted_cross_entropy_with_logits
    #print("class_weight",class_weight["weight_for_1"],class_weight["weight_for_0"])
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels,predicted_y,class_weight["weight_for_1"]))
    # description: implemented by conditions
    # ce = []
    # for y, y_hat in zip(labels, predicted_y):
    #     if y == 1:
    #         ce.append(tf.keras.losses.binary_crossentropy([y], [y_hat], from_logits=from_logits) * class_weight[
    #             "weight_for_1"])
    #     elif y == 0:
    #         ce.append(tf.keras.losses.binary_crossentropy([y], [y_hat], from_logits=from_logits))
    # return tf.reduce_mean(ce)