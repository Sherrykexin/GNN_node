import numpy as np
import pickle
import os,glob,shutil
from distutils.dir_util import copy_tree
import json
import matplotlib.pyplot as plt
from numba import cuda
import tensorflow as tf

def GPU_switch(GPU):
    if GPU==False:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.keras.backend.clear_session()
    else:
        #watch nvidia-smi
        cuda.select_device(0)
        cuda.close()
        print('CUDA memory released: GPU0')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

def drawBinaryLabelPieChart(learning_label,label,graph_type,benchmark_name,df):
    flat_list = [item for sublist in learning_label for item in sublist]
    positive_label_number =sum(flat_list)
    negative_label_number=len(flat_list)-positive_label_number
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = '1', "0"
    sizes = [positive_label_number, negative_label_number]
    explode = (0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Label distribution")
    plt.savefig("/Users/sherry/Downloads/Systematic-Predicate-Abstraction-using-Machine-Learning-master/Heuristic_selection/src/trained_model/" + label + "-" + graph_type +"-" + df + "-" + benchmark_name + "pie_chart.png")
    plt.clf()

def add_JSON_field(fileName="",file_type=".layerHornGraph.JSON",old_field=[],new_field=[],new_field_content=[]):
    json_file = fileName + file_type
    json_obj = {}
    with open(json_file) as f:
        loaded_graph = json.load(f)
        for field in old_field:
            json_obj[field] = loaded_graph[field]
    # add more field
    with open(json_file) as f:
        loaded_graph = json.load(f)
        for field,content in zip(new_field,new_field_content):
            json_obj[field] = content
    # write json object to JSON file
    clear_file(json_file)
    with open(json_file, 'w') as f:
        json.dump(json_obj, f)

def copy_and_remove(src,dst):
    if os.path.isdir(src):
        hints_dir=dst+src[src.rfind("/"):]
        os.mkdir(hints_dir)
        copy_tree(src,hints_dir)
        shutil.rmtree(src)
    else:
        shutil.copy(src, dst)
        os.remove(src)

def remove_list_of_file(name):
    if os.path.exists(name):
        os.remove(name)
    if os.path.exists(name+".initialHints"):
        os.remove(name+".initialHints")
    if os.path.exists(name+".negativeHints"):
        os.remove(name+".negativeHints")
    if os.path.exists(name+".positiveHints"):
        os.remove(name+".positiveHints")
    if os.path.exists(name+"-auto.gv"):
        os.remove(name+"-auto.gv")
    if os.path.exists(name+".JSON"):
        os.remove(name+".JSON")
    if os.path.exists(name+".HornGraph"):
        os.remove(name+".HornGraph")
    if os.path.exists(name+".horn"):
        os.remove(name+".horn")
    if os.path.exists(name+".arguments"):
        os.remove(name+".arguments")
    if os.path.exists(name+".gv"):
        os.remove(name+".gv")


def clear_directory(name):
    if os.path.exists(name):
        shutil.rmtree(name)
        os.makedirs(name)
    else:
        os.makedirs(name)
def clear_file(name):
    if os.path.exists(name):
        os.remove(name)
        with open(name, 'w') as fp:
            pass
    else:
        with open(name, 'w') as fp:
            pass


def copy_train_data_from_src_to_dst(src,dst):
    arguments_file_list=glob.glob(src+"*.arguments")
    initial_hints_file_list=glob.glob(src+"*.initialHints")
    negative_hints_file_list = glob.glob(src + "*.negativeHints")
    positive_hints_file_list = glob.glob(src + "*.positiveHints")
    json_file_list = glob.glob(src + "*.JSON")
    original_file_list = glob.glob(src + "*.smt2")
    auto_gv_file_list = glob.glob(src + "*auto.gv")
    for arguments_file,initial_hints_file,negative_hints_file,positive_hints_file,json_file,original_file,auto_gv_file in zip(arguments_file_list,initial_hints_file_list,negative_hints_file_list,positive_hints_file_list,json_file_list,original_file_list,auto_gv_file_list):
        shutil.copy(arguments_file, dst)
        shutil.copy(initial_hints_file, dst)
        shutil.copy(negative_hints_file, dst)
        shutil.copy(positive_hints_file, dst)
        shutil.copy(json_file, dst)
        shutil.copy(original_file, dst)
        shutil.copy(auto_gv_file, dst)

def renameBenchmarkFiles():
    counter=0
    for txt_file in glob.iglob('/home/chencheng/Downloads/sv-benchmarks-master/c/*/*.c'):
        print(txt_file)
        if os.path.exists(txt_file + ".annot.c"):
            print("")
        else:
            shutil.copy2(txt_file, txt_file + ".annot.c")
            counter=counter+1
    print("Program counter:"+str(counter))

def checkSplitData(X_train, X_test, y_train, y_test):
    print("------train-----")
    print("X_train",len(X_train))
    print("y_train", len(y_train))
    for i,j in zip(X_train,y_train):
        print(i[1])
        print(j)
    print("-----test-----")
    print("X_train", len(X_test))
    print("y_train", len(y_test))
    for i,j in zip(X_test,y_test):
        print(i[1])
        print(j)



def recoverPredictedText(predictedX,predictedY):
    recoverdX=list()

    programList=list()
    for p in range(len(predictedX)):
        programList.append(predictedX[p][0])
    progranList=list(set(programList))  #delete duplication in program list
    for i in range(len(progranList)):
        program=progranList[i]
        hints=list()
        for j in range(len(predictedX)):
            if (progranList[i] == predictedX[j][0] and predictedY[j]==1):
                hints.append(predictedX[j][1])
        hints=list(set(hints))
        recoverdX.append([[program],hints])
    return recoverdX

def printOnePredictedTextInStringForm(recoverdX,index,printProgram=False):

        #print program in string form
        if(printProgram == True):
            print('program in horn format:')
            strProgram=''.join(str(p) for p in recoverdX[index][0])
            print(strProgram)

        strHints=''.join(str(h)+str('\n') for h in recoverdX[index][1])
        #print(strProgram)
        #print(strHints)
        hintsDictionary = dict()
        for head in recoverdX[index][1]:
            if (head.find('main') != -1):
                head = head.strip()
                hintsDictionary[head[:head.find('\n')]]=list()
        for head in hintsDictionary.keys():
            for content in recoverdX[index][1]:
                if (content.find(head) != -1):
                    hintsDictionary[head].append(content[content.find('\n'):].strip())
        #print(hintsDictionary)

        # print hints in string form
        for head in hintsDictionary.keys():
            print(head)
            print(''.join(str(h)+str('\n') for h in hintsDictionary[head]))




def testAccuracy(predictedY,trueY):
    counter=0
    for i in range(len(predictedY)):
        if(predictedY[i]==trueY[i]):
            counter=counter+1
    acc=counter/len(predictedY)
    print("test accuracy:", acc)
    return acc

def pickleWrite(content,name,path='/Users/sherry/Downloads/Systematic-Predicate-Abstraction-using-Machine-Learning-master/Heuristic_selection/pickleData/'):
    #parenDir = os.path.abspath(os.path.pardir)
    file=path+name+".txt"
    print('pickle write to '+file)
    with open(file,"wb") as fp :
        pickle.dump(content,fp)

def pickleRead(name,path='/Users/sherry/Downloads/Systematic-Predicate-Abstraction-using-Machine-Learning-master/Heuristic_selection/pickleData/'):
    #parenDir = os.path.abspath(os.path.pardir)
    file=path + name +".txt"
    print('pickle read '+file)
    with open(file,"rb") as fp :
        content=pickle.load(fp)
    return content
    
def printList(l):
    for i in l:
        print(i)

def sortHints(unsortedList):
    def sortSecond(val):
        return val[1]
    unsortedList.sort(key=sortSecond,reverse=True)
    #printList(unsortedList)
    return unsortedList

def rank_arguments_naive(arr):
    array = np.array(arr)
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks

def replicate_files(directory,times):
    for file in glob.glob(directory+"*"):
        for i in range(1, times):
            fileName=file[file.rfind("/")+1:]
            shutil.copy(file,directory+str(i)+"-"+fileName)








