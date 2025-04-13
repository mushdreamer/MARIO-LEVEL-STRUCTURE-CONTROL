import os
import sys
import csv
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print(os.getcwd())
sys.path.append(os.getcwd())
from util import SearchHelper
from util import bc_calculate
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
#print('AAAAAH', str(pathlib.Path().absolute()))
os.environ['CLASSPATH']=os.path.join(str(pathlib.Path().absolute()),"Mario.jar")
#os.environ['CLASSPATH'] = "/home/tehqin/Projects/MarioGAN-LSI/Mario.jar"

from util.SearchHelper import detect_structure_failure
from util.SearchHelper import compute_structure_score


import pandas as pd
import numpy as np
from numpy.linalg import eig
import torch
#import torchvision.utils as vutils
from torch.autograd import Variable

import toml
import json
import numpy
import util.models.dcgan as dcgan
import torch
#import torchvision.utils as vutils
from torch.autograd import Variable
import json
import numpy
import util.models.dcgan as dcgan
import math
import random
from collections import OrderedDict
import csv
from algorithms import *
from util.SearchHelper import *

from jnius import autoclass
MarioGame = autoclass('engine.core.MarioGame')
Agent = autoclass('agents.robinBaumgarten.Agent')

"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c','--config', help='path of experiment config file',required=True)
opt = parser.parse_args()
"""

batch_size =1
nz = 32
record_frequency=20

if not os.path.exists("logs"):
    os.mkdir("logs")

global EliteMapConfig
EliteMapConfig=[]

import sys

def eval_mario(ind,visualize):
    is_pass = False
    realLevel=to_level(ind.level)
    JString = autoclass('java.lang.String')
    agent = Agent()
    game = MarioGame()
    result = game.runGame(agent, JString(realLevel), 20, 0, visualize)
    #print(result)
    messageReceived=str(result.getCompletionPercentage())+","
    messageReceived+=str(result.getNumJumps())+","
    messageReceived+=str(result.getKillsTotal())+","
    messageReceived+=str(result.getCurrentLives())+","
    messageReceived+=str(result.getNumCollectedTileCoins())+","
    messageReceived+=str(result.getRemainingTime())
    
    
    #messageReceived=sys.stdin.readline()
    statsList=messageReceived.split(',')
    ind.statsList=statsList
    ind.features=[]
    for bc in EliteMapConfig["Map"]["Features"]:
        get_feature=bc["name"]
        get_feature=getattr(bc_calculate,get_feature)
        feature_value=get_feature(ind,result)
        ind.features.append(feature_value)
    ind.features=tuple(ind.features)

    #新增：定义通关成功与否
    completion_percentage=float(statsList[0]) * 100
    print(f"Completion Percentage: {completion_percentage:.2f}%")

    is_pass = abs(completion_percentage - 100.0) < 1.0
    structure_score = compute_structure_score(ind.level)
    fitness = completion_percentage + 10 * is_pass + 5 * structure_score

    #新增：如果通关成功，则给予额外奖励
    completion_percentage = 10 * is_pass
    #########################################

    #新增：打印每轮是否通关
    if is_pass:
        print("PASS")
    else:
        print("FAIL TO PASS")

    # 统一结构失败检测（静态 + 行为）
    valid, failure_reason = detect_structure_failure(ind.level, statsList, is_pass)
    if not valid:
        print(f"Warning: Skipping structurally invalid level due to: {failure_reason}")
        ind.failure_type = failure_reason
        ind.blocked = True  #新增：设置阻止加入地图
        penalty_map = {
            "START_NO_GROUND": -10,
            "GAP_TOO_WIDE": -10,
            "WALL_TOO_HIGH": -10,
        }
        fitness += penalty_map.get(failure_reason, -2)  #结构失败惩罚信号
        ind.statsList = ['0'] * 6
        ind.features = [0.0] * len(EliteMapConfig["Map"]["Features"])
        return fitness, is_pass
    #########################

    return fitness, is_pass

evaluate = eval_mario



def run_trial(num_to_evaluate,algorithm_name,algorithm_config,elite_map_config,trial_name,model_path,visualize):
    feature_ranges=[]
    column_names=['emitterName','latentVector', 'completionPercentage','jumpActionsPerformed','killsTotal','livesLeft','coinsCollected','remainingTime (20-timeSpent)']
    bc_names=[]
    #新增：通关统计
    pass_count = 0
    ##############
    #新增：100次通关统计
    pass_rate_history = []
    ######################
    # 新增：结构错误统计器
    failure_counter = {
        "START_NO_GROUND": 0,
        "GAP_TOO_WIDE": 0,
        "WALL_TOO_HIGH": 0
        }
    failure_history = []
    for bc in elite_map_config["Map"]["Features"]:
        feature_ranges.append((bc["low"],bc["high"]))
        column_names.append(bc["name"])
        bc_names.append(bc["name"])

    if(trial_name.split('_')[1]=="8Binary"):
        feature_map = FeatureMap(num_to_evaluate, feature_ranges,resolutions=(2,)*8)
    elif(trial_name.split('_')[1]=="MarioGANBC"):
        feature_map = FeatureMap(num_to_evaluate, feature_ranges, resolutions=(151,26))
    elif(trial_name.split('_')[1]=="KLBC"):
        feature_map = FeatureMap(num_to_evaluate, feature_ranges, resolutions=(60,60))
    else:
        sys.exit('unknown BC name. Exiting the program.')

    if algorithm_name=="CMAES":
        print("Start Running CMAES")
        mutation_power=algorithm_config["mutation_power"]
        population_size=algorithm_config["population_size"]
        algorithm_instance=CMA_ES_Algorithm(num_to_evaluate,mutation_power,population_size,feature_map,trial_name,column_names,bc_names)
    elif algorithm_name=="CMAME":
        print("Start Running CMAME")
        mutation_power=algorithm_config["mutation_power"]
        population_size=algorithm_config["population_size"]
        initial_population = algorithm_config["initial_population"] 
        emitter_type = algorithm_config["emitter_type"] 
        algorithm_instance=CMA_ME_Algorithm(mutation_power,initial_population, num_to_evaluate,population_size,feature_map,trial_name,column_names,bc_names, emitter_type) 
    elif algorithm_name=="MAPELITES":
        print("Start Running MAPELITES")
        mutation_power=algorithm_config["mutation_power"]
        initial_population=algorithm_config["initial_population"]
        algorithm_instance=MapElitesAlgorithm(mutation_power, initial_population, num_to_evaluate, feature_map,trial_name,column_names,bc_names)
    elif algorithm_name=="ISOLINEDD":
        print("Start Running MAP-Elites with ISOLINEDD")
        mutation_power1=algorithm_config["mutation_power1"]
        mutation_power2=algorithm_config["mutation_power2"]
        initial_population=algorithm_config["initial_population"]
        algorithm_instance=MapElitesLineAlgorithm(mutation_power1, mutation_power2,initial_population, num_to_evaluate, feature_map,trial_name,column_names,bc_names)
    elif algorithm_name=="RANDOM":
        print("Start Running RANDOM")
        algorithm_instance=RandomGenerator(num_to_evaluate,feature_map,trial_name,column_names,bc_names)
    
    #新增：通关统计
    simulation=1
    while algorithm_instance.is_running():
        ind = algorithm_instance.generate_individual()
        ind.level=gan_generate(ind.param_vector,batch_size,nz,model_path)
        #新增：通关奖励，将fitness赋值给ind.fitness，is_pass赋值给is_pass#
        ind.fitness, is_pass = evaluate(ind, visualize)

        #新增：通过统计
        if is_pass:
            pass_count += 1

        # 统计结构错误
        if hasattr(ind, "failure_type") and ind.failure_type in failure_counter:
            failure_counter[ind.failure_type] += 1

        # 每 100 次记录一次
        if simulation % 100 == 0:
            pass_rate = pass_count / simulation * 100
            pass_rate_history.append((simulation, pass_rate))
            print(f"[Info] At simulation {simulation}: Pass rate = {pass_rate:.2f}%")

            failure_history.append([
                simulation,
                failure_counter["START_NO_GROUND"],
                failure_counter["GAP_TOO_WIDE"],
                failure_counter["WALL_TOO_HIGH"]
            ])
            print(f"[Failure Count] @ {simulation}: {failure_counter}")

            for k in failure_counter:
                failure_counter[k] = 0

        algorithm_instance.return_evaluated_individual(ind)
        print(f"{simulation}/{num_to_evaluate} simulations finished")
        simulation += 1

    algorithm_instance.all_records.to_csv("logs/"+trial_name+"_all_simulations.csv")

    #新增：保存百次成功率并制图
    csv_path = f"logs/{trial_name}_pass_rate.csv"
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Simulation", "Pass Rate (%)"])
        writer.writerows(pass_rate_history)

    # 保存结构失败 CSV（不嵌套绘图！）
    failure_csv_path = f"logs/{trial_name}_failure_stats.csv"
    with open(failure_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Simulation", "START_NO_GROUND", "GAP_TOO_WIDE", "WALL_TOO_HIGH"])
        writer.writerows(failure_history)

    #成功绘图
    try:
        df = pd.DataFrame(pass_rate_history, columns=["Simulation", "Pass Rate (%)"])
        plt.figure(figsize=(10,6))
        plt.plot(df["Simulation"], df["Pass Rate (%)"], marker='o', color='royalblue')
        plt.xlabel("Simulation")
        plt.ylabel("Pass Rate (%)")
        plt.title(f"Pass Rate Over Time\n{trial_name}")
        plt.grid(True)
        plt.tight_layout()
        plot_path = f"logs/{trial_name}_pass_rate_plot.png"
        plt.savefig(plot_path)
        print(f"Pass rate plot saved to: {plot_path}")
    except Exception as e:
        print("Failed to generate pass rate plot:", e)

    #结构错误趋势图
    try:
        df_fail = pd.DataFrame(failure_history, columns=["Simulation", "START_NO_GROUND", "GAP_TOO_WIDE", "WALL_TOO_HIGH"])
        plt.figure(figsize=(10,6))
        plt.plot(df_fail["Simulation"], df_fail["START_NO_GROUND"], marker="o", label="START_NO_GROUND")
        plt.plot(df_fail["Simulation"], df_fail["GAP_TOO_WIDE"], marker="s", label="GAP_TOO_WIDE")
        plt.plot(df_fail["Simulation"], df_fail["WALL_TOO_HIGH"], marker="^", label="WALL_TOO_HIGH")
        plt.xlabel("Simulation")
        plt.ylabel("Failure Count (per 100)")
        plt.title(f"Structure Failure Trends\n{trial_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        failure_plot_path = f"logs/{trial_name}_failure_trend.png"
        plt.savefig(failure_plot_path)
        print(f"Failure trend plot saved to: {failure_plot_path}")
    except Exception as e:
        print("Failed to generate failure trend plot:", e)

    plt.show()

    print(f"\nSummary: {pass_count} passed / {num_to_evaluate} total ({(pass_count / num_to_evaluate) * 100:.2f}%)")
    ################################################################################################################

"""
if __name__ == '__main__':
    print("READY") # Java loops until it sees this special signal
    #sys.stdout.flush() # Make sure Java can sense this output before Python blocks waiting for input 
    experiment_toml=toml.load(opt.config)
    num_trials=experiment_toml["num_trials"]
    trial_toml=toml.load(experiment_toml["trial_config"])
    for t in range (num_trials):
        NumSimulations=trial_toml["num_simulations"]
        AlgorithmToRun=trial_toml["algorithm"]
        AlgorithmConfig=toml.load(trial_toml["algorithm_config"])
        EliteMapConfig=toml.load(trial_toml["elite_map_config"])
        TrialName=trial_toml["trial_name"]+str(t+1)
        run_trial(NumSimulations,AlgorithmToRun,AlgorithmConfig,EliteMapConfig,TrialName)
        #below needs to be changed
        print("Finished One Trial")
    print("Finished All Trials")
"""

def start_search(sim_number,trial_index,experiment_toml,model_path,visualize):
    experiment_toml=experiment_toml["Trials"][trial_index]
    trial_toml=toml.load(experiment_toml["trial_config"])
    NumSimulations=trial_toml["num_simulations"]
    AlgorithmToRun=trial_toml["algorithm"]
    AlgorithmConfig=toml.load(trial_toml["algorithm_config"])
    global EliteMapConfig
    EliteMapConfig=toml.load(trial_toml["elite_map_config"])
    TrialName=trial_toml["trial_name"]+"_sim"+str(sim_number)
    run_trial(NumSimulations,AlgorithmToRun,AlgorithmConfig,EliteMapConfig,TrialName,model_path,visualize)
    print("Finished One Trial")
	


