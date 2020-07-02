# augment instance set
import sys
import time
import random
import json
import subprocess
import math
import os
from copy import deepcopy
import psutil
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

sys.path.append(sys.path[0] + '/../../')
from path_converter import path_con
from instance_set.diverse_mutator.VRPSPDTW_solution_checker import read_VRPSPDTW_problem
MAX_PAR = 40

class vrp(object):
    def __init__(self, pro, num_cus, capacity, Dispatching):
        self.pro = pro
        self.num_cus = num_cus
        self.capacity = capacity
        self.Dispatching = Dispatching
    def __eq__(self, other):
        return self.pro.shape == other.pro.shape and\
               np.all(self.pro == other.pro) and\
               self.num_cus == other.num_cus and\
               self.capacity == other.capacity and\
               self.Dispatching == other.Dispatching

def mutation(inst, domain):
    mutation_rate = 0.9
    if domain == 'TSP':
        x_list = [cor_tuple[0] for cor_tuple in inst]
        max_x = max(x_list)
        min_x = min(x_list)
        y_list = [cor_tuple[1] for cor_tuple in inst]
        max_y = max(y_list)
        min_y = min(y_list)
        new_inst = []
        for cor_tuple in inst:
            if random.random() < mutation_rate:
                sigma = 0.025
                success = False
                while not success:
                    delta_1 = np.random.normal(0, sigma*(max_x-min_x), 1)[0]
                    delta_2 = np.random.normal(0, sigma*(max_y-min_y), 1)[0]
                    xcity = cor_tuple[0] + int(delta_1)
                    ycity = cor_tuple[1] + int(delta_2)
                    xcity = min(max(0, xcity), max_x)
                    ycity = min(max(0, ycity), max_y)
                    # if xcity < 0 or xcity > max_x or ycity < 0 or ycity > max_y:
                    #     xcity, ycity = cor_tuple # set to parent
                    if (xcity, ycity) not in new_inst:
                        new_inst.append((xcity, ycity))
                        success = True
            else:
                success = False
                while not success:
                    xcity, ycity = random.randint(min_x, max_x), random.randint(min_y, max_y)
                    if (xcity, ycity) not in new_inst:
                        new_inst.append((xcity, ycity))
                        success = True
    if domain == 'VRPSPDTW':
        # for each customer, offset coordinate, time windows and pdemand or ddemand
        max_x = np.max(inst.pro[:, 1])
        min_x = np.min(inst.pro[:, 1])
        max_y = np.max(inst.pro[:, 2])
        min_y = np.min(inst.pro[:, 2])
        max_ddemand = np.max(inst.pro[:, 3])
        min_ddemand = np.min(inst.pro[:, 3])
        max_pdemand = np.max(inst.pro[:, 4])
        min_pdemand = np.min(inst.pro[:, 4])
        max_demand = np.max(inst.pro[:, 3] + inst.pro[:, 4])
        max_tw = inst.pro[0, 6]
        min_tw = inst.pro[0, 5]
        min_tw_w = np.min(inst.pro[:, 6] - inst.pro[:, 5])

        new_pro = np.zeros(inst.pro.shape, dtype=int)
        new_pro[0] = inst.pro[0]
        for i in range(1, inst.pro.shape[0]):
            node = inst.pro[i]
            if random.random() < mutation_rate:
                sigma = 0.025
                success = False
                while not success:
                    delta_1 = np.random.normal(0, sigma*(max_x-min_x), 1)[0]
                    delta_2 = np.random.normal(0, sigma*(max_y-min_y), 1)[0]
                    x = node[1] + int(delta_1)
                    y = node[2] + int(delta_2)
                    x = min(max(0, x), max_x)
                    y = min(max(0, y), max_y)
                    # if x < 0 or x > max_x or y < 0 or y > max_y:
                    #     x, y = node[1], node[2] # set to parent
                    if np.unique(np.vstack((new_pro[:i, 1:3], [x, y])), axis=0).shape[0] == i+1:
                        new_pro[i, 0:3] = [i, x, y]
                        success = True
            else:
                success = False
                while not success:
                    x, y = random.randint(min_x, max_x), random.randint(min_y, max_y)
                    if np.unique(np.vstack((new_pro[:i, 1:3], [x, y])), axis=0).shape[0] == i+1:
                        new_pro[i, 0:3] = [i, x, y]
                        success = True
            # shift pdemand or ddemand
            sigma = 0.025
            delta_1 = np.random.normal(0, sigma*(max_ddemand-min_ddemand), 1)[0]
            delta_2 = np.random.normal(0, sigma*(max_pdemand-min_pdemand), 1)[0]
            ddemand = node[3] + int(delta_1)
            pdemand = node[4] + int(delta_2)
            ddemand = min(max(min_ddemand, ddemand), max_ddemand)
            pdemand = min(max(min_pdemand, pdemand), max_pdemand)
            total = ddemand + pdemand
            # total cound not be larger than max_demand
            if total > max_demand:
                min_val = min(ddemand, pdemand)
                first = min(int((total-max_demand) / 2), min_val)
                if ddemand <= pdemand:
                    ddemand -= first
                    pdemand -= (total - max_demand - first)
                else:
                    pdemand -= first
                    ddemand -= (total - max_demand - first)
            if ddemand == 0 and pdemand == 0:
                ddemand, pdemand = node[3], node[4]
            new_pro[i, 3:5] = [ddemand, pdemand]
            new_pro[i, 7] = 10 if pdemand == 0 or ddemand == 0 else 20
            #shift time windows
            delta_1, delta_2 = np.random.normal(0, sigma*(max_tw-min_tw), 2)
            tw_1 = node[5] + int(delta_1)
            tw_2 = node[6] + int(delta_2)
            tw_1 = min(max(min_tw, tw_1), max_tw)
            tw_2 = min(max(min_tw, tw_2), max_tw)
            tw_1, tw_2 = min(tw_1, tw_2), max(tw_1, tw_2)
            # dist(0, node) < tw_2
            dist = np.sqrt((new_pro[i, 1] - new_pro[0, 1]) ** 2 +\
                           (new_pro[i, 2] - new_pro[0, 2]) ** 2)
            tw_2 = max(dist + random.randint(0, 5), tw_2)
            if tw_2 > max_tw:
                print 'tw_2 %d > max_tw %d\n' % (tw_2, max_tw)
                print 'delta_1/2 %d %d, tw_1/2 %d %d\n' % (delta_1, delta_2, tw_1, tw_2)
                print 'current node %s\n' % new_pro[i]
                sys.exit(-1)
            if tw_2 - tw_1 < min_tw_w:
                tw_1 = max(min_tw, tw_2 - min_tw_w)
            new_pro[i, 5:7] = [tw_1, tw_2]
            success = True
        new_inst = vrp(new_pro, inst.num_cus, inst.capacity, inst.Dispatching)
    return new_inst

def generate_new(insts, domain, new_size=MAX_PAR):
    new_insts = []
    for _ in range(new_size):
        new_insts.append(mutation(random.choice(insts), domain))
    return new_insts

def distance(f_m):
    ins_num = f_m.shape[0]
    distance_matrix = np.zeros((ins_num, ins_num))
    for xx in range(ins_num):
        for yy in range(xx+1, ins_num):
            distance_matrix[xx, yy] = math.sqrt((f_m[xx, 0] - f_m[yy, 0]) ** 2 +\
                                               (f_m[xx, 1] - f_m[yy, 1]) ** 2)
            distance_matrix[yy, xx] = distance_matrix[xx, yy]
    max_distance = np.max(distance_matrix)
    for xx in range(ins_num):
        distance_matrix[xx, xx] = max_distance
    return distance_matrix


def worse_than_ins(ins_fitness, ins_feature, fitness, f_m, all_deleted_f_m):
    worse_indices = []
    tmp_f_m = np.vstack((f_m, all_deleted_f_m, ins_feature))
    d_m = distance(tmp_f_m)

    for index in range(f_m.shape[0]):
        if fitness[index] > ins_fitness:
            continue
        if np.min(np.delete(d_m, -1, axis=0)[:, index]) >\
            np.min(np.delete(d_m, index, axis=0)[:, -1]):
            continue
        worse_indices.append(index)
    return worse_indices

def compute_feature(inst_names, domain):
    feature_matrix = np.zeros((0, 0))
    if domain == "TSP":
        os.chdir(path_con('instance_set/diverse_mutator/'))
        feature_compute = "Rscript ./compute_tsp_features.R "
        feature_num = 7
        feature_matrix = np.zeros((len(inst_names), feature_num))

        # compute features
        running = 0
        solve_process = set()
        for i, ins in enumerate(inst_names):
            while True:
                if running >= MAX_PAR:
                    time.sleep(0.1)
                    finished_process = [
                        pid for pid in solve_process if pid.poll() is not None
                    ]
                    solve_process -= set(finished_process)
                    running = len(solve_process)
                    continue
                else:
                    result_file = "feature_%d" % (i+1)
                    cmd = feature_compute + ins + " > " + result_file
                    solve_process.add(psutil.Popen(cmd, shell=True))
                    running = len(solve_process)
                    break
        while solve_process:
            time.sleep(5)
            print 'Still %d feature computing process not exits' % len(solve_process)
            finished = [pid for pid in solve_process if pid.poll() is not None]
            solve_process -= set(finished)

        for i, _ in enumerate(inst_names):
            result_file = "feature_%d" % (i+1)
            with open(result_file, 'r') as f:
                lines = f.read().strip().split('\n')
                for k, line in enumerate(lines):
                    feature_matrix[i, k] = float(line.split()[1])

        cmd = 'rm feature_*'
        pid = subprocess.Popen(cmd, shell=True)
        pid.communicate()

        # # do scaling and pca
        # with open('../diverse_TSP/indices/tsp_scaler', 'r') as f:
        #     scaler = pickle.load(f)
        # with open('../diverse_TSP/indices/tsp_pca', 'r') as f:
        #     pca = pickle.load(f)
        # feature_matrix = pca.transform(scaler.transform(feature_matrix))

        os.chdir(path_con('src/GenAS/'))

    return feature_matrix


def represent_insts(inst_names, domain):
    insts = []
    if domain == 'TSP':
        for ins_name in inst_names:
            insCor = []
            with open(ins_name.replace('"', ''), 'r') as f:
                lines = f.read().strip().split('\n')
                i = 0
                for line in lines:
                    if "NODE_COORD_SECTION" in line:
                        break
                    i += 1
                for line in lines[i + 1:]:
                    if "EOF" in line:
                        break
                    if len(line) < 2:
                        continue
                    _, x, y = line.strip().split()
                    insCor.append((int(float(x)), int(float(y))))
            insts.append(insCor)
    if domain == 'VRPSPDTW':
        for ins_name in inst_names:
            pro, num_cus, capacity, Dispatching = read_VRPSPDTW_problem(ins_name)
            insts.append(vrp(pro, num_cus, capacity, Dispatching))
    return insts

def construct_ins_file(insts, domain, folder='tmp'):
    # first clear files in folder
    if domain == 'TSP':
        cmd = 'rm ' + path_con('AC_output/GenAS/%s/*.tsp' % folder)
    if domain == 'VRPSPDTW':
        cmd = 'rm ' + path_con('AC_output/GenAS/%s/*.VRPSPDTW' % folder)
    pid = subprocess.Popen(cmd, shell=True)
    pid.communicate()

    inst_names = []
    if domain == 'TSP':
        for i, ins in enumerate(insts):
            insName = path_con('AC_output/GenAS/%s/%d.tsp' % (folder, (i+1)))
            with open(insName, 'w+') as f:
                f.write('NAME : newins-%d\n' % (i+1))
                f.write('COMMENT : NONE\n')
                f.write('TYPE : TSP\n')
                f.write('DIMENSION : %d\n' % len(ins))
                f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
                f.write('NODE_COORD_SECTION\n')
                for k, city in enumerate(ins):
                    f.write('%d %d %d\n' % (k+1, city[0], city[1]))
            inst_names.append(insName)
    if domain == 'VRPSPDTW':
        for i, ins in enumerate(insts):
            insName = path_con('AC_output/GenAS/%s/%d.VRPSPDTW' % (folder, (i+1)))
            with open(insName, 'w+') as f:
                f.write('%d_%d\n\n' % (ins.num_cus, i))
                f.write('CUSTOMER  VEHICLE\n')
                f.write('NUMBER    NUMBER     CAPACITY    DispatchingCost\n')
                f.write('%7.d%8.d%15.d%10.d\n\n' % (ins.num_cus, 500, ins.capacity, ins.Dispatching))
                f.write('CUSTOMER\n')
                f.write('CUST NO.  XCOORD.   YCOORD.   DDEMAND   PDEMAND   '
                        'READY TIME  DUE TIME   SERVICE TIME\n\n')
                for j in range(ins.pro.shape[0]):
                    node_data = ins.pro[j, :]
                    f.write('%5.1d%8.1d%11.1d%11.1d%11.1d%11.1d%11.1d%11.1d\n' %\
                    (node_data[0], node_data[1], node_data[2],\
                     node_data[3], node_data[4], node_data[5],\
                     node_data[6], node_data[7]))
            inst_names.append(insName)
    return inst_names


def solve_to_optimum(inst_names, domain):
    optimum = dict()
    if domain == 'TSP':
        concorde = path_con('Solver/Concorde/concorde')
        os.chdir(path_con('Solver/Concorde'))

        # solve tsp instances
        running = 0
        solve_process = set()
        for i, ins in enumerate(inst_names):
            while True:
                if running >= MAX_PAR:
                    time.sleep(0.1)
                    finished_process = [
                        pid for pid in solve_process if pid.poll() is not None
                    ]
                    solve_process -= set(finished_process)
                    running = len(solve_process)
                    continue
                else:
                    cmd = '%s %s > ./qua_n%d' % (concorde, ins, i+1)
                    solve_process.add(psutil.Popen(cmd, shell=True))
                    running = len(solve_process)
                    break
        while solve_process:
            time.sleep(5)
            print 'Still %d solving process not exits' % len(solve_process)
            finished = [pid for pid in solve_process if pid.poll() is not None]
            solve_process -= set(finished)

        # read quality files and clear configlogs
        cmd = 'rm *.mas *.pul *.sav *.sol *.res'
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()
        # read quality file
        for i, ins in enumerate(inst_names):
            with open('./qua_n%d' % (i+1), 'r') as f:
                lines = f.readlines()
            for line in lines:
                if 'Optimal Solution' in line:
                    solution = line[line.find(':') + 1:].strip()
                    optimum[ins.replace('"', '')] = solution
                    break
        cmd = 'rm qua_n*'
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()

        os.chdir(path_con('src/GenAS'))

    return optimum

def test_new(insts, metric, solution_checker, portfolio, minTestTimes, cutoffTime, domain,
             diverse, inst_names=None, solution_file=None, complete=False):
    # test insts complete or not
    # and compute features of insts if diverse=True
    r1 = None
    r2 = None
    alg_num = len(portfolio)

    if inst_names is None:
        inst_names = construct_ins_file(insts, domain)

    if solution_file is None and domain == 'TSP':
        optimum = solve_to_optimum(inst_names, domain)
        solution_file = path_con('AC_output/GenAS/tmp/tmp_optimum.json')
        with open(solution_file, 'w+') as f:
            json.dump(optimum, f)

    fullSolver = list()
    for i in range(alg_num):
        fullSolver.append(portfolio[i].replace('-@1', '-@%d' % (i+1)))
    fullSolver = ' '.join(fullSolver)

    running_tasks = 0
    sub_process = set()
    outDir = path_con('AC_output/GenAS/tmp/')
    for i, ins in enumerate(inst_names):
        for j in range(minTestTimes):
            while True:
                if running_tasks * alg_num >= MAX_PAR:
                    time.sleep(0.1)
                    finished = [
                        pid for pid in sub_process if pid.poll() is not None]
                    sub_process -= set(finished)
                    running_tasks = len(sub_process)
                    continue
                else:
                    seed = random.randint(0, 1000000)
                    output_file = '%sIns%d_Seed%d' % (outDir, i, j)
                    cmd = 'python ' + path_con('src/util/testing_wrapper.py ')
                    if solution_file:
                        cmd += '--opt-fn %s ' % solution_file
                    if complete or metric == 'quality':
                        cmd += '--full-performance '
                    if solution_checker:
                        cmd += '--solution-checker %s ' % solution_checker
                    cmd += '%s %s %d %d %d %s' %\
                            (ins, output_file, cutoffTime, 0, seed, fullSolver)
                    sub_process.add(psutil.Popen(cmd, shell=True))
                    running_tasks = len(sub_process)
                    break

    # check if subprocess all exits
    while sub_process:
        time.sleep(5)
        print 'Still %d testing-instance process not exits' % len(sub_process)
        finished = [pid for pid in sub_process if pid.poll() is not None]
        sub_process -= set(finished)

    # extract testing results
    if metric == 'runtime':
        punish = 10
    elif metric == 'quality':
        punish = 100000
    # performance matrix, [i,j] i+1 run j ins
    newFitness = np.zeros((len(inst_names),)) * np.nan
    new_p_m = np.zeros((alg_num, len(inst_names))) * np.nan
    runCount = np.zeros(newFitness.shape) * np.nan
    runCount_d = np.zeros(new_p_m.shape) * np.nan
    crashed_indice = set()
    for i, _ in enumerate(inst_names):
        for j in range(minTestTimes):
            output_file = '%sIns%d_Seed%d' % (outDir, i, j)
            with open(output_file, 'r') as f:
                lines = f.read().strip().split('\n')
            outputLine = lines[0]
            values = outputLine[outputLine.find(\
                ':') + 1:].strip().replace(' ', '').split(',')
            (status, runtime, quality) = (values[0], float(values[1]), float(values[3]))
            if metric == 'runtime' and 'TIMEOUT' in status:
                runtime = runtime * punish
            if metric == 'quality' and 'TIMEOUT' in status:
                quality = punish
                crashed_indice.add(i)
            if np.isnan(newFitness[i]):
                if metric == 'runtime':
                    newFitness[i] = runtime
                elif metric == 'quality':
                    newFitness[i] = quality
                runCount[i] = 1
            else:
                if metric == 'runtime':
                    newFitness[i] += runtime
                elif metric == 'quality':
                    newFitness[i] += quality
                runCount[i] += 1

            for line in lines[1:alg_num+1]:
                detailedR = line.split(',')
                thread_index = int(detailedR[0])
                status = detailedR[1]
                runtime = float(detailedR[2])
                quality = float(detailedR[3])

                if metric == 'runtime' and 'TIMEOUT' in status:
                    runtime = runtime * punish
                if metric == 'quality' and 'TIMEOUT' in status:
                    quality = punish

                if np.isnan(new_p_m[thread_index-1, i]):
                    if metric == 'runtime':
                        new_p_m[thread_index-1, i] = runtime
                    elif metric == 'quality':
                        new_p_m[thread_index - 1, i] = quality
                    runCount_d[thread_index-1, i] = 1
                else:
                    if metric == 'runtime':
                        new_p_m[thread_index-1, i] += runtime
                    elif metric == 'quality':
                        new_p_m[thread_index - 1, i] += quality
                    runCount_d[thread_index-1, i] += 1

    if domain == 'VRPSPDTW' and metric == 'quality' and not complete:
        # in this case, filter those crashed instances
        crashed_indice = list(crashed_indice)
        print 'Crashed indice %s, filter them \n' % (crashed_indice)
        newFitness = np.delete(newFitness, crashed_indice)
        runCount = np.delete(runCount, crashed_indice)
        new_p_m = np.delete(new_p_m, crashed_indice, axis=1)
        runCount_d = np.delete(runCount_d, crashed_indice, axis=1)
        crashed_indice = sorted(crashed_indice, reverse=True)
        for indice in crashed_indice:
            del insts[indice]
            del inst_names[indice]

    newFitness = np.true_divide(newFitness, runCount)
    new_p_m = np.true_divide(new_p_m, runCount_d)
    # clear dir
    cmd = 'rm %sIns*' % outDir
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()

    if complete:
        r1 = new_p_m
    else:
        r1 = newFitness
    if diverse:
        r2 = compute_feature(inst_names, domain)
    return r1, r2

def insGeneration(portfolio, instanceIndexFile, per_matrix, solutionFile,
                  metric, solution_checker, generation_time, minTestTimes,
                  maxIt, cutoffTime, domain, diverse, logFile):
    # first we create a copy of current instances
    with open(instanceIndexFile, 'r') as f:
        initial_inst_names = f.read().strip().split('\n')
    insts = represent_insts(initial_inst_names, domain)

    initial_insts = deepcopy(insts)

    logFile.write('-----------------Instance Generation-------------\n')
    # obtain fitness, the bigger, the better
    fitness = np.min(per_matrix, axis=0)
    logFile.write('Initial mean fitness: %f\n' % np.mean(fitness))
    # obtain feature matrix if necessary
    f_m = None
    if diverse:
        f_m = compute_feature(initial_inst_names, domain)
        scaler = MinMaxScaler()
        f_m = scaler.fit_transform(f_m)
        # call PCA to determine the transform matrix
        pca = PCA(n_components=2)
        f_m = pca.fit_transform(f_m)

        all_deleted = set()
        all_deleted_f_m = np.zeros((0, 2))

    start = time.time()
    ite = 1
    while ite <= maxIt and time.time() - start < generation_time:
        logFile.write('iteration %d\n' % ite)
        logFile.write('mean fitness: %f\n' % np.mean(fitness))
        new_insts = generate_new(insts, domain)
        len_new = len(new_insts)
        logFile.write('Generated %d new instances\n' % len_new)
        new_fitness, new_feature = test_new(new_insts, metric, solution_checker,
                                            portfolio, minTestTimes, cutoffTime,
                                            domain, diverse)
        len_new = len(new_insts)
        logFile.write('After testing, we have %d instances\n' % len_new)
        if len_new == 0:
            ite += 1
            continue
        # remove
        if not diverse:
            # if not considering diverse
            logFile.write('Not considering diverse, only on fitness\n')
            tmp = np.concatenate([fitness, new_fitness])
            logFile.write('Fitness: %s\nNew_fitness: %s\nMerge: %s\n' %\
                          (str(fitness), str(new_fitness), str(tmp)))
            sort_index = np.argsort(tmp)
            logFile.write('sorting results: %s\n' % str(sort_index))
            delete_index = np.sort(sort_index[0:len_new])
            delete_index = delete_index[::-1]
            logFile.write('delete index: %s\n' % str(delete_index))
            # rearrange insts and fitness
            insts.extend(new_insts)
            for index in delete_index:
                del insts[index]
            fitness = np.delete(tmp, delete_index)
        else:
            # consider fitness and diverse
            logFile.write('Considering diversity and fitness\n')
            new_feature = pca.transform(scaler.transform(new_feature))
            for i, ins in enumerate(new_insts):
                # find all instanes in insts that are worse than ins
                worse_indices = worse_than_ins(new_fitness[i], new_feature[i],
                                               fitness, f_m, all_deleted_f_m)
                if worse_indices:
                    logFile.write('Examing #%d ins\n' % i)
                    logFile.write('Worse indices in current set: %s\n' %\
                                  str(worse_indices))
                    delete_index = random.sample(worse_indices, 1)[0]
                    logFile.write('delete #%d ins\n' % delete_index)
                    # rearrange insts, fitness and feature
                    insts[delete_index] = ins
                    fitness[delete_index] = new_fitness[i]
                    if delete_index not in all_deleted:
                        all_deleted.add(delete_index)
                        all_deleted_f_m = np.vstack((all_deleted_f_m, f_m[delete_index]))
                        logFile.write("len of all_deleted_f_m: %d\n" % all_deleted_f_m.shape[0])
                    f_m[delete_index] = new_feature[i]
        ite += 1

    # restore initial_insts to insts
    ori_len = len(initial_insts)
    logFile.write('Before instance generation, we have %d ins\n' % ori_len)

    for ins in insts:
        if ins not in initial_insts:
            initial_insts.append(ins)
    insts = initial_insts

    inst_names = construct_ins_file(insts, domain, folder='generated')

    new_len = len(insts)
    logFile.write('After instance generation, we have %d ins\n' % new_len)

    instanceIndexFile = path_con('AC_output/GenAS/generated/instance_index')
    with open(instanceIndexFile, 'w+') as f:
        for ins_name in inst_names:
            f.write('%s\n' % ins_name)

    # solve all new insts to optimum and merge new and old optimum
    if solutionFile is not None:
        optimum = dict()
        with open(solutionFile, 'r') as f:
            old_optimum = json.load(f)
        for i, ins_name in enumerate(initial_inst_names):
            optimum[inst_names[i]] = old_optimum[ins_name]
        optimum.update(solve_to_optimum(inst_names[ori_len:], domain))
        solutionFile = path_con('AC_output/GenAS/generated/generated_optimum.json')
        with open(solutionFile, 'w+') as f:
            json.dump(optimum, f)
    # test new insts with portfolio and obtain the per_matrix
    # note we need to know the performance of each instance on each solver
    n_per_matrix, _ = test_new([], metric, solution_checker, portfolio,
                               minTestTimes, cutoffTime,
                               domain, False,
                               inst_names=inst_names[ori_len:],
                               solution_file=solutionFile, complete=True)
    per_matrix = np.concatenate([per_matrix, n_per_matrix], axis=1)
    logFile.write('Final mean fitness: %f\n' % np.mean(np.min(per_matrix, axis=0)))
    return instanceIndexFile, solutionFile, per_matrix
