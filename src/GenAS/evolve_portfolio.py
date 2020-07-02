import os
import glob
import random
import subprocess
import sys
import time
from datetime import datetime
sys.path.append(sys.path[0] + '/../../')
import numpy as np
from path_converter import path_con


def validate(instanceIndexFile, solutionFile, metric, solution_checker,
             minTestTimes, cutoffTime, acRuns, logFile, algNum, existingSolver_list):
    for i in range(acRuns):
        cmd = 'rm -r ' + path_con('validation_output/GenAS/run%d*' % i)
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()
    processes = set()
    logFile.write('We will validate the full portfolio-------\n')

    runs = range(acRuns)
    logFile.write('Executing %s runs\n' % str(runs))
    logFile.flush()

    solutioncmd = 'None' if solutionFile is None else solutionFile
    checkercmd = 'None' if solution_checker is None else solution_checker
    for runNum in runs:
        cmd = 'python GenAS_validation.py %s %s %s %s %d %d %d %d %s' %\
              (instanceIndexFile, solutioncmd, metric, checkercmd,
               minTestTimes, cutoffTime, runNum, algNum-1, existingSolver_list[runNum])
        p = subprocess.Popen(cmd, shell=True)
        processes.add(p)

    # waiting for validation finish
    while processes:
        time.sleep(20)
        finished = [pid for pid in processes if pid.poll() is not None]
        processes -= set(finished)

    # compare validation results
    outputdir = path_con('validation_output/GenAS/')
    if metric == 'runtime':
        punish = 10
    elif metric == 'quality':
        punish = 100000

    with open(instanceIndexFile, 'r') as f:
        instances = f.read().strip().split('\n')
    insL = len(instances)
    # performance matrix, [i,j] i+1 run j ins
    perM = np.zeros((acRuns, insL)) * np.nan
    detailedPerM = np.zeros((acRuns, algNum, insL)) * np.nan
    runCount = np.zeros(perM.shape) * np.nan
    runCountD = np.zeros(detailedPerM.shape) * np.nan
    # write to /validation_output/GenAS/validation_results.txt
    fileList = os.listdir(outputdir)
    for f in fileList:
        if 'run' in f:
            begin = f.find('n')
            end = f.find('_')
            run_number = int(f[begin + 1:end])
            begin = f.find('s')
            end = f.find('S') - 1
            ins_index = int(f[begin + 1:end])
            with open(outputdir + f, 'r') as f:
                lines = f.read().strip().split('\n')
            outputLine = lines[0]
            values = outputLine[outputLine.find(':') + 1:].strip().replace(
                ' ', '').split(',')
            (status, runtime, quality) = (values[0], float(values[1]), float(values[3]))
            if metric == 'runtime' and 'TIMEOUT' in status:
                runtime = runtime * punish
            if metric == 'quality' and 'TIMEOUT' in status:
                quality = punish
            if np.isnan(perM[run_number, ins_index]):
                if metric == 'runtime':
                    perM[run_number, ins_index] = runtime
                elif metric == 'quality':
                    perM[run_number, ins_index] = quality
                runCount[run_number, ins_index] = 1
            else:
                if metric == 'runtime':
                    perM[run_number, ins_index] += runtime
                elif metric == 'quality':
                    perM[run_number, ins_index] += quality
                runCount[run_number, ins_index] += 1

            for line in lines[1:algNum+1]:
                detailedR = line.split(',')
                thread_index = int(detailedR[0])
                status = detailedR[1]
                runtime = float(detailedR[2])
                quality = float(detailedR[3])

                if metric == 'runtime' and 'TIMEOUT' in status:
                    runtime = runtime * punish
                if metric == 'quality' and 'TIMEOUT' in status:
                    quality = punish

                if np.isnan(detailedPerM[run_number, thread_index-1, ins_index]):
                    if metric == 'runtime':
                        detailedPerM[run_number, thread_index-1, ins_index] = runtime
                    elif metric == 'quality':
                        detailedPerM[run_number, thread_index-1, ins_index] = quality
                    runCountD[run_number, thread_index-1, ins_index] = 1
                else:
                    if metric == 'runtime':
                        detailedPerM[run_number, thread_index-1, ins_index] += runtime
                    elif metric == 'quality':
                        detailedPerM[run_number, thread_index - 1, ins_index] += quality
                    runCountD[run_number, thread_index-1, ins_index] += 1

    perM = np.true_divide(perM, runCount)
    detailedPerM = np.true_divide(detailedPerM, runCountD)
    if np.sum(np.isnan(perM)) > 0:
        print 'there are nan in validation results, more validation budget!'
        sys.exit(1)
    return perM, detailedPerM


def validation(instanceIndexFile, solutionFile, metric, solution_checker,
               minTestTimes, cutoffTime, acRuns, logFile,
               algNum, existingSolver_list):
    # validate each portfolio, to determine the best one
    perM, detailedPerM = validate(instanceIndexFile, solutionFile, metric, solution_checker,
                                  minTestTimes, cutoffTime, acRuns, logFile, algNum,
                                  existingSolver_list)

    # clear run* files
    cmd = 'rm ' + path_con('validation_output/GenAS/run*')
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()
    # logFile.write('validation done-------, performance matrix of new configs\n')
    # logFile.write(str(perM) + '\n')

    return perM, detailedPerM


def gathering(acRuns):
    # fullConfigs are for initialize incumbent for SMAC
    configs = dict()
    fullConfigs = dict()
    for run in range(acRuns):
        outputDir = glob.glob(path_con("AC_output/GenAS/run%d/output/run%d/log-run*.txt" %\
                                       (run, run)))[0]
        with open(outputDir, "r") as FILE:
            lines = FILE.read().strip()
            lines = lines[lines.find('has finished'):]
            lines = lines[lines.find('-@1'):]
            configs[run] = lines.split('\n')[0]

        outputDir = glob.glob(path_con("AC_output/GenAS/run%d/output/run%d/detailed-traj-run-*.csv"%\
                                       (run, run)))[0]
        with open(outputDir, 'r') as f:
            line = f.read().strip().split('\n')[-1]
            line = line.replace(' ', '').replace('"', '')
            fullConfig = line.split(',')[5:-1]
            for j, value in list(enumerate(fullConfig)):
                fullConfig[j] = '-' + value.replace('=', ' ')
            fullConfigs[run] = ' '.join(fullConfig)

    return configs, fullConfigs


def configuration(instanceIndexFile, paramFile, solutionFile, metric,
                  solution_checker, configurationTime, cutoffTime, acRuns,
                  logFile, existingSolver_list, initialInc, algNum):

    # first we need to write the rest portfolio into existing_solver.txt
    for run_number in range(acRuns):
        with open('exsiting_solver_%d.txt' % run_number, 'w+') as f:
            f.write(existingSolver_list[run_number] + '\n')
            f.write('%d\n' % (algNum-1))

    for runNum in range(acRuns):
        cmd1 = "rm -r " + path_con("AC_output/GenAS/run" + str(runNum) + "/output")
        cmd2 = "mkdir " + path_con("AC_output/GenAS/run" + str(runNum) + "/output")
        tmp = subprocess.Popen(cmd1, shell=True)
        tmp.communicate()
        tmp = subprocess.Popen(cmd2, shell=True)
        tmp.communicate()

    # then we need to construct scenario files
    training = instanceIndexFile
    testing = training

    for run_number in range(acRuns):
        scenarioFile = path_con('AC_output/GenAS/run%d/scenario.txt' % run_number)
        f = open(scenarioFile, "w+")
        lines = []
        algo = "algo = python %s %d " %\
               (path_con('src/GenAS/GenAS_solver.py '), run_number)
        if solutionFile:
            algo += '--opt-fn %s ' % solutionFile
        run_obj = 'RUNTIME'
        overall_obj = 'MEAN10'
        if metric == 'quality':
            run_obj = 'QUALITY'
            overall_obj = 'MEAN'
            algo += '--full-performance '
        if solution_checker:
            algo += '--solution-checker %s ' % solution_checker
        algo = algo + '\n'
        lines.append(algo)
        lines.append("execdir = /\n")
        lines.append("deterministic = 0\n")
        lines.append("run_obj = %s\n" % run_obj)
        lines.append("overall_obj = %s\n" % overall_obj)
        lines.append(("target_run_cputime_limit = " + str(cutoffTime) + "\n"))
        lines.append("paramfile = %s\n" % paramFile)
        lines.append(("instance_file = " + training + "\n"))
        lines.append(("test_instance_file = " + testing + "\n"))
        lines.append('outdir = ' + path_con('AC_output/GenAS/run%d/output' % run_number))

        f.writelines(lines)
        f.close()

    # run configurators

    logFile.write('Executing %s runs\n' % str(acRuns))
    os.chdir(path_con("AC"))
    pool = set()
    seedList = []
    while len(seedList) <= acRuns:
        seed = random.randint(1, 10000000)
        if seed not in seedList:
            seedList.append(seed)

    # 0 2 4 6 8 : use default
    # 1 3 5 7 9 : use random
    # note currently we do not use initialInc
    for run_number in range(acRuns):
        if run_number % 2 == 0:
            cmd = "./smac " + " --scenario-file " +\
                    path_con('AC_output/GenAS/run%d/scenario.txt' % run_number) +\
                    " --wallclock-limit " + \
                    str(configurationTime) + " --seed " + str(seedList[run_number]) + \
                    " --validation false " + \
                    " --console-log-level OFF" + \
                    " --log-level INFO"
        else:
            cmd = "./smac " + " --scenario-file " +\
                path_con('AC_output/GenAS/run%d/scenario.txt' % run_number) +\
                " --wallclock-limit " + \
                str(configurationTime) + " --seed " + str(seedList[run_number]) + \
                " --validation false " + \
                " --console-log-level OFF" + \
                " --log-level INFO" + \
                " --initial-incumbent RANDOM "
        # if run_number <= 3:
        #     cmd = "./smac " + " --scenario-file " +\
        #             path_con('AC_output/GenAS/run%d/scenario.txt' % run_number) +\
        #             " --wallclock-limit " + \
        #             str(configurationTime) + " --seed " + str(seedList[run_number]) + \
        #             " --validation false " + \
        #             " --console-log-level OFF" + \
        #             " --log-level INFO"
        # elif run_number <= 3:
        #     cmd = "./smac " + " --scenario-file " +\
        #         path_con('AC_output/GenAS/run%d/scenario.txt' % run_number) +\
        #         " --wallclock-limit " + \
        #         str(configurationTime) + " --seed " + str(seedList[run_number]) + \
        #         " --validation false " + \
        #         " --console-log-level OFF" + \
        #         " --log-level INFO" + \
        #         " --initial-incumbent " + '"' + initialInc + ' "'
        # elif run_number <= 9:
        #     cmd = "./smac " + " --scenario-file " +\
        #         path_con('AC_output/GenAS/run%d/scenario.txt' % run_number) +\
        #         " --wallclock-limit " + \
        #         str(configurationTime) + " --seed " + str(seedList[run_number]) + \
        #         " --validation false " + \
        #         " --console-log-level OFF" + \
        #         " --log-level INFO" + \
        #         " --initial-incumbent RANDOM "

        pool.add(subprocess.Popen(cmd, shell=True))

    finished = False
    estimated_time = 0
    while not finished:
        time.sleep(20)
        estimated_time += 20
        Finished_pid = [pid for pid in pool if pid.poll() is not None]
        pool -= set(Finished_pid)
        if not bool(pool):
            finished = True
        if estimated_time % 3600 == 0:
            logFile.write(str(datetime.now()) + "\n")
            logFile.write("Now " + str(len(pool)) + " AC" + " are running\n")
            logFile.flush()
            cmd = 'free -m'
            logFile.write(str(subprocess.check_output(cmd, shell=True)))
            logFile.flush()
            logFile.write("Now running tasks: " +
                          subprocess.check_output("ps r|wc -l", shell=True))
            logFile.flush()
    os.chdir(path_con('src/GenAS'))


def portfolio_evolution(portfolio, portfolio_fullconfig, algNum, instanceIndexFile,
                        paramFile, solutionFile, metric, solution_checker,
                        configurationTime, minTestTimes, cutoffTime, per_matrix, acRuns,
                        logFile, all_improve, ites=None):
    if ites is None:
        ites = algNum
    logFile.write('Evolve portoflio, %d iterations in total\n' % ites)
    for i in range(ites):
        # note improving for algNum times is intuitive
        # first we find the config with the least contribution
        logFile.write('ite %d---\n' % i)
        logFile.write('Compute contribution of each component solver\n')

        delete_index = None
        worst_contribution = None
        current_quality = np.mean(np.min(per_matrix, axis=0))
        logFile.write('Quality of current portfolio: %f\n' % (current_quality))
        for j in range(algNum):
            contribution = np.mean(np.min(np.delete(per_matrix, j, axis=0), axis=0)) -\
                           current_quality
            logFile.write('Alg %d, contribution %f\n' % (j, contribution))
            if worst_contribution is None or contribution < worst_contribution:
                worst_contribution = contribution
                delete_index = j

        # remove it from portoflio
        logFile.write('selected index: %d\n' % delete_index)
        # tmp_per_matrix = np.delete(per_matrix, delete_index, axis=0)
        # create existing solver cmd list
        existingSolver_list = []
        indices = []
        for alg_index in range(algNum):
            if alg_index == delete_index:
                indices.extend([alg_index] * ((acRuns/algNum)+(acRuns%algNum)))
            else:
                indices.extend([alg_index] * (acRuns/algNum))
        if not all_improve:
            indices = [delete_index] * acRuns

        for indice in indices:
            existingSolver = ''
            j = 1
            for k in range(algNum):
                if k == indice:
                    continue
                existingSolver = existingSolver + ' ' +\
                                 portfolio[k].replace('-@1', '-@%d' % j) + ' '
                j += 1
            existingSolver_list.append(existingSolver)
        initialInc = portfolio_fullconfig[delete_index]

        configuration(instanceIndexFile, paramFile, solutionFile, metric,
                      solution_checker, configurationTime, cutoffTime, acRuns,
                      logFile, existingSolver_list, initialInc, algNum)
        # gathering
        configs, fullconfigs = gathering(acRuns)

        # validation
        n_per_matrix, d_n_per_matrix = validation(instanceIndexFile, solutionFile, metric,
                                                  solution_checker, minTestTimes, cutoffTime,
                                                  acRuns, logFile, algNum, existingSolver_list)

        # select the best one, check if the best one is better than current portfolio
        best_one = None
        best_quality = None
        logFile.write('Current quality %f\n' % current_quality)
        logFile.write('Validation results\n')
        for j in range(acRuns):
            logFile.write('Run %d, replace solver %d, from overall %f, from detail %f\n' %\
                          (j, indices[j], np.mean(n_per_matrix[j, :]),
                           np.mean(np.min(d_n_per_matrix[j, :, :], axis=0))))
            if best_one is None or np.mean(n_per_matrix[j, :]) < best_quality:
                best_one = j
                best_quality = np.mean(n_per_matrix[j, :])
        logFile.write('Best one: %d Quality: %f\n' % (best_one, best_quality))
        if best_quality < current_quality:
            logFile.write('best quality: %f < current quality %f, update portfolio\n' %\
                          (best_quality, current_quality))
            portfolio[indices[best_one]] = configs[best_one]
            portfolio_fullconfig[indices[best_one]] = fullconfigs[best_one]
            per_matrix = d_n_per_matrix[best_one, :, :].reshape((d_n_per_matrix.shape[1], d_n_per_matrix.shape[2]))
            per_matrix = np.vstack((per_matrix[:indices[best_one]],\
                                    per_matrix[-1], per_matrix[indices[best_one]:-1]))
        else:
            logFile.write('best quality: %f >= current quality %f, do nothing\n' %\
                          (best_quality, current_quality))


        # if it is better, update portoflio and portoflio_fullconfig
        # otherwise do nothing
        # add_index = None
        # best_contribution = None
        # current_quality = np.mean(np.min(tmp_per_matrix, axis=0))
        # for j in range(acRuns):
        #     logFile.write('trying insert #%d config\n' % j)
        #     contribution = current_quality -\
        #                    np.mean(np.min(np.vstack((n_per_matrix[j, :], tmp_per_matrix)), axis=0))
        #     logFile.write('contribtuion: %f\n' % contribution)
        #     if best_contribution is None or contribution > best_contribution:
        #         best_contribution = contribution
        #         add_index = j
        # logFile.write('Add index %d, contribution %f\n' % (add_index, best_contribution))
        # if best_contribution >= worst_contribution:
        #     logFile.write('best_contribution: %f > worst_contribution %f, update portfolio\n' %\
        #                   (best_contribution, worst_contribution))
        #     portfolio[delete_index] = configs[add_index]
        #     portfolio_fullconfig[delete_index] = fullconfigs[add_index]
        #     per_matrix[delete_index, :] = n_per_matrix[add_index, :]
        # else:
        #     logFile.write('best_contribution: %f < worst_contribution %f, do nothing\n' %\
        #       (best_contribution, worst_contribution))

        current_quality = np.mean(np.min(per_matrix, axis=0))
        logFile.write('End insert, Quality of current portfolio: %f\n' % (current_quality))

    return portfolio, portfolio_fullconfig, per_matrix
