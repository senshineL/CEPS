'''
Generative Adversarial Search for Parallel Portfolios
'''
import subprocess
import pickle
import sys
sys.path.append(sys.path[0] + '/../../')
from path_converter import path_con
from src.GenAS.init_portfolio import initialization
from src.GenAS.evolve_portfolio import portfolio_evolution
from src.GenAS.instance_generation import insGeneration

# Set parameter file and algorithm number
paramFile = path_con('Solver/paramfile/Single_ga_pcs.txt')
algNum = 4

# Set initial training instance index file
domain = 'VRPSPDTW'
metric = 'quality'
option = 'mutator'
mode = 'small'
expNum = 1
augment = False
instanceIndexFile = path_con('instance_set/%s_%s/indices/training_index_%s_%d' %\
                             (option, domain, mode, expNum))
if augment:
    instanceIndexFile = path_con('instance_set/%s_%s/indices/training_index_%s_%d_augment' %\
                                 (option, domain, mode, expNum))

solutionFile = None
if domain == 'TSP':
    metric = 'runtime'
    solutionFile = path_con('instance_set/%s_%s/indices/tsp_set_optimum.json' %\
                                (option, domain))
    if augment:
        solutionFile = path_con('instance_set/%s_%s/indices/training_index_%s_%d_augment_optimum.json' %\
                                (option, domain, mode, expNum))
solution_checker = None
if domain == 'VRPSPDTW':
    metric = 'quality'
    solution_checker = path_con('instance_set/diverse_mutator/VRPSPDTW_solution_checker.py')

# Set number of iterations
num_ite = 5
# Set time options
initTime = 36000
configurationTime = 36000
# note now we are not using validation time
# we use mintesttime to control
validationTime = 900
generationTime = 7200
# Set target algorithm cutoff time
cutoffTime = 150
# Set performance metric
# 'runtime', PAR10
# 'quality, mean
# metric = 'runtime'
# Set Algorithm configurator runs for each component solver
acRuns = 10
# Set instance generation options
# diverse: if we should consider diverity in the instance generation
diverse = False
# skip: whether to skip instance generation
skip = True
minTestTimes = 1
maxIt_ins_gen = 10
maxIt_evolve_portfolio = 1
# True: means try improving each component solver
all_improve = True

currentP = dict()
logFile = open("GenAS_log.txt", "w+", buffering=0)

logFile.write('Initialization.....\n')
currentP, portfolio_fullconfig, per_matrix = initialization(domain, algNum, instanceIndexFile,
                                                            solutionFile, metric, solution_checker,
                                                            initTime, cutoffTime,
                                                            minTestTimes, logFile)

# for debug
with open('initial_portfolio', 'w+') as f:
    pickle.dump(currentP, f)
with open('initial_portfolio_fullconfig', 'w+') as f:
    pickle.dump(portfolio_fullconfig, f)
with open('initial_permatrix', 'w+') as f:
    pickle.dump(per_matrix, f)

# with open('initial_portfolio', 'r') as f:
#     currentP = pickle.load(f)
# with open('initial_portfolio_fullconfig', 'r') as f:
#     portfolio_fullconfig = pickle.load(f)
# with open('initial_permatrix', 'r') as f:
#     per_matrix = pickle.load(f)

ite = 1
while ite <= num_ite:
    logFile.write('# iteration: %d, portoflio evolution begins\n' % ite)
    currentP, portfolio_fullconfig, per_matrix = portfolio_evolution(currentP, portfolio_fullconfig,
                                                                     algNum, instanceIndexFile,
                                                                     paramFile, solutionFile,
                                                                     metric, solution_checker,
                                                                     configurationTime,
                                                                     minTestTimes, cutoffTime,
                                                                     per_matrix, acRuns, logFile,
                                                                     all_improve,
                                                                     ites=maxIt_evolve_portfolio)

    # for debug
    with open('portfolio_ite_%d' % ite, 'w+') as f:
        pickle.dump(currentP, f)
    with open('portfolio_fullconfig_ite_%d' % ite, 'w+') as f:
        pickle.dump(portfolio_fullconfig, f)
    with open('permatrix_ite_%d' % ite, 'w+') as f:
        pickle.dump(per_matrix, f)

    if ite == num_ite:
        break
    # with open('initial_portfolio_ite_%d' % ite, 'r') as f:
    #     currentP = pickle.load(f)
    # with open('initial_portfolio_fullconfig_ite_%d' % ite, 'r') as f:
    #     portfolio_fullconfig = pickle.load(f)
    # with open('initial_permatrix_ite_%d' % ite, 'r') as f:
    #     per_matrix = pickle.load(f)

    logFile.write('Instance generation begins:\n')
    if skip:
        ite += 1
        continue
    instanceIndexFile, solutionFile, per_matrix = insGeneration(currentP, instanceIndexFile,
                                                                per_matrix, solutionFile,
                                                                metric, solution_checker,
                                                                generationTime, minTestTimes,
                                                                maxIt_ins_gen,
                                                                cutoffTime, domain, diverse,
                                                                logFile)
    ite += 1

# store the final solver
fullSolver = []
for runNum in range(algNum):
    fullSolver.append(currentP[runNum].replace('-@1', '-@%d' % (runNum+1)))
fullSolver = ' '.join(fullSolver)
logFile.write('Final solver:\n%s' % fullSolver)
logFile.close()

# run testing
# python2 control_testing.py method TSP mode expNumber testTimes cutoff_time
cmd = 'cp GenAS_log.txt ' + path_con('validation_output/GenAS/validation_results.txt')
pid = subprocess.Popen(cmd, shell=True)
pid.communicate()

testTimes = 3
testScript = path_con('src/test/control_testing.py')
cmd = 'python %s GenAS %s %s %s %d %d %d' %\
      (testScript, domain, option, mode, expNum, testTimes, cutoffTime)
pid = subprocess.Popen(cmd, shell=True)
pid.communicate()
extractScript = path_con('src/test/extract_testing.py')
# python2 extract_testing.py method testSize testTimes
testSize = 219
cmd = 'python %s GenAS %d %d %d %s' % (extractScript, testSize, testTimes, algNum, metric)
pid = subprocess.Popen(cmd, shell=True)
pid.communicate()
