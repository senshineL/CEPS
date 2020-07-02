import time
import sys
import glob
import psutil
sys.path.append(sys.path[0] + '/../../')
from path_converter import path_con


def start_validation(sol, instances, solution,
                     m, checker,
                     seeds, cutoffTime, test_times,
                     outdir, cutoff_length=0):

    for insIndex, instance in enumerate(instances):
        for _ in range(test_times):
            seed = int(seeds.pop(0))
            output_file = '%s_Ins%d_Seed%d' % (outdir, insIndex, seed)
            # for GenAS, we need full performance of each component solver
            cmd = 'python ' + path_con('src/util/testing_wrapper.py ')
            if solution:
                cmd += '--opt-fn %s ' % solution
            cmd += '--full-performance '
            if checker:
                cmd += '--solution-checker %s ' % checker
            cmd += '%s %s %s %s %s %s' % (instance, output_file, cutoffTime,
                                          cutoff_length, seed, sol)
            PID = psutil.Popen(cmd, shell=True)
            stdOut = PID.communicate()[0]
            print stdOut

if __name__ == "__main__":
    vInsIndex = sys.argv[1]
    solutionFile = sys.argv[2]
    metric = sys.argv[3]
    solution_checker = sys.argv[4]
    if 'None' in solutionFile:
        solutionFile = None
    if 'None' in solution_checker:
        solution_checker = None
    exsitingSolver = sys.argv[9:]
    algNum = int(sys.argv[8])

    with open(vInsIndex, "r") as FILE:
        instance_list = FILE.read().strip().split('\n')
    seed_index_file = path_con("validation_output/GenAS/seed_index.txt")
    with open(seed_index_file, "r") as FILE:
        seed_list = FILE.read().strip().split()

    # set algorithm
    run_number = int(sys.argv[7])
    outputdir = glob.glob(path_con("AC_output/GenAS/run%d/output/run%d/log-run*.txt" %\
                                   (run_number, run_number)))[0]
    with open(outputdir, "r") as FILE:
        lines = FILE.read().strip()
        lines = lines[lines.find('has finished'):]
        lines = lines[lines.find('-@1'):]
        solver = lines.split('\n')[0]
        solver = solver.replace('-@1', '-@%d' % (algNum+1))
    solver = ' '.join(exsitingSolver) + ' ' + solver
    # set other options
    cutoff_time = int(sys.argv[6])
    minTestTimes = int(sys.argv[5])
    outputdir = path_con("validation_output/GenAS/run%d" % run_number)

    start_validation(solver, instance_list, solutionFile, metric, solution_checker,
                     seed_list, cutoff_time, minTestTimes, outputdir)
