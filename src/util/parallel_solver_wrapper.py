#!/usr/bin/python
'''
Parallel solver wrapper invoked by Algorithm configurator
Smac/ParamilS.
Command Line:
python parallel_solver_wrapper.py
[--mem-limit limit] [--opt-fn file]
<instance name> <instance specific information> <cutoff time>
<cutoff length> <seed> <param> ...
<param> : -name 'value'
'''
import sys
import json
import subprocess
from tempfile import NamedTemporaryFile
sys.path.append(sys.path[0] + '/../../')
from src.util.parameter_parser import parameter_parser
from src.util.execution import execution


class parallel_solver_wrapper(object):
    def __init__(self):
        self._thread_to_solver = dict()
        self._thread_to_params = dict()
        self._solver_log = dict()
        self._runsolver_log = dict()
        self._cmds = dict()
        self._sub_process = dict()

        self._verbose = False
        self._mem_limit = 12 * 1024 # 12 GB at default
        self._opt_fn = None # used for optiming runtime for optimization problems
        self.paramFile = dict() # used shen solver reads parameter settings from file

        self._log_path = sys.path[0] + "/configlogs/"
        self._watcher_path = self._log_path
        self._full_performance = False
        self._solution_checker = None
        self._objective = 'runtime'

    def parse_args(self, argv):
        ''' parse command line arguments
            Args:
                argv: command line arguments list
        '''
        start = 1
        for i, arg in enumerate(argv[1:8]):
            if '--mem-limit' in arg:
                self._mem_limit = int(argv[i+2])
                start += 2
            elif '--opt-fn' in arg:
                self._opt_fn = argv[i+2]
                start += 2
            elif '--full-performance' in arg:
                self._full_performance = True
                start += 1
            elif '--solution-checker' in arg:
                self._solution_checker = argv[i+2]
                start += 2

        self._instance = argv[start]
        self._instance_specific = argv[start+1]
        self._cutoff_time = float(argv[start+2])
        self._cutoff_length = int(argv[start+3])
        self._random_seed = argv[start+4]
        self._params = argv[start+5:]

        # Handle spaces in instance name
        self._instance = self._instance.replace(" ", "\ ")

        # parse params to assign solvers and params to threads
        parser = parameter_parser()
        [self._thread_to_solver, self._thread_to_params, _] = parser.parse(
            self._params)

    def __construct_cmd(self, solver_path, parameter_list):
        # for lingeling-ala
        if "lingeling-ala" in solver_path:
            cmd = solver_path + " "
            for parameter in parameter_list:
                cmd = cmd + parameter + " "
            cmd = cmd + "--seed=" + self._random_seed + " "
            cmd = cmd + self._instance + " "

            return cmd

        # for lingeling-ars
        if "lingeling-ars" in solver_path:
            cmd = solver_path + " "
            for parameter in parameter_list:
                cmd = cmd + parameter + " "
            cmd = cmd + "--seed=" + self._random_seed + " "
            cmd = cmd + self._instance + " "

            return cmd

        # for clasp
        if 'clasp' in solver_path:
            cmd = solver_path + " "
            for parameter in parameter_list:
                cmd = cmd + parameter + " "
            cmd = cmd + "--seed=" + self._random_seed + " "
            cmd = cmd + " -f " + self._instance + " "

            return cmd

        # for riss fixed seed
        if 'riss' in solver_path\
            or 'TNM' in solver_path\
                or 'MPhaseSAT_M' in solver_path\
                or 'march_hi' in solver_path\
                or 'bin/lingeling' in solver_path\
                or 'satUZK_wrapper' in solver_path\
                or 'glucose_wrapper' in solver_path\
                or 'contrasat' in solver_path:
            cmd = solver_path + " "
            for parameter in parameter_list:
                cmd = cmd + parameter + " "
            cmd = cmd + self._instance + " "

            return cmd

        # for sparrow fixed seed: 0
        if 'sparrow' in solver_path:
            cmd = solver_path + " "
            for parameter in parameter_list:
                cmd = cmd + parameter + " "
            cmd = cmd + self._instance + ' 0'

            return cmd

        # for GA-EAX, we need opt fn
        if 'jikken' in solver_path:
            if self._opt_fn is None:
                print "No optimium file found for GA-EAX\n"
                sys.exit(1)
            else:
                optimumFile = self._opt_fn
            with open(optimumFile, 'r') as f:
                optimum = json.load(f)
            cmd = solver_path + ' 10000 ' + ' 1.txt '
            for parameter in parameter_list:
                if 'populationsize' in parameter:
                    pSize = parameter.strip().split('=')[1]
                if 'offspringsize' in parameter:
                    oSize = parameter.strip().split('=')[1]
            cmd = cmd + ' ' + pSize + ' ' + oSize + ' '
            cmd = cmd + ' ' + optimum[self._instance] + ' '
            cmd = cmd + self._instance

            return cmd

        # for CLK-linkern, we need opt fn
        if "linkern" in solver_path:
            if self._opt_fn is None:
                print "No optimium file found for CLK\n"
                sys.exit(1)
            else:
                optimumFile = self._opt_fn
            with open(optimumFile, 'r') as f:
                optimum = json.load(f)
            cmd = solver_path + " "
            for parameter in parameter_list:
                cmd = cmd + parameter + " "
            cmd = cmd + " -h " + optimum[self._instance]
            cmd = cmd + " -s 0 "
            cmd = cmd + self._instance + " "

            return cmd

        # for LKH, we need opt fn
        if 'LKH' in solver_path:
            if self._opt_fn is None:
                print "No optimium file found for LKH\n"
                sys.exit(1)
            else:
                optimumFile = self._opt_fn

            with open(optimumFile, 'r') as f:
                optimum = json.load(f)
            tmp = NamedTemporaryFile('w+b', prefix='Paramfile')
            self.paramFile[len(self.paramFile)] = tmp
            tmp.write('PROBLEM_FILE=%s\n' % self._instance)
            tmp.write('OPTIMUM=%s\n' % optimum[self._instance])
            for parameter in parameter_list:
                tmp.write(parameter[1:] + '\n')
            # tmp.write('SEED=%s\n' % self._random_seed)
            tmp.write('SEED=0\n')
            tmp.flush()
            cmd = solver_path + ' ' + tmp.name

            return cmd

        # for GA in VRPSPDTW, we need solution check
        if 'GA' in solver_path:
            if self._solution_checker is None:
                print "NO solution checker found for GA\n"
                sys.exit(1)
            self._objective = 'quality'
            # build cmd for GA
            # cmd = "%s %s %s " % (solver_path, self._instance, self._random_seed)
            cmd = "%s %s 0 " % (solver_path, self._instance)
            record = dict()
            for parameter in parameter_list:
                name, value = parameter.strip().split('=')
                record[name[1:]] = value

            cmd += record['PopSize'] + " "
            cmd += record['Num_MPCIH'] + " "
            cmd += record['Num_RSCIM'] + " "
            cmd += record['ConvergenceCountLimit'] + " "
            cmd += record['mutation_rate'] + " "
            cmd += record['Num_Elimisms'] + " "
            cmd += record['isolated_lower'] + " "
            cmd += record['isolated_upper'] + " "
            cmd += record['beta_start'] + " "
            cmd += record['beta_increment'] + " "
            cmd += record['reinsertion_thresh'] + " "
            cmd += record['swap_thresh'] + " "

            return cmd

    def __read_output(self, objective, status_dict, time_dict):
        # Result for SMAC: <solved>, <runtime>, <runlength>,
        # <quality>, <seed>, <additional rundata>
        if objective == 'runtime':
            result = "TIMEOUT"
            runtime = self._cutoff_time
            s_flag = 0
            for thread_id, status in status_dict.items():
                if status == 'SUCCESS':
                    result = status
                    if s_flag == 0:
                        runtime = time_dict[thread_id]
                    elif time_dict[thread_id] < runtime:
                        runtime = time_dict[thread_id]
                    s_flag += 1

            output_line = "Result for SMAC: " + result + ", " + \
                str(runtime) + ", 0, 0, " + str(self._random_seed) + ", 0"
        # when optimzing quality, need other interpretations
        elif objective == 'quality':
            runtime = self._cutoff_time
            result = 'TIMEOUT'
            quality = 0.0
            quality_dict = dict()
            s_flag = 0
            for thread_id in status_dict:
                cmd = 'python %s %s %s' % (self._solution_checker, self._instance,\
                                           self._solver_log[thread_id].name)
                pid = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                stdout, stderr = pid.communicate()
                if not stdout:
                    print 'no standard output from solution checker, stderr %s!\n' % stderr
                    sys.exit(1)
                # TIMEOUT: real crash or not find any feasible solutions
                # SUCCESS: find a feasible solution
                # cost: original cost
                # n_cost: normalizaed cost
                status, _, n_cost = stdout.strip().split(',')
                if status == 'TIMEOUT':
                    status_dict[thread_id] = 'TIMEOUT'
                    quality_dict[thread_id] = 100000
                elif status == 'SUCCESS':
                    status_dict[thread_id] = 'SUCCESS'
                    quality_dict[thread_id] = float(n_cost)
                    result = 'SUCCESS'
                    if s_flag == 0:
                        quality = quality_dict[thread_id]
                    elif quality_dict[thread_id] < quality:
                        quality = quality_dict[thread_id]
                    s_flag += 1
            output_line = "Result for SMAC: " + result + ", " +\
                          str(runtime) + ", 0, " + str(quality) + ", " +\
                          str(self._random_seed) + ", 0"
        return output_line

    def start(self):
        '''
        start solver
        '''
        # build watcher logs: temporary file
        self._watcher_log = NamedTemporaryFile(
            mode="w+b", prefix="watcher", dir=self._watcher_path)

        for thread_id in self._thread_to_solver:
            # build log files: temporary file
            self._solver_log[thread_id] = NamedTemporaryFile(
                mode="w+b", prefix=("Thread" + str(thread_id) + "solverLog"),
                dir=self._log_path)

            self._runsolver_log[thread_id] = NamedTemporaryFile(
                mode="w+b", prefix=("Thread" + str(thread_id) + "run_solverLog"),
                dir=self._log_path)

            self._cmds[thread_id] = self.__construct_cmd(
                self._thread_to_solver[thread_id],
                self._thread_to_params[thread_id])

        # call execution wrapper
        exe = execution(self._cmds, self._cutoff_time, self._mem_limit,
                        self._watcher_log, self._solver_log,
                        self._runsolver_log, self._verbose,
                        self._full_performance)

        status_dict, time_dict = exe.run()
        print self.__read_output(self._objective, status_dict, time_dict)


if __name__ == "__main__":
    parallel_solver = parallel_solver_wrapper()
    parallel_solver.parse_args(sys.argv)
    parallel_solver.start()
    if parallel_solver.paramFile:
        for k, v in parallel_solver.paramFile.items():
            v.close()
