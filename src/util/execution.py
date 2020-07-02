'''
Run cmds in paralle mode with time and memory limits.
Once any subprocess ends, check if it
was exited normally or terminated

Input: cmds, cutoff time, mem limit
Output: excution result of each cmd
including: exit status
'''

import time
import sys
import re
import psutil
import os
sys.path.append(sys.path[0] + '/../../')
from path_converter import path_con


class execution(object):
    def __init__(self, cmds, cutoff_time,
                 mem_limit, watcher_log,
                 solver_log, runsolver_log,
                 verbose=False, full_performance=False):
        self._cmds = cmds  # dict
        self._cutoff_time = cutoff_time  # number in seconds
        self._mem_limit = mem_limit
        self._verbose = verbose
        self._sub_process = dict()
        self._watcher_log = watcher_log
        self._solver_log = solver_log
        self._runsolver_log = runsolver_log
        self._runsolver = path_con('runsolver/src/runsolver')
        self._inq_freq = 0.1  # inqury frequency
        # self._crash_quality = 1000000000
        self._full_performance = full_performance # whether to termintate all process

        # self._fast_solving_time = 0.01
        self._sleeping = 0.01 # inqury frequency

    def float_regex(self):
        return "[+-]?\d+(?:\.\d+)?(?:[eE][+-]\d+)?"

    def run(self):
        for thread_id, cmd in self._cmds.items():
            self._watcher_log.write("Thread %d runsolver log: %s, solver log: %s\n" %\
                                   (thread_id, self._runsolver_log[thread_id].name,\
                                    self._solver_log[thread_id].name))
        self._watcher_log.flush()

        # Starting execution
        # starting_time = time.time()
        for thread_id, cmd in self._cmds.items():
            full_cmd = self._runsolver + " -M " + str(self._mem_limit) +\
                       " -C " + str(self._cutoff_time) +\
                       " -w " + self._runsolver_log[thread_id].name +\
                       " -o " + self._solver_log[thread_id].name + " " + cmd
            if self._verbose:
                print "Now execute " + full_cmd
            self._sub_process[thread_id] = psutil.Popen(full_cmd, shell=True)

        # Monitoring
        sucessStr = ['s UNSATISFIABLE', 's SATISFIABLE',
                     'Have hit the optimum', 'Has hit the optimum']
        if not self._full_performance:
            termination = False
            while not termination:
                time.sleep(self._sleeping)
                for thread_id, sub_process in self._sub_process.items():
                    if sub_process.poll() is not None:
                        # check solver_log
                        self._watcher_log.write('Thread %d terminates\n' %
                                                thread_id)
                        self._watcher_log.flush()
                        with open(self._solver_log[thread_id].name, 'r') as f:
                            content = f.read()
                            if any(s in content for s in sucessStr):
                                # success
                                termination = True
                                self._watcher_log.write('Reason: success')
                                self._watcher_log.flush()
                                break
                            if content.find('Successes/Runs'):
                                line = content[content.find('Successes/Runs'):]
                                rrr = re.search(r'\d+', line)
                                if rrr:
                                    if int(rrr.group()) > 0:
                                        termination = True
                                        self._watcher_log.write('Reason: success')
                                        self._watcher_log.flush()
                                        break
                            # crash
                            self._watcher_log.write('Reason: crash')
                            self._watcher_log.flush()
                            self._sub_process.pop(thread_id, 0)
                if not self._sub_process:
                    self._watcher_log.write('No thread running\n')
                    self._watcher_log.flush()
                    termination = True
                    break

            # termination and return
            self.__terminate()
        else:
            # check if subprocess all exits, rely on runsolver to terminate them
            working_process = set(self._sub_process.values())
            while working_process:
                time.sleep(0.1)
                finished = [pid for pid in working_process if pid.poll() is not None]
                working_process -= set(finished)
        # interpret output by all thread solvers
        status_dict = dict()
        time_dict = dict()
        # quality_dict = dict() # unused now
        for thread_id, runsolver_log in self._runsolver_log.items():
            status_dict[thread_id] = "TIMEOUT"
            time_dict[thread_id] = self._cutoff_time
            # quality_dict[thread_id] = self._crash_quality

            with open(runsolver_log.name, 'r') as f:
                data = f.read()

            if (re.search('runsolver_max_cpu_time_exceeded', data) or\
                re.search('Maximum CPU time exceeded', data)):
                status_dict[thread_id] = "TIMEOUT"

            if (re.search('runsolver_max_memory_limit_exceeded', data) or\
                re.search('Maximum VSize exceeded', data)):
                status_dict[thread_id] = "TIMEOUT"

            # cpu_pattern1 = re.compile('^runsolver_cputime: (%s)' %\
            #                         (self.float_regex()), re.MULTILINE)
            # cpu_match1 = re.search(cpu_pattern1, data)

            cpu_pattern2 = re.compile('total CPU time \\(s\\): (%s)' % (self.float_regex()))
            cpu_match2 = re.search(cpu_pattern2, data)

            # if cpu_match1:
            # time_dict[thread_id] = float(cpu_match1.group(1))
            if cpu_match2:
                time_dict[thread_id] = float(cpu_match2.group(1))
                if time_dict[thread_id] > self._cutoff_time:
                    time_dict[thread_id] = self._cutoff_time

            with open(self._solver_log[thread_id].name, 'r') as f:
                data = f.read()

            if any(s in data for s in sucessStr):
                # success
                status_dict[thread_id] = 'SUCCESS'
            if data.find('Successes/Runs'):
                line = data[data.find('Successes/Runs'):]
                rrr = re.search(r'\d+', line)
                if rrr:
                    if int(rrr.group()) > 0:
                        status_dict[thread_id] = 'SUCCESS'

            if status_dict[thread_id] != 'SUCCESS':
                time_dict[thread_id] = self._cutoff_time

        return status_dict, time_dict


    def __terminate(self):
        # Terminates all threads
        self._watcher_log.write("Now we are terminating all threads\n")
        self._watcher_log.flush()

        for thread_id, sub_process in self._sub_process.items():
            # terminate child process
            self._watcher_log.write('Terminate solver process of thread %d' %
                                    thread_id)
            self._watcher_log.flush()
            # try:
            #     children = sub_process.children(recursive=True)
            # except psutil.NoSuchProcess:
            #     continue
            # for p in children:
            #     try:
            #         p.terminate()
            #     except(psutil.NoSuchProcess, KeyError):
            #         # Key error means subprocess terminated before
            #         # obtaining child process
            #         pass
            #     else:
            #         waiting_time = 0
            #         time.sleep(self._sleeping)
            #         waiting_time += self._sleeping

            #         # ensure child process is terminated
            #         while psutil.pid_exists(p.pid):
            #             # Not terminated yet
            #             self._watcher_log.write(
            #                 "Waiting " + str(waiting_time) +
            #                 " for solver process of thread " +
            #                 str(thread_id) + " terminated \n")
            #             self._watcher_log.flush()
            #             time.sleep(self._sleeping)
            #             waiting_time += self._sleeping
            #             if waiting_time >= 1 and waiting_time <= 2:
            #                 # SEND SIGTERM AGAIN
            #                 try:
            #                     p.terminate()
            #                 except psutil.NoSuchProcess:
            #                     pass
            #             if waiting_time > 2:
            #                 # SEND SIGKILL
            #                 try:
            #                     p.kill()
            #                 except psutil.NoSuchProcess:
            #                     pass

            # terminate sub process
            self._watcher_log.write(
                "Trying to terminate subprocess of thread " +
                str(thread_id) + "\n")
            self._watcher_log.flush()

            try:
                sub_process.terminate()
            except psutil.NoSuchProcess:
                pass
            else:
                waiting_time = 0
                while sub_process.poll() is None:
                    # Not terminated yet
                    self._watcher_log.write(
                        "Waiting" + str(waiting_time) +
                        "for subprocess of thread " +
                        str(thread_id) + " terminated \n")
                    self._watcher_log.flush()
                    time.sleep(self._sleeping)
                    waiting_time += self._sleeping
                    if waiting_time >= 1 and waiting_time <= 3:
                        # SEND SIGTERM AGAIN
                        try:
                            sub_process.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    if waiting_time > 3:
                        # SEND SIGKILL
                        try:
                            sub_process.kill()
                        except psutil.NoSuchProcess:
                            pass

            self._watcher_log.write("Thread " + str(thread_id) +
                                    " is terminated\n")
            self._watcher_log.flush()
