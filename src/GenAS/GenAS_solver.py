import sys
import subprocess
sys.path.append(sys.path[0] + '/../../')
from path_converter import path_con

if __name__ == "__main__":
    run_number = int(sys.argv[1])
    del sys.argv[1]
    param_start = 6
    for start, param in enumerate(sys.argv):
        if '-@1' in param:
            param_start = start
            break
    params = ' '.join(sys.argv[param_start:])
    with open(path_con('src/GenAS/exsiting_solver_%d.txt' % run_number), 'r') as f:
        lines = f.readlines()
        existing_solver = ' ' + lines[0].strip() + ' '
        algNum = int(lines[1].strip())
    params = existing_solver + params.replace('-@1', '-@%d' % (algNum + 1))
    # sys.argv[1] = '"' + sys.argv[1] + '"'
    newparams = ' '.join(sys.argv[1:param_start]) + ' ' + params
    cmd = 'python ' + path_con('src/util/parallel_solver_wrapper.py %s' % newparams)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    print p.communicate()[0]
