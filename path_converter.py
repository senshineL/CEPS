import os

def path_con(relative_path):
    return os.path.dirname(os.path.realpath(__file__)) + '/' + relative_path
