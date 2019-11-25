from shutil import *
import shlex
import subprocess


def run(*args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', exitcode=False, shell=True, check=False):
    if len(args) == 1:
        args = args[0]

    if args:   
        proc = subprocess.run(args, stdout=stdout, stderr=stderr, shell=shell, check=check)
        output = proc.stdout.decode('utf-8').strip()
    else:
        exitcode = 0
        output = None
    return (proc.returncode, output) if exitcode else output