import os
import subprocess

a = os.getpid()
print(a)


def always():
    import numpy as np
    while True:
        np.random.rand(100000)


if __name__ == '__main__':
    subprocess.run(f"ps -p {a} -o %cpu,%mem", shell=True)
    always()
