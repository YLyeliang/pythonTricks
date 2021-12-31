file = "test.txt"
with open(file, 'w') as f:
    for i in range(10):
        f.write(f"{i}______________\n")

with open(file, 'r') as f:
    lines = f.read().splitlines()
    print(lines)
