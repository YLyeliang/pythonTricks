if __name__ == '__main__':
    x_list = [-1, -2, -3, 0, 1, 2, 3]
    x_list.sort(key=lambda x: abs(x - 0))
    print(x_list)
