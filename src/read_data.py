from env import CuttingStock
import Parameters
DATA_PATH = Parameters.DATA_PATH
# read file


def instance_test(count, name_file):
    f = open(
        DATA_PATH+name_file, 'r')
    roll_count = int(f.readline())
    roll_length = int(f.readline())
    order_length = []
    order_count = []
    for b in range(roll_count):
        wj_dj = f.readline().split()
        order_length.append(int(wj_dj[0]))
        order_count.append(int(wj_dj[1]))
    name_ = name_file
    INSTANCE = CuttingStock(order_count, order_length, roll_length, name_)
    return INSTANCE
