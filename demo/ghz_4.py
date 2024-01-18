import numpy as np

from partition_tools import generate_k_producible_partitions, generate_k_partitionable_partitions
from tools import generate_measure_init, handle_partition_list, train

if __name__ == "__main__":
    rho = np.zeros((16, 16))
    n_qubit = 4

    indices = [0, 15]
    for index in indices:
        for index2 in indices:
            rho[index, index2] = 0.5

    untrusted_part = [1]  # start from 0
    untrusted_part_for_partition = [part_index + 1 for part_index in untrusted_part]  # start from 1
    setting_number = 2
    angle = np.pi / 36
    measure_vec_init_list = generate_measure_init(untrusted_part, setting_number)

    # handle partition
    partition_list = generate_k_partitionable_partitions(4, 3)
    sdp_part_list = handle_partition_list(partition_list, untrusted_part_for_partition)

    best_result = 2

    for epoch in range(30):
        print("epoch", epoch, "angle", angle)
        current_best_result = best_result
        best_measure_vec_list, best_result = train(rho, n_qubit, measure_vec_init_list, setting_number, untrusted_part,
                                                   current_best_result, angle, sdp_part_list)
        if current_best_result == best_result:
            angle = angle / 2

