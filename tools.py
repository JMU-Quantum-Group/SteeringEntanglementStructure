from itertools import product

import numpy as np
import picos as pic

sigma_z = np.matrix([[1, 0], [0, -1]])
sigma_x = np.matrix([[0, 1], [1, 0]])
sigma_y = np.matrix([[0, -1j], [1j, 0]])


def random_state(n):
    real_current_matrix = np.random.rand(n, 1)
    real_current_matrix = np.matrix(real_current_matrix)

    imaginary_current_matrix = np.random.rand(n, 1)
    imaginary_current_matrix = np.matrix(imaginary_current_matrix)

    current_matrix = real_current_matrix + 1j * imaginary_current_matrix

    state = current_matrix * np.transpose(np.conj(current_matrix))
    return np.array(state / state.trace())


def generate_measure_init(untrusted_part, setting_num):
    measure_vec_list = list()
    for part_index in range(len(untrusted_part)):
        measure_vec_list.append(generate_measure_init_by_setting(setting_num))
    return measure_vec_list


class PartitionException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


def swap_chars(s, i, j):
    lst = list(s)
    lst[i], lst[j] = lst[j], lst[i]
    return ''.join(lst)


def generate_all_exchange_matrix(n_qubit):
    exchange_matrix_np = list()
    for num1 in range(n_qubit - 1):
        temp_matrix_list_np = list()
        for num2 in range(num1 + 1, n_qubit):
            the_matrix = np.zeros([2 ** n_qubit, 2 ** n_qubit])
            for number in range(2 ** n_qubit):
                number_str = format(number, '0{}b'.format(n_qubit))
                number_str = swap_chars(number_str, num1, num2)
                number_23 = int(number_str, 2)
                the_matrix[number, number_23] = 1
            temp_matrix_list_np.append(np.matrix(the_matrix))
        exchange_matrix_np.append(temp_matrix_list_np)
    return exchange_matrix_np


def generate_measure_init_by_setting(setting_num):
    measure_vec_list = list()
    for i in range(setting_num):
        measure_vec = random_one_measure_vec()
        measure_vec_list.append(measure_vec)
    return measure_vec_list


def handle_partition_list(partition_list, untrusted_part):
    sdp_part_list = list()
    for partition in partition_list:
        current_partition_by_set = partition.partition_by_set
        sdp_part_item = [s - set(untrusted_part) for s in current_partition_by_set]
        sdp_part_item = list(filter(None, sdp_part_item))
        sdp_part_item = handle_partition_set(sdp_part_item)
        sdp_part_list.append(sdp_part_item)
    return sdp_part_list


def handle_partition_set(sdp_part_item):
    if len(sdp_part_item) == 1:
        return sdp_part_item
    else:
        new_sdp_part_item = list()
        current_one_list = list()
        for s in sdp_part_item:
            if len(s) > 1:
                new_sdp_part_item.append(s)
            else:
                current_one_list.append(s)
                if len(current_one_list) == 2:
                    new_sdp_part_item.append(current_one_list)
                    current_one_list = list()
        if len(current_one_list) > 0:
            new_sdp_part_item.append(current_one_list[0])
        if len(new_sdp_part_item) > 2:
            raise PartitionException("Unsupported Partition")
        else:
            return new_sdp_part_item


def random_measure_vec_list(current_measure, angle):
    measure_vec_list = list()
    measure_vec_list.append(current_measure)

    new_x = np.cross(current_measure, random_one_measure_vec())
    new_x = new_x / np.linalg.norm(new_x)
    new_y = np.cross(current_measure, new_x)
    new_y = new_y / np.linalg.norm(new_y)

    direction = [new_x, np.sqrt(2) / 2 * (new_x + new_y), new_y, np.sqrt(2) / 2 * (-new_x + new_y),
                 -new_x, np.sqrt(2) / 2 * (-new_x - new_y), -new_y, np.sqrt(2) / 2 * (new_x - new_y)]

    # direction = [new_x, new_y, -new_x, -new_y]

    for item in direction:
        new_measure = current_measure * np.cos(angle) + item * np.sin(angle)
        measure_vec_list.append(new_measure)

    return measure_vec_list


def random_one_measure_vec():
    theta = 2 * np.pi * np.random.rand()
    phi = np.arccos(2 * np.random.rand() - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.array([x, y, z])


def generate_measure_matrix_by_untrusted_part(measure_matrix_list, untrusted_part, n_qubit):
    untrusted_index = 0
    measure_matrix = None
    for i in range(n_qubit):
        if untrusted_index < len(untrusted_part) and i == untrusted_part[untrusted_index]:
            if measure_matrix is None:
                measure_matrix = measure_matrix_list[untrusted_index]
            else:
                measure_matrix = np.kron(measure_matrix, measure_matrix_list[untrusted_index])
            untrusted_index += 1
        else:
            if measure_matrix is None:
                measure_matrix = np.eye(2)
            else:
                measure_matrix = np.kron(measure_matrix, np.eye(2))
    return measure_matrix


def convert_to_base_n(number, base, length=0):
    if number == 0:
        return [0] * length
    digits = []
    while number:
        digits.append(number % base)
        number //= base
    while len(digits) < length:
        digits.append(0)
    return digits[::-1]


def picos_generate_rho(prob, type_mark, index, i, sdp_part_list, part_index):
    if type_mark == 0:
        rho = pic.HermitianVariable('rho_' + str(index) + '_' + str(i), 2 ** len(sdp_part_list[i][part_index]))
        prob.add_constraint(rho >> 0)
    else:
        rho = pic.HermitianVariable('rho_' + str(index) + '_' + str(i), 4)
        prob.add_constraint(rho.partial_trace([1]) >> 0)
        # prob.add_constraint(rho.partial_transpose([0]) >> 0)
        prob.add_constraint(rho >> 0)
    return rho


def bubble_sort_steps(nums):
    steps = list()
    for i in range(len(nums)):
        for j in range(len(nums) - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                steps.append([j, j + 1])
    return steps


def numpy_generate_rho(type_mark, i, sdp_part_list, part_index):
    if type_mark == 0:
        rho = random_state(2 ** len(sdp_part_list[i][part_index]))
    else:
        rho = np.kron(random_state(2), random_state(2))
    return rho


def sdp_rho_func(prob, index, sdp_part_list, sdp_type_mark_list, part_index, rho_raw_list, exchange_matrix_step_list,
                 all_exchange_matrix_list, start_part_index_list):
    rho_next_list = list()
    new_rho_raw_list = list()
    rho_raw_index = 0
    for i in range(len(sdp_type_mark_list)):
        current_part_index = (start_part_index_list[i] + part_index) % 2
        if len(sdp_type_mark_list[i]) == 1:
            if part_index == 0:
                rho = picos_generate_rho(prob, sdp_type_mark_list[i][current_part_index], index, i, sdp_part_list,
                                         current_part_index)
                rho_next_list.append(rho)
                new_rho_raw_list.append(rho)
            else:
                if len(rho_raw_list) == 0:
                    rho_next_list.append(
                        numpy_generate_rho(sdp_type_mark_list[i][current_part_index], i, sdp_part_list,
                                           current_part_index))
                else:
                    rho_next_list.append(pic.Constant(rho_raw_list[rho_raw_index]))
                    rho_raw_index += 1
        else:
            rho = picos_generate_rho(prob, sdp_type_mark_list[i][current_part_index], index, i, sdp_part_list,
                                     current_part_index)
            # exchange part
            if len(rho_raw_list) == 0:
                if current_part_index == 0:
                    temp_rho_next = rho @ numpy_generate_rho(sdp_type_mark_list[i][1], i, sdp_part_list, 1)
                else:
                    temp_rho_next = numpy_generate_rho(sdp_type_mark_list[i][0], i, sdp_part_list, 0) @ rho
            else:
                if current_part_index == 0:
                    temp_rho_next = rho @ rho_raw_list[rho_raw_index]
                else:
                    temp_rho_next = rho_raw_list[rho_raw_index] @ rho
                rho_raw_index += 1
            for exchange_pair in exchange_matrix_step_list[i]:
                temp_rho_next = all_exchange_matrix_list[exchange_pair[0]][exchange_pair[1] - exchange_pair[0] - 1] * \
                                temp_rho_next * \
                                all_exchange_matrix_list[exchange_pair[0]][exchange_pair[1] - exchange_pair[0] - 1]
            rho_next_list.append(temp_rho_next)
            new_rho_raw_list.append(rho)
    return rho_next_list, new_rho_raw_list


def sdp_measure(rho, n_qubit, measure_vec_list, untrusted_part, sdp_part_list, sdp_type_mark_list, part_index,
                rho_raw_list, exchange_matrix_step_list, all_exchange_matrix_list, start_part_index_list):
    prob = pic.Problem()
    p = pic.RealVariable("p", 1)

    rho_right_list = list()

    for measure_vec_item_list in product(*measure_vec_list[:]):
        measure_matrix_list = [get_measure_list(measure_vec) for measure_vec in measure_vec_item_list]
        mea_vec_rho_right_list = list()
        for measure_matrix in product(*measure_matrix_list[:]):
            rho_right = ((p * rho + (1 - p) * np.eye(2 ** n_qubit) / (
                    2 ** n_qubit)) * generate_measure_matrix_by_untrusted_part(measure_matrix,
                                                                               untrusted_part,
                                                                               n_qubit)).partial_trace(
                untrusted_part)
            mea_vec_rho_right_list.append(rho_right)
        rho_right_list.append(mea_vec_rho_right_list)

    rho_sdp_list = list()
    new_rho_raw_list = list()
    for i in range(len(rho_right_list[0]) ** len(rho_right_list)):
        if len(rho_raw_list) == 0:
            rho_next_list, rho_list = sdp_rho_func(prob, i, sdp_part_list, sdp_type_mark_list, part_index,
                                                   list(), exchange_matrix_step_list, all_exchange_matrix_list,
                                                   start_part_index_list)
        else:
            rho_next_list, rho_list = sdp_rho_func(prob, i, sdp_part_list, sdp_type_mark_list, part_index,
                                                   rho_raw_list[i], exchange_matrix_step_list, all_exchange_matrix_list,
                                                   start_part_index_list)
        rho_sdp_list.append(rho_next_list)
        new_rho_raw_list.append(rho_list)

    index1 = 0
    for mea_vec_rho_right_list in rho_right_list:
        index2 = 0
        for rho_right in mea_vec_rho_right_list:
            rho_next = None
            for i in range(len(rho_right_list[0]) ** len(rho_right_list)):
                number_list = convert_to_base_n(i, len(rho_right_list[0]), len(rho_right_list))
                if number_list[index1] == index2:
                    if rho_next is None:
                        rho_next = rho_sdp_list[i][1]
                    else:
                        rho_next += rho_sdp_list[i][1]
                    # for rho_sdp_index in range(1, len(rho_sdp_list[i])):
                    #     rho_next += rho_sdp_list[i][rho_sdp_index]
            index2 += 1
            prob.add_constraint(rho_next == rho_right)
        index1 += 1

    # print(prob)
    prob.set_objective("max", p)
    prob.solve(solver="mosek", primals=True)
    # print(p.value)

    # p_value = p.value
    # the_right = pic.Constant(p_value * rho + (1 - p_value) * np.eye(2 ** n_qubit) / (2 ** n_qubit)).partial_trace(
    #     untrusted_part)
    # print(the_right)

    np_new_rho_list = list()
    the_left = np.zeros([2 ** (n_qubit - len(untrusted_part)), 2 ** (n_qubit - len(untrusted_part))],
                        dtype=np.complex128)
    for rho_list in new_rho_raw_list:
        current_np_matrix = [np.array(item) for item in rho_list]
        np_new_rho_list.append(current_np_matrix)
        for np_matrix in current_np_matrix:
            the_left += np_matrix
    # print(the_left)
    return p.value, np_new_rho_list


def get_measure_list(measure_vec):
    measure_matrix_0 = 0.5 * (
            np.eye(2) + (measure_vec[0] * sigma_x + measure_vec[1] * sigma_y + measure_vec[2] * sigma_z))
    measure_matrix_1 = 0.5 * (
            np.eye(2) - (measure_vec[0] * sigma_x + measure_vec[1] * sigma_y + measure_vec[2] * sigma_z))
    return [measure_matrix_0, measure_matrix_1]


def get_combinations(lst):
    if isinstance(lst[0], list):
        return [get_combinations(sublist) for sublist in lst]
    else:
        return lst


def handle_result(vec_list, need_seesaw, rho, n_qubit, untrusted_part, sdp_part_list, sdp_type_mark_list,
                  exchange_matrix_step_list, all_exchange_matrix_list, start_part_index_list):
    rho_raw_list = list()
    if need_seesaw:
        for i in range(10):
            result, rho_raw_list = sdp_measure(rho, n_qubit, vec_list, untrusted_part, sdp_part_list,
                                               sdp_type_mark_list, i % 2, rho_raw_list, exchange_matrix_step_list,
                                               all_exchange_matrix_list, start_part_index_list)
    else:
        result, rho_raw_list = sdp_measure(rho, n_qubit, vec_list, untrusted_part, sdp_part_list,
                                           sdp_type_mark_list, 0, rho_raw_list, exchange_matrix_step_list,
                                           all_exchange_matrix_list, start_part_index_list)
    return result


def opti_process(untrusted_part, setting_num, candidate_measure_vec, selected_elements, need_seesaw, rho, n_qubit,
                 sdp_part_list,
                 sdp_type_mark_list, exchange_matrix_step_list, all_exchange_matrix_list,
                 start_part_index_list):
    best_result = 3
    best_select_elements = list()
    for part_index in range(len(untrusted_part)):
        current_part_selected_list = list()
        for setting_index in range(setting_num):
            current_best_vec = None
            current_best_result = 2
            for vec in candidate_measure_vec[part_index][setting_index]:
                if setting_index == setting_num - 1:
                    current_part_vec = current_part_selected_list + [vec]
                else:
                    current_part_vec = current_part_selected_list + [vec] + selected_elements[part_index][
                                                                            setting_index + 1:]
                if part_index == len(untrusted_part) - 1:
                    current_vec_list = best_select_elements[0:part_index] + [current_part_vec]
                else:
                    current_vec_list = best_select_elements[0:part_index] + [current_part_vec] + selected_elements[
                                                                                                 part_index + 1:]

                result = handle_result(current_vec_list, need_seesaw, rho, n_qubit, untrusted_part, sdp_part_list,
                                       sdp_type_mark_list, exchange_matrix_step_list, all_exchange_matrix_list,
                                       start_part_index_list)
                if result < current_best_result:
                    print("part_index:", part_index, "setting_index:", setting_index, "result:", result, vec)
                    current_best_result = result
                    current_best_vec = vec
                    if current_best_result < best_result:
                        best_result = current_best_result
            current_part_selected_list.append(current_best_vec)
        best_select_elements.append(current_part_selected_list)
    return best_select_elements, best_result


def train(rho, n_qubit, measure_vec_list, setting_num, untrusted_part, best_result, angle, sdp_part_list):
    candidate_measure_vec = list()
    for part_index in range(len(untrusted_part)):
        candidate_part_measure_vec = list()
        for setting_index in range(setting_num):
            measure_vec = measure_vec_list[part_index][setting_index]
            current_measure_vec_list = random_measure_vec_list(measure_vec, angle)
            candidate_part_measure_vec.append(current_measure_vec_list)
        candidate_measure_vec.append(candidate_part_measure_vec)

    need_seesaw = False
    sdp_type_mark_list = list()
    exchange_matrix_step_list = list()
    all_exchange_matrix_list = generate_all_exchange_matrix(n_qubit - len(untrusted_part))
    start_part_index_list = list()

    # for seesaw
    for sdp_part_item in sdp_part_list:
        type_mark_item = list()
        concatenated_list = list()
        length_list = list()
        for part in sdp_part_item:
            if type(part) is set:
                type_mark_item.append(0)
                sorted_part_list = list(part)
                sorted_part_list.sort()
                concatenated_list.extend(sorted_part_list)
                length_list.append(len(sorted_part_list))
            else:
                type_mark_item.append(1)
                concatenated_list.extend(part[0])
                concatenated_list.extend(part[1])
                length_list.append(2)
        if len(length_list) > 1:
            start_part_index_list.append(1 if length_list[0] <= length_list[1] else 0)
        else:
            start_part_index_list.append(0)
        sdp_type_mark_list.append(type_mark_item)

        exchange_matrix_step_list.append(bubble_sort_steps(concatenated_list))

        if len(sdp_part_item) > 1:
            need_seesaw = True

    best_measure_vec_list = measure_vec_list

    print(best_measure_vec_list)
    best_select_elements, current_best_result = opti_process(untrusted_part, setting_num, candidate_measure_vec,
                                                             best_measure_vec_list,
                                                             need_seesaw, rho, n_qubit, sdp_part_list,
                                                             sdp_type_mark_list,
                                                             exchange_matrix_step_list, all_exchange_matrix_list,
                                                             start_part_index_list)

    print("first:", best_select_elements)
    if current_best_result < best_result:
        best_result = current_best_result

    best_select_elements, current_best_result = opti_process(untrusted_part, setting_num, candidate_measure_vec,
                                                             best_select_elements,
                                                             need_seesaw, rho, n_qubit, sdp_part_list,
                                                             sdp_type_mark_list,
                                                             exchange_matrix_step_list, all_exchange_matrix_list,
                                                             start_part_index_list)
    print("second:", best_select_elements)
    if current_best_result < best_result:
        best_result = current_best_result

    return best_select_elements, best_result
