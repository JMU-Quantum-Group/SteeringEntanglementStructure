import copy
import itertools
from collections import Counter

from partition import Partition


def all_partition(elements, type_list, type_count):
    partition_list = list()
    if len(type_list) == 1:
        return handle_same_partition(elements, type_list[0])

    for item in itertools.combinations(elements, type_list[0] * type_count[type_list[0]]):
        the_part = list(item)
        the_last = copy.deepcopy(set(elements))
        the_last.difference_update(the_part)
        current_partition_list = handle_same_partition(the_part, type_list[0])
        for partition in all_partition(list(the_last), type_list[1:], type_count):
            for temp_partition in current_partition_list:
                partition_list.append(partition + temp_partition)

    return partition_list


def handle_same_partition(elements, length):
    partition_list = list()
    if len(elements) == length:
        partition_list.append([elements])
        return partition_list

    for item in itertools.combinations(elements[1:], length - 1):
        the_part = list(item) + [elements[0]]
        the_last = copy.deepcopy(set(elements))
        the_last.difference_update(the_part)
        partition_list.extend(
            [partition + [the_part] for partition in handle_same_partition(list(the_last), length)])
    return partition_list


def format_type(type_str):
    part_number_in_type = set()
    item_list = [int(x) for x in type_str.split('|')]
    for item in item_list:
        part_number_in_type.add(item)
    current_type_dict = dict(Counter(item_list))
    return list(part_number_in_type), current_type_dict


def get_partitions(n):
    result = []

    def partition_helper(current_n, max_val, current_partition):
        if current_n == 0:
            result.append(current_partition)
            return
        for i in range(min(max_val, current_n), 0, -1):
            partition_helper(current_n - i, i, current_partition + [i])

    partition_helper(n, n, [])
    return result


def max_type(target_type_list):
    remove_set = set()
    for target_type in target_type_list:
        item_list = [int(x) for x in target_type.split('|')]
        for i in range(len(item_list)):
            for j in range(i + 1, len(item_list)):
                new_item_list = item_list[0:i] + [item_list[i] + item_list[j]] + item_list[i + 1:j] + item_list[j + 1:]
                new_item_list = sorted(new_item_list, reverse=True)
                new_type = "|".join([str(item) for item in new_item_list])
                if new_type in target_type_list:
                    remove_set.add(target_type)
                    break
            if target_type in remove_set:
                break

    for target_type in remove_set:
        target_type_list.remove(target_type)


def generate_partition_list_from_type(target_type_list, partite_number):
    max_type(target_type_list)
    partition_list = list()
    for type_str in target_type_list:
        part_number_in_type, current_type_dict = format_type(type_str)
        current_partition_list = all_partition(list(range(1, partite_number + 1)), part_number_in_type,
                                               current_type_dict)
        partition_list.extend(current_partition_list)
    Gamma_list = list()
    for partition_str in partition_list:
        Gamma_list.append(Partition(partition_str))
    return Gamma_list


def generate_k_stretchable_partitions(partite_number, k):
    partition_type_list = get_partitions(partite_number)
    target_type_list = list()
    for partition_type in partition_type_list:
        stretchable = max(partition_type) - len(partition_type)
        if stretchable > k:
            continue
        type_string = "|".join([str(item) for item in partition_type])
        target_type_list.append(type_string)
    return generate_partition_list_from_type(target_type_list, partite_number)


def generate_k_producible_partitions(partite_number, k):
    partition_type_list = get_partitions(partite_number)
    target_type_list = list()
    for partition_type in partition_type_list:
        producible = max(partition_type)
        if producible > k:
            continue
        type_string = "|".join([str(item) for item in partition_type])
        target_type_list.append(type_string)
    return generate_partition_list_from_type(target_type_list, partite_number)


def generate_k_partitionable_partitions(partite_number, k):
    partition_type_list = get_partitions(partite_number)
    target_type_list = list()
    for partition_type in partition_type_list:
        partitionable = len(partition_type)
        if partitionable < k:
            continue
        type_string = "|".join([str(item) for item in partition_type])
        target_type_list.append(type_string)
    return generate_partition_list_from_type(target_type_list, partite_number)
