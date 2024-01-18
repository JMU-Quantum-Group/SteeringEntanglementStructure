def format_partition(partition_list):
    new_partition_list = [sorted(current_part) for current_part in partition_list]
    return sorted(new_partition_list, key=lambda part: part[0])


class Partition(object):
    def __init__(self, list_type_of_partition):
        self.partition_by_list = format_partition(list_type_of_partition)
        str_list = list()
        item_len_list = list()
        for part in self.partition_by_list:
            str_list.append(".".join([str(item) for item in part]))
            item_len_list.append(len(part))
        self.partition_by_str = "|".join(str_list)
        self.partition_by_set = [set(part) for part in self.partition_by_list]
        self.partitionable = len(self.partition_by_list)
        self.producible = max([len(part) for part in self.partition_by_list])
        self.stretchable = self.producible - self.partitionable

    def __le__(self, other):
        for part in self.partition_by_set:
            is_subset = False
            for part_other in other.partition_by_set:
                if part <= part_other:
                    is_subset = True
            if not is_subset:
                return False
        return True

    def __ge__(self, other):
        for part in other.partition_by_set:
            is_subset = False
            for part_self in self.partition_by_set:
                if part <= part_self:
                    is_subset = True
            if not is_subset:
                return False
        return True

    def __str__(self):
        return self.partition_by_str

    def __hash__(self):
        return hash(self.partition_by_str)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        else:
            return False

