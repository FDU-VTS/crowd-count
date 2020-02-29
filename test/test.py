import copy


def twoSum(nums, target):
    nums_cp = copy.deepcopy(nums)
    nums.sort()
    i = 0
    j = len(nums) - 1
    for k in range(len(nums)):
        if i == j:
            return 0
        if nums[i] + nums[j] == target:
            index_i = nums_cp.index(nums[i])
            index_j = nums_cp.index(nums[j])
            if index_i == index_j:
                index_j = len(nums_cp) - 1 - nums_cp[::-1].index(nums[j])
            return [index_i, index_j]
        elif nums[i] + nums[j] > target:
            j -= 1
        else:
            i += 1


print(twoSum([3, 3], 6))

