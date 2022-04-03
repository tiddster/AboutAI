class FS():
    @staticmethod
    def FeatureScaling(nums):
        resNum = []
        maxVal, minVal = max(nums), min(nums)
        for num in nums:
            num = num / (maxVal-minVal)
            resNum.append(num)
        return resNum


if __name__ == '__main__':
    nums = [89,72,94,69]
    print(FS.FeatureScaling(nums))