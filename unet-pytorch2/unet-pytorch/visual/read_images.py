import os
import re
"""
"""

prival_path = r'D:\Code\Pycharm\unet-pytorch\unet-pytorch\dataset\test_xml_center'
out_path = r'D:\Code\Pycharm\unet-pytorch\unet-pytorch\dataset\ImageSets\test.txt'

patient_nums = os.listdir(prival_path)
# print(patient_nums)
nums_all = 0
list_all = []
for patient_num in patient_nums:
    matchObj = re.match(r'(.*).txt', patient_num)
    list_all.append(matchObj.group(1))
    # with open(out_path, "a") as file_txt:
    #     # file_txt.write(matchObj.group(1))
    #     file_txt.write(matchObj.group(1))
    #     file_txt.write('\n')
print(list_all)
# list_all = list(set(list_all))
# print(list_all)
for i in list_all:
    with open(out_path, "a") as file_txt:
    # file_txt.write(matchObj.group(1))
        file_txt.write(i)
        file_txt.write('\n')