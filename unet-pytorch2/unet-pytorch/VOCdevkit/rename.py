
import os
import re

My_path = r'D:\Code\Pycharm\unet-pytorch\unet-pytorch\dataset\test_xml_center'
patient_lists = os.listdir(My_path)


for i in range(len(patient_lists)):
    matchObj = re.match(r'(.*)-icontour-manual.txt', patient_lists[i])
    p_id = matchObj.group(1)

    old_path = os.path.join(My_path, patient_lists[i])
    new_name = p_id + ".txt"
    new_path = os.path.join(My_path, new_name)
    os.rename(old_path, new_path)
