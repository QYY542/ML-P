from sklearn.preprocessing import StandardScaler
import numpy as np

import csv

# 指定原始CSV文件和目标CSV文件的名称
input_csv_file = 'dataloader/datasets/student/student.csv'
output_csv_file = 'dataloader/datasets/student/student.csv'

# 读取原始CSV文件的内容，并替换分号为逗号，然后写入新文件
with open(input_csv_file, 'r', encoding='utf-8') as infile, \
        open(output_csv_file, 'w', newline='', encoding='utf-8') as outfile:
    for line in infile:
        modified_line = line.replace(';', ',')
        outfile.write(modified_line)

print(f'All semicolons replaced with commas in the file. New file saved as {output_csv_file}.')
