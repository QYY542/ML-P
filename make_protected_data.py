import pandas as pd
from diffprivlib.tools import mean, std
from diffprivlib.mechanisms import Laplace

def apply_differential_privacy_adult(csv_path, output_path, epsilon=1.0):
    df = pd.read_csv(csv_path)

    numeric_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    sensitivity = 1  # 假定每列的敏感度为1

    for column in numeric_columns:
        mechanism = Laplace(epsilon=epsilon, sensitivity=sensitivity)
        df[column] = df[column].apply(lambda x: mechanism.randomise(x))

    df.to_csv(output_path, index=False)
    print(f"Protected adult dataset has been saved to {output_path}.")

def apply_differential_privacy_obesity(csv_path, output_path, epsilon=1.0):
    df = pd.read_csv(csv_path)

    numeric_columns = ['Age', 'Height', 'Weight']
    sensitivity = 1  # 假定每列的敏感度为1

    for column in numeric_columns:
        mechanism = Laplace(epsilon=epsilon, sensitivity=sensitivity)
        df[column] = df[column].apply(lambda x: mechanism.randomise(x))

    df.to_csv(output_path, index=False)
    print(f"Protected obesity dataset has been saved to {output_path}.")

def apply_differential_privacy_student(csv_path, output_path, epsilon=1.0):
    df = pd.read_csv(csv_path)

    numeric_columns = [
        'Application order', 'Previous qualification (grade)', 'Admission grade',
        'Age at enrollment', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)'
    ]
    sensitivity = 1  # 假定每列的敏感度为1

    for column in numeric_columns:
        mechanism = Laplace(epsilon=epsilon, sensitivity=sensitivity)
        df[column] = df[column].apply(lambda x: mechanism.randomise(x))

    df.to_csv(output_path, index=False)
    print(f"Protected student dataset has been saved to {output_path}.")







if __name__ == '__main__':
    apply_differential_privacy_adult("./dataloader/datasets/adult/adult.csv", "./dataloader/datasets/adult/Protected_adult.csv")
    apply_differential_privacy_obesity("./dataloader/datasets/obesity/obesity.csv", "./dataloader/datasets/obesity/Protected_obesity.csv")
    apply_differential_privacy_student("./dataloader/datasets/student/student.csv", "./dataloader/datasets/student/Protected_student.csv")