import pandas as pd

read_file = pd.read_excel (r'C:/Users/AlanLin/Desktop/idk/train_data.xlsx')
read_file.to_csv (r'C:/Users/AlanLin/Desktop/idk/train_data.csv', index = None, header=True)