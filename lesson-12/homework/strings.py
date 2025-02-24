import csv

ids = []
names = []
ages = []

with open('student.csv', 'r', newline='', encoding='utf-8') as file:
    for row in csv.DictReader(file):
        ids.append(int(row['ID']))
        names.append(row['Name']) 
        ages.append(int(row['Age']))

res = {'id': ids, 'name': names, 'age': ages}
print(f"res = {res}")

