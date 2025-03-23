
# pyodbc -- odbc -- jobc
# sqlalchemy

import pyodbc

connection = pyodbc.connect( 
                               'DRIVER={SQL Server};'
                               'Server=DESKTOP-CSSCKLQ;'
                               'Database=master;'
                               'trusted_Connection=yes;'
                               )

print('Connected Successfully')


cursor = connection.cursor()

select_qy = '''select * from [W3Resource].[Employee].[emp_details]'''

res = cursor.execute(select_qy)

# import json
# with open('employee.csv', 'w') as f:
#     col = ",".join([col_name[0] for col_name in res.description]) + '\n'
#     f.write(col)

#     for i in res.fetchall():
#         values = ','.join(map(str,i)) + '\n'
#         f.write(values)

create_tbl = '''create table student ('''

insert_tbl = '''insert into student values ('''

with open('student.csv') as f:
    columns = f.readline().replace('\n', '').split(',')

    for col in columns:
        create_tbl += col + ' varchar(255),'
    create_tbl = create_tbl[:-1] + ')'
    
    
    values = f.readlines()
    print(values)
    
    for val in values:
        # val = '(' + val[:-2].split(',') + '''),'''
        val = tuple(val[:-2].split(','))
        print(val)
        # insert_tbl += val + ','
    print(insert_tbl)
        