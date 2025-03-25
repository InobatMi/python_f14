import sqlite3

conn = sqlite3.connect('sample_db.sqlite')

cursor = conn.cursor()

crt_qr = 'create table employee (id int, name varchar(30), age int)'

insert_qr = '''insert into employee values (1, 'Alex', 26)'''
cursor.execute(insert_qr)

conn.commit()
cursor.close()
conn.close()

