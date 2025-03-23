import pyodbc

connection = pyodbc.connect( 
                               'DRIVER={SQL Server};'
                               'Server=DESKTOP-CSSCKLQ;'
                               'Database=master;'
                               'trusted_Connection=yes;',
                               autocommit=True
                               )

print('Connected Successfully')
cursor = connection.cursor()


# 1. Create a new database with a table named Roster that has 
# three fields: Name, Species, and Age. The Name and Species columns should be text fields, 
# and the Age column should be an integer field.

# creating database
createdb_qy = '''if not exists (select name FROM sys.databases where name = 'Roster')
                create database Roster'''
res = cursor.execute(createdb_qy)

connection.close()
connection = pyodbc.connect(
    'DRIVER={SQL Server};'
    'Server=DESKTOP-CSSCKLQ;'
    'Database=Roster;'  
    'trusted_Connection=yes;',
    autocommit=True
)
cursor = connection.cursor()

# creating table
createtable_qy = '''if Object_ID('dbo.Roster', 'U') is null
                    create table Roster (Name varchar(50), Species varchar(50), Age int)'''

try:
    res = cursor.execute(createtable_qy)
    connection.commit()
except pyodbc.Error as e:
    print(f'Error: {e}')

# 2.  Populate your new table with the following values:

# | Name            | Species      | Age |
# |-----------------|--------------|-----|
# | Benjamin Sisko  | Human        |  40 |
# | Jadzia Dax      | Trill        | 300 |
# | Kira Nerys      | Bajoran      |  29 |

insert_qy = '''insert into  Roster values ('Benjamin Sisko','Human', 40), 
                                    ('Jadzia Dax', 'Trill', 300),
                                    ('Kira Nerys', 'Bajoran',29)'''

try:
    res = cursor.execute(insert_qy)
    connection.commit()
except pyodbc.Error as e:
    print(f'Error: {e}')



# 3. Update the Name of Jadzia Dax to be Ezri Dax


update_qy = '''update Roster
                set Name = 'Ezri Dax' 
                where Name = 'Jadzia Dax' '''

try:
    res = cursor.execute(update_qy)
    connection.commit()
except pyodbc.Error as e:
    print(f'Error: {e}')


# 4.  Display the Name and Age of everyone in the table classified as Bajoran.

select_qy = '''select Name, Age from Roster
                where Species = 'Bajoran' '''

try:
    res = cursor.execute(select_qy)
    results = cursor.fetchall()
    for row in results:
        print(f"Name: {row[0]}, Age: {row[1]}")
except pyodbc.Error as e:
    print(f'Error: {e}')

cursor.close()
connection.close()