import pathlib

filename = 'input-coffee.txt'
path = pathlib.Path('data')/'coffee'/filename

f = open(path, 'r')
file_contents = f.read()
print (file_contents)
f.close()