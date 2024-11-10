import tarfile

with tarfile.open("./Data/OIBC_2024_DATA.gz", 'r:gz') as tr:
    tr.extractall(path="./Data/OIBC_2024_DATA")
    
print("done")