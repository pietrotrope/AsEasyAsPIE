import numpy as np
import shutil
import os

path_source = "path1"
path_destination = "path2"


dirs = []
for file in os.listdir(path_source):
    d = os.path.join(path_source, file)
    n = os.path.join(path_destination, file)

    if not os.path.isdir(n):
        os.makedirs(n)

    if os.path.isdir(d):
        dirs.append(file)

for dir_f in dirs:
    if "IMDB" in dir_f:

        files = os.listdir(path_source+"/"+dir_f)

        for file_to_check in files:

            p = path_source+dir_f+"/"+file_to_check
            out_file = path_destination+dir_f+"/"+file_to_check[:-4]

            if not os.path.isfile(out_file) and not os.path.isfile(out_file+".npy"):
                if "csv" in file_to_check:

                    tmp = np.genfromtxt(p, delimiter=' ')
                    np.save(out_file, tmp)

                else:
                    shutil.copy2(p, out_file)
        print(dir_f)
