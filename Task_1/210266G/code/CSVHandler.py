import numpy as np
import pandas as pd

# Save and Load CSV Files
class CSVHandler:
    @staticmethod
    def write_to_csv(file_path, *arrays):
        with open(file_path, 'w') as fid:
            for array in arrays:
                for row in array:
                    np.savetxt(fid, [np.array(row).astype(np.float32)], delimiter=',', fmt='%0.16f', newline='\n')