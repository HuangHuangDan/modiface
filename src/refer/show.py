import numpy as np
 
file_path = '/Users/torryy/Documents/modiface/src/data/mtcnn_weights.npy'
data = np.load(file_path, allow_pickle=True)

print(data)