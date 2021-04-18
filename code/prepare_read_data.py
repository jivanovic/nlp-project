import numpy as np
from read_data import read_messages, read_classes, prepare_data

#messages = read_messages()
#classes = read_classes()
data = prepare_data()

a = np.array(data)
np.savetxt('data_mapbook.csv', a, fmt='%s', delimiter=',')
