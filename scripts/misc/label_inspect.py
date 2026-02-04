import numpy as np

Y_label = np.load('dataset/Y_label.npy')


print('   idx: ', end='')
for idx in range(len(Y_label)):
    print(f'{idx:5}\t', end='')
print('\ncnt -1: ', end='')
for cnt in np.sum(Y_label == -1, axis=0):
    print(f'{cnt:5}\t', end='')
print('\ncnt  0: ', end='')
for cnt in np.sum(Y_label == 0, axis=0):
    print(f'{cnt:5}\t', end='')
print('\ncnt  1: ', end='')
for cnt in np.sum(Y_label == 1, axis=0):
    print(f'{cnt:5}\t', end='')
print('\n')
