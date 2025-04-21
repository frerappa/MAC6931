
import torch
from torchvision import datasets, transforms
import numpy as np

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
test_X, test_y = next(iter(test_loader))

test_X = test_X.squeeze().numpy()  
test_y = test_y.numpy()           

print('X_test:', test_X.shape)
print('Y_test:', test_y.shape)

N = test_X.shape[0]

in_dat = ''
out_dat = ''
for i in range(N):
    label = test_y[i]
    pixels = test_X[i]  
    for row in pixels:
        in_dat += ' '.join(str(int(pixel * 255)) for pixel in row) + '\n'
    in_dat += '\n'
    out_dat += str(label) + '\n'

with open('../data/in.dat', 'w') as f:
    f.write(in_dat)
    print('Written ' + str(N) + ' images in ../data/in.dat')

with open('../data/out.dat', 'w') as f:
    f.write(out_dat)
    print('Written ' + str(N) + ' labels in ../data/in.dat')
