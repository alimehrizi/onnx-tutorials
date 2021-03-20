import torch 
import torchvision 


with open('/home/altex/fake.txt', 'w') as f:
    for i in range(1000):
        f.write(str(i))
        f.write('\n')