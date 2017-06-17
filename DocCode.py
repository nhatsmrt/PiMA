# Explain line by line what does each line do.
# Answer question (the questions are about the lines directly below them)
import numpy as np
from random import random


class Randomizer():
    # Why self?
    def __init__(self, size = 10):
        self.size = size

    def randomize(self):
        ret = np.array([])
        ret = np.append(ret, [random() for i in range(self.size)])
        return ret

    @staticmethod
    # Why no self?
    def get_random_number():
        return random()



randomizer = Randomizer(15)

# What does this line of code do?
print(randomizer.randomize())

a = np.array([1, 2, 3])
b = np.array([[1, 2, 3],[1, 4, 5],[1, 2, 6]])

print(np.matmul(b, a))

for i, X in enumerate(a):
    print(a[i])
# What does this code do? Explain?
print(b[1 : : , : : -2 ])

print(sum(filter(lambda x : x%3 == 0, [1, 2, 3, 4, 5, 6])))

