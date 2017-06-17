# Giải thích từng dòng
# Đồng thời, trả lời các câu hỏi trong comment (liên quan đến dòng ngay dưới)
import numpy as np
from random import random


class Randomizer():
    def __init__(self, size = 10):
        self.size = size

    def randomize(self):
        ret = np.array([])
        ret = np.append(ret, [random() for i in range(self.size)])
        return ret

    @staticmethod
    # Tại sao không có self? Static method là gì?
    def get_random_number():
        return random()



randomizer = Randomizer(15)

# Dòng này in gì và tại sao? (Chỉ cần nói ý tưởng, không cần nói đáp án cụ thể)
print(randomizer.randomize())

a = np.array([1, 2, 3])
b = np.array([[1, 2, 3],[1, 4, 5],[1, 2, 6]])

# Kết quả phép tính này là gì? Giải thích?
print(np.matmul(b, a))

# Ý nghĩa của dòng này?
for i, X in enumerate(a):
    print(a[i])

# Dòng này in gì và tại sao?
print(b[1 : : , : : -2 ])

# Dòng này in gì và tại sao?
print(sum(filter(lambda x : x%3 == 0, [1, 2, 3, 4, 5, 6])))

