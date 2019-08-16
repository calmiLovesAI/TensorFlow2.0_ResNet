import random


a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
a_length = len(a)
random.shuffle(a)
print(a[:6])
print(a[6:6+2])
print(a[6+2:])