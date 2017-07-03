from random import randint

training_set = []

for _ in range(1000):
    x = randint(1, 9)
    y = randint(1, 9)
    training_set.append([x, y, x * y])

print training_set