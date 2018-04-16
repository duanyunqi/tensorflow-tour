import os

flag = 0
for i in range(60000):
    for m in range(10):
        file_name = '{0}_{1}.jpg'.format(i, m)
        dir = os.path.join(os.getcwd(), 'AI-Practice-Tensorflow-Notes', 'fc4', 'mnist_data_jpg','mnist_train_jpg_60000', file_name)
        flag = flag + int(os.path.exists(dir))
    if(flag < 1):
        print(i)
        flag = 0
    flag = 0
