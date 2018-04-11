from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from PIL import ImageDraw

mnist = input_data.read_data_sets('../data/', one_hot=True)

width = 28
height = 28

for n in range(10):
    m_img = mnist.train.images[n]
    print(mnist.train.labels[n])
    i = 0
    image = Image.new('L', (width, height), (100))
    draw = ImageDraw.Draw(image)
    for x in range(width):
        for y in range(height):
            draw.point((y, x), fill=(int((1-m_img[i])*255)))
            i = i + 1
    title = 'test{0}.jpg'.format(n)
    image.save(title, 'jpeg')