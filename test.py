from tkinter import Tk, Button, Canvas

import pickle as pkl

import numpy as np

from main import GAN

with open('./GAN.pkl', 'rb') as file:
    gan = pkl.load(file)

root = Tk()

CANVAS_SIZE = 128

canvas = Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE)

def generate():
    noise = np.random.randn(1, 100)
    
    image = gan.generator.forward(noise, False)[0]
    image = (image * 127.5 + 127.5).astype(np.uint8)
    image = image.transpose((1, 2, 0))
    
    size = CANVAS_SIZE // image.shape[0]
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            i0 = i * size
            j0 = j * size
            i1 = i0 + size
            j1 = j0 + size
            
            r = f'{image[j][i][0]:02x}'
            g = f'{image[j][i][1]:02x}'
            b = f'{image[j][i][2]:02x}'
            
            rgb = f'#{r}{g}{b}'
            
            canvas.create_rectangle(i0, j0, i1, j1, fill=rgb, width=0)


button = Button(root, text='Generate', command=generate)

canvas.pack(expand=True)
button.pack(expand=True)

root.mainloop()