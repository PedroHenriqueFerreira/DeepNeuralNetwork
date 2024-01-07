from tkinter import Tk, Button, Canvas

import pickle as pkl

import numpy as np

from main import GAN

gan = pkl.load(open('./GAN.pkl', 'rb'))

root = Tk()

canvas = Canvas(root, width=512, height=512)

def generate():
    noise = np.random.randn(1, 100)
    
    image = gan.generator.forward(noise, False)[0]
    image = (image * 127.5 + 127.5).astype(np.uint8)
    
    size = 512 // 128
    
    for i in range(128):
        for j in range(128):
            i0 = i * size
            j0 = j * size
            
            i1 = (i + 1) * size
            j1 = (j + 1) * size
            
            r = f'{image[0, i, j]:02x}'
            g = f'{image[1, i, j]:02x}'
            b = f'{image[2, i, j]:02x}'
            
            rgb = f'#{r}{g}{b}'
            
            canvas.create_rectangle(i0, j0, i1, j1, fill=rgb, width=0)

button = Button(root, text='Generate', command=generate)

canvas.pack()
button.pack(expand=True, fill='x')

root.mainloop()