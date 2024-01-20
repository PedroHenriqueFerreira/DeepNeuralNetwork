from tkinter import Tk, Button, Canvas, Frame, Scale, Label

from tkinter.ttk import Style

import numpy as np
import pickle as pkl
from PIL import Image
from PIL.ImageTk import PhotoImage

from main import GAN

CANVAS_SIZE = 256
INPUT_ROWS = 10
INPUT_COLS = 10

class UI:
    def __init__(self):
        self.gan = pkl.load(open('./GAN_670.pkl', 'rb'))
        
        self.root = Tk()
        
        self.canvas_frame = Frame(self.root)
        
        self.noise_canvas = Canvas(self.canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bd=0)
        self.arrow = Label(self.canvas_frame, text='→', font=('Arial', 30, 'bold'))
        self.generator_canvas = Canvas(self.canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bd=0)
        
        self.noise_canvas.grid(row=0, column=0, padx=50, pady=10)
        self.arrow.grid(row=0, column=1)
        self.generator_canvas.grid(row=0, column=2, padx=50, pady=10)
        
        self.canvas_frame.pack(expand=True)
        
        self.sliders_frame = Frame(self.root)
        
        self.noise = np.random.randn(1, 100)
        
        self.sliders = self.create_sliders()
        self.update_sliders()
        
        self.sliders_frame.pack(expand=True, padx=50, pady=10)

        self.buttons_frame = Frame(self.root)
        
        self.options_frame = Frame(self.buttons_frame)
        
        random_button = Button(self.options_frame, text='RANDOMIZE', command=self.randomize, bd=0, bg='#fff')
        variation_button = Button(self.options_frame, text='VARIATION', command=self.variate, bd=0, bg='#fff')
        
        generate_button = Button(self.buttons_frame, text='GENERATE', command=self.generate, bd=0, bg='#fff')
    
        random_button.grid(row=0, column=0)
        variation_button.grid(row=0, column=1)
        
        self.options_frame.grid(row=0, column=0, padx=(0, 100))
        
        generate_button.grid(row=0, column=1, padx=(100, 0))
        
        self.buttons_frame.pack(expand=True, padx=50, pady=10)
        
        self.generate()
        
        Style(self.root).theme_use('clam')
        
        self.root.mainloop()
        
    def create_sliders(self):
        sliders = []
        
        for i in range(INPUT_ROWS):
            for j in range(INPUT_COLS):
                
                slider = Scale(
                    self.sliders_frame, 
                    from_=-5, 
                    to=5, 
                    orient='horizontal', 
                    resolution=0.02, 
                    length=100,
                    command=self.update_noise,
                    width=5,
                    bd=0,
                    font=('Arial', 10, 'bold')
                )
                
                slider.grid(row=i, column=j, pady=5)
                
                sliders.append(slider)
                
        return sliders

    def update_sliders(self):
        for slider, value in zip(self.sliders, self.noise[0]):
            slider.set(value)

    def update_noise(self, *args):
        for i in range(INPUT_ROWS * INPUT_COLS):
            self.noise[0][i] = self.sliders[i].get()    
            
        image = (self.noise / 5 * 127.5 + 127.5).astype(np.uint8)
        image = image.reshape((10, 10))
        image = Image.fromarray(image, mode='L')
        image = image.resize((CANVAS_SIZE, CANVAS_SIZE), Image.NEAREST) # type: ignore
        
        photo_image = PhotoImage(image)
        
        self.noise_canvas.delete('all')
        
        self.noise_canvas.create_image(0, 0, image=photo_image, anchor='nw')
        self.noise_canvas.image = photo_image # type: ignore

    def generate(self):
        images = self.gan.generator.forward(self.noise, False)
        avaliations = self.gan.discriminator.forward(images, False)
        
        image = (images[0] * 127.5 + 127.5).astype(np.uint8)
        image = image.transpose((1, 2, 0))
        image = Image.fromarray(image, mode='RGB')
        image = image.resize((CANVAS_SIZE, CANVAS_SIZE))
        
        print(avaliations.mean())

        photo_image = PhotoImage(image)
        
        self.generator_canvas.delete('all')
        
        self.generator_canvas.create_image(0, 0, image=photo_image, anchor='nw')
        self.generator_canvas.image = photo_image # type: ignore
        
    def variate(self):
        for i in range(INPUT_ROWS * INPUT_COLS):
            # if np.random.uniform(0, 1) >= 0.25:
            #     continue
            
            self.noise[0][i] += np.random.randn() * 0.2
            
        self.update_sliders()
        
    def randomize(self):
        self.noise = np.random.randn(1, 100)
        self.update_sliders()
UI()