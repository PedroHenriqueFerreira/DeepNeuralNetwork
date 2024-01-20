from tkinter import Tk, Button, Canvas, Frame, Scale, Label

import numpy as np
import pickle as pkl

from PIL import Image
from PIL.ImageTk import PhotoImage

from main import GAN

CANVAS_SIZE = 256
INPUT_ROWS = 10
INPUT_COLS = 10

class UI(Tk):
    def __init__(self):
        Tk.__init__(self)
        
        self.noise = np.zeros((1, 100))
        
        self.gan = pkl.load(open('./model/GAN.pkl', 'rb'))
        
        self.title('GAN')
        
        self.configure(bg='#222')
        
        self.option_add('*Background', '#222')
        self.option_add('*Foreground', '#FFF')
        self.option_add('*Font', ('Arial', 10, 'bold'))
        self.option_add('*HighlightThickness', 0)
        self.option_add('*BorderWidth', 0)
        
        self.option_add('*Button.Background', '#3C80EC')
        self.option_add('*Button.DisabledForeground', '#5EA2FE')
        
        self.option_add('*Label.Font', ('Arial', 30, 'bold'))
                        
        canvas_frame = Frame(self)
        
        self.noise_canvas = Canvas(canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.noise_canvas.grid(row=0, column=0, padx=50, pady=10)
        
        arrow_label = Label(canvas_frame, text='→')
        arrow_label.grid(row=0, column=1)
        
        self.generator_canvas = Canvas(canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.generator_canvas.grid(row=0, column=2, padx=50, pady=10)
        
        canvas_frame.pack(expand=True)
        
        sliders_frame = Frame(self)
        
        self.sliders = self.create_sliders(sliders_frame)
        
        sliders_frame.pack(expand=True, padx=50, pady=10)

        buttons_frame = Frame(self)
        
        variate_button = Button(buttons_frame, text='VARIATION', command=self.variate)
        new_button = Button(buttons_frame, text='NEW', command=self.new)
        self.generate_button = Button(buttons_frame, text='GENERATE', command=self.generate, state='disabled')
    
        variate_button.grid(row=0, column=0, padx=2)
        new_button.grid(row=0, column=1, padx=2)
        self.generate_button.grid(row=0, column=2, padx=(390, 0))
        
        buttons_frame.pack(expand=True, padx=50, pady=10)
        
        self.generate()
        
    def create_sliders(self, parent):
        sliders = []
        
        for i in range(INPUT_ROWS):
            for j in range(INPUT_COLS):
                
                slider = Scale(
                    parent, 
                    from_=-5, 
                    to=5, 
                    resolution=0.01, 
                    length=100,
                    width=5,
                    sliderlength=20,
                    orient='horizontal', 
                    command=self.update_noise,
                    troughcolor='#444',
                )
                
                slider.set(np.random.randn())
                
                slider.grid(row=i, column=j, pady=5, padx=2)
                
                sliders.append(slider)
                
        return sliders

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
        self.generate_button.config(state='disabled')
        
        images = self.gan.generator.forward(self.noise, False)
        
        image = (images[0] * 127.5 + 127.5).astype(np.uint8)
        image = image.transpose((1, 2, 0))
        image = Image.fromarray(image, mode='RGB')
        image = image.resize((CANVAS_SIZE, CANVAS_SIZE))

        photo_image = PhotoImage(image)
        
        self.generator_canvas.delete('all')
        
        self.generator_canvas.create_image(0, 0, image=photo_image, anchor='nw')
        self.generator_canvas.image = photo_image # type: ignore
        
    def variate(self):
        self.generate_button.config(state='normal')
        
        np.random.choice(self.sliders).set(np.random.randn())
        
    def new(self):
        self.generate_button.config(state='normal')
        
        for slider in self.sliders:
            slider.set(np.random.randn())
            
tk = UI()
tk.mainloop()