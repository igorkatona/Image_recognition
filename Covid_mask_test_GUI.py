###Image Recognition with Trained Model
# Import the libraries

import numpy as np
from keras.preprocessing import image
from pandas import notnull
import tensorflow as tf
from keras.models import load_model
import cv2 
# FOR GUI
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter.filedialog import *
from PIL import ImageTk, Image 
import os


model_for_loading = 'myModel_saved.h5'

def about_menu():
    hide_all_frames()
    about_frame.grid(row=0, column=0)
    my_label = tk.Label(about_frame, 
        text='This application uses various model to predict if person on picture is wearing a mask or not.',
        wraplength=300,
        justify="left")
    my_label.grid(row =0, column=0)
    
def cretor_menu():
    hide_all_frames()
    creator_frame.grid(row=0, column=0)
    my_label = tk.Label(creator_frame, 
        text='All credits for creating this application go to: ',
        wraplength=300,
        justify="left")
    my_label.grid(row =0, column=0, columnspan=2,sticky='nw')
    tk.Label(creator_frame, text='').grid(row=0, column=1 )
    tk.Label(creator_frame, text='').grid(row=1, column=0 )
    tk.Label(creator_frame, text='Igor Katona', justify="left" , anchor='nw').grid(sticky='nw', row=1, column=1)
    tk.Label(creator_frame, text='').grid(row=2, column=0)
    tk.Label(creator_frame, text='Rahul Krishnan', justify="left", anchor='nw').grid(sticky='nw', row=2, column=1)
    tk.Label(creator_frame, text='').grid(row=3, column=0)
    tk.Label(creator_frame, text='Amr Ibrahim', justify="left", anchor='nw').grid(sticky='nw',row=3, column=1)
    tk.Label(creator_frame, text='').grid(row=4, column=0)
    tk.Label(creator_frame, text='Al Shayma Aldossary', justify="left", anchor='nw').grid(sticky='nw',row=4, column=1)


def settings_menu():
    hide_all_frames()

    def select_model():

        file = askopenfile( mode = 'rb', title='Choose a file', filetypes=[("h5 Files", "*.h5")])
        global model_for_loading
        model_for_loading = os.path.abspath(file.name)
        #model_for_loading = file
        text_box.delete(1.0, tk.END)
        text_box.insert(tk.END, model_for_loading)
        

    settings_frame.grid(row=0, column=0)
    ttk.Label(settings_frame, text="Settings", justify="left", anchor='nw').grid(sticky='nw', row=0, column=0 )

    separator = ttk.Separator(settings_frame, orient='horizontal').grid(row=1, column=0 , ipadx=100, columnspan=6)
    ttk.Label(settings_frame, text="Saved Model: ", anchor='nw').grid(sticky='nw', row=2, column=0 )

    text_box = tk.Text(settings_frame, height = 1, width = 20)
    text_box.grid(row=2, column=1)
    text_box.insert(tk.END, model_for_loading)
    button = tk.Button(master=settings_frame, text='Select Model', command=select_model)
    button.grid(row=3, column=0)




def open_image_menu():
    hide_all_frames()
    open_image_frame.grid(row=0, column=0)
    global my_image
    global my_image_label
    file = askopenfile( mode = 'rb', title='Choose a file', filetypes=[("Jpg Files", "*.jpg")])
    my_image = Image.open(file)
    my_image_location = os.path.abspath(file.name)
    width, height = my_image.size 
    if width + height > 650:
        new_d_width = int
        new_d_width = width / 300
        new_width = int(width /new_d_width)
        new_height = int(height /new_d_width)
        my_image= my_image.resize((new_width, new_height), Image.ANTIALIAS)
    my_image = ImageTk.PhotoImage(my_image)
    my_image_label = tk.Label(open_image_frame,image=my_image)
    my_image_label.grid(column=0, row=1)
    def call_back():
        output_class,score = function_predict(my_image_location)
        path, filename = os.path.split(model_for_loading)
        return_text = "Predicted class is: " + str(output_class) + "\nPrediction score is: " + str(score) + "\nPrediction model: " + str(filename)
        return return_text

    tk.Label(open_image_frame, text=call_back(), justify="left", anchor='nw').grid(sticky='nw', row=3, column=0 , columnspan=3)

        
def hide_all_frames():
    for w in open_image_frame.winfo_children():
        w.destroy()
    about_frame.grid_forget()
    creator_frame.grid_forget()
    settings_frame.grid_forget()
    open_image_frame.grid_forget()




window = tk.Tk()

window.title('Image recognition')
window.geometry('300x300')


# Creating menubar
# Creating File inside menubar
menubar = tk.Menu(window)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Open Image", command= open_image_menu)
#filemenu.add_command(label="Open Folder", command=donothing)
#filemenu.add_command(label="Save", command=donothing)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=window.quit)
menubar.add_cascade(label="File", menu=filemenu)
# Creating Settings inside menubar
settingmenu = tk.Menu(menubar, tearoff=0)
settingmenu.add_command(label="Settings", command=settings_menu)
menubar.add_cascade(label="Settings", menu=settingmenu)
# Creating Help inside menubar
helpmenu = tk.Menu(settingmenu, tearoff=0)
helpmenu.add_command(label="Credits", command=cretor_menu)
helpmenu.add_command(label="About...", command=about_menu)
menubar.add_cascade(label="Help", menu=helpmenu)

window.config(menu=menubar)

# Creating frames

about_frame = tk.Frame(window, width=300, height=300)
creator_frame = tk.Frame(window, width=300, height=300)
settings_frame = tk.Frame(window, width=300, height=300)
open_image_frame = tk.Frame(window, width=300, height=300)


def function_predict(argImage):

    #model_for_loading ='myModel_saved.h5'
    resnet_model = load_model(model_for_loading)

    img_height, img_width = 180, 180
    class_names = ['with_mask', 'without_mask']

    imageSize=[img_height,img_width]
    imageLocation = argImage
    print(argImage)
    predictImage = tf.keras.preprocessing.image.load_img(
        imageLocation, 
        target_size=imageSize
    )
    img_array = tf.keras.preprocessing.image.img_to_array(predictImage)
    # Expading dims again to apropriate for modelo
    img_array = tf.expand_dims(img_array, 0)

    # Making a prediction and dividing it with 255.0 to be able to represent likelihood in percentages
    predictions = resnet_model.predict(img_array)

    score = predictions[0]

    score = max(score)
    
    output_class=class_names[np.argmax(predictions)]
    print("The predicted class is", output_class)
    return output_class, score

window.mainloop()