import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import sys
import cv2
import numpy as np
import random
import math

def select_image():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename()
    global inputfilepath
    global caption_left
    global image

    if(image != None):
        image.image = None
    if(caption_left != None):
        caption_left.place_forget()

    
    inputfilepath = file_path
    image_file = Image.open(file_path)
    tk_image = ImageTk.PhotoImage(image_file)
    
    image = tk.Label(root, image=tk_image)
    image.place(x=100,y=100)
   
    caption_left = tk.Label(root, text="Original")
    caption_width = caption_left.winfo_reqwidth();
    image_width = tk_image.width()
    x_pos = (image_width - caption_width)/2
    caption_left.place(in_=image,x=x_pos,y = tk_image.height() - 10,anchor='s')
    image.image = tk_image

def grayscale(input_path, output_path):
    img = cv2.imread(input_path)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path, grayscale_img)
    print(f"Grayscale image saved to {output_path}")



def resize_popup(inputfilepath, outputfilepath):
    popup = tk.Toplevel(root)
    popup.title("Input Parameters")

    popup.geometry("300x400")
    # Create the input fields
    label1 = tk.Label(popup, text="Width:")
    label1.grid(row=0, column=0, padx=10, pady=10)
    entry1 = tk.Entry(popup)
    entry1.grid(row=0, column=1, padx=10, pady=10)

    label2 = tk.Label(popup, text="Height:")
    label2.grid(row=1, column=0, padx=10, pady=10)
    entry2 = tk.Entry(popup)
    entry2.grid(row=1, column=1, padx=10, pady=10)

    def resize_image(input_path, output_path, new_width, new_height):
        img = cv2.imread(input_path)
        new_size = (int(new_width), int(new_height))
        resized_img = cv2.resize(img, new_size)
        cv2.imwrite(output_path, resized_img)
        print(f"Resized image saved to {output_path}")
        popup.destroy()

    # Create a button to submit the parameters
    submit_button = tk.Button(popup, text="Submit", command=lambda: resize_image(inputfilepath,outputfilepath, entry1.get(), entry2.get()))
    submit_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
    popup.wait_window()
    



def  crop_popup(inputfilepath,outputfilepath):
    popup = tk.Toplevel(root)
    popup.title("Input Parameters")

    popup.geometry("300x400")
    # Create the input fields
    label1 = tk.Label(popup, text="Width:")
    label1.grid(row=0, column=0, padx=10, pady=10)
    entry1 = tk.Entry(popup)
    entry1.grid(row=0, column=1, padx=10, pady=10)

    label2 = tk.Label(popup, text="Height:")
    label2.grid(row=1, column=0, padx=10, pady=10)
    entry2 = tk.Entry(popup)
    entry2.grid(row=1, column=1, padx=10, pady=10)

    label3 = tk.Label(popup, text="Starting X:")
    label3.grid(row=2, column=0, padx=10, pady=10)
    entry3 = tk.Entry(popup)
    entry3.grid(row=2, column=1, padx=10, pady=10)

    label4 = tk.Label(popup, text="Starting Y:")
    label4.grid(row=3, column=0, padx=10, pady=10)
    entry4 = tk.Entry(popup)
    entry4.grid(row=3, column=1, padx=10, pady=10)

    def crop_image(input_path, output_path,width, height,x,y):
        img = cv2.imread(input_path)
        cropped_img = img[y:y+height, x:x+width]
        cv2.imwrite(output_path, cropped_img)
        print(f"Cropped image saved to {output_path}")
        popup.destroy();

    submit_button = tk.Button(popup, text="Submit", command=lambda: crop_image(inputfilepath,outputfilepath, int(entry1.get()), int(entry2.get()),int(entry3.get()), int(entry4.get())))
    submit_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)
    popup.wait_window()

def blur_popup(inputfilepath, outputfilepath):
    popup = tk.Toplevel(root)
    popup.title("Input Parameters")

    popup.geometry("300x400")
    # Create the input fields
    label1 = tk.Label(popup, text="Kernel size:")
    label1.grid(row=0, column=0, padx=10, pady=10)
    entry1 = tk.Entry(popup)
    entry1.grid(row=0, column=1, padx=10, pady=10)

    label2 = tk.Label(popup, text="Sigma:")
    label2.grid(row=1, column=0, padx=10, pady=10)
    entry2 = tk.Entry(popup)
    entry2.grid(row=1, column=1, padx=10, pady=10)

    def gaussian_blur(input_path, output_path, kernel_size, sigma):
        img = cv2.imread(input_path)
        blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        cv2.imwrite(output_path, blurred_img)
        print(f"Blurred image saved to {output_path}")
        popup.destroy()

    # Create a button to submit the parameters
    submit_button = tk.Button(popup, text="Submit", command=lambda: gaussian_blur(inputfilepath,outputfilepath, int(entry1.get()), int(entry2.get())))
    submit_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
    popup.wait_window()

def detect_edges_popup(inputfilepath, outputfilepath):
    popup = tk.Toplevel(root)
    popup.title("Input in ratio 1:3")

    popup.geometry("300x400")
    # Create the input fields
    label1 = tk.Label(popup, text="Threshold low:")
    label1.grid(row=0, column=0, padx=10, pady=10)
    entry1 = tk.Entry(popup)
    entry1.grid(row=0, column=1, padx=10, pady=10)

    label2 = tk.Label(popup, text="Threshold high:")
    label2.grid(row=1, column=0, padx=10, pady=10)
    entry2 = tk.Entry(popup)
    entry2.grid(row=1, column=1, padx=10, pady=10)

    def detect_edges(input_path, output_path, lowwer_threshold, upper_threshold):
        img = cv2.imread(input_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.blur(gray_img, (3,3))
        edges = cv2.Canny(blur_img, lowwer_threshold, upper_threshold)
        cv2.imwrite(output_path, edges)
        popup.destroy()

    # Create a button to submit the parameters
    submit_button = tk.Button(popup, text="Submit", command=lambda: detect_edges(inputfilepath,outputfilepath, int(entry1.get()), int(entry2.get())))
    submit_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
    popup.wait_window()

def detect_lines_popup(inputfilepath, outputfilepath):
    popup = tk.Toplevel(root)
    popup.title("Input Parameters")

    popup.geometry("300x400")
    # Create the input fields
    label1 = tk.Label(popup, text="Threshold:")
    label1.grid(row=0, column=0, padx=10, pady=10)
    entry1 = tk.Entry(popup)
    entry1.grid(row=0, column=1, padx=10, pady=10)


    def detect_lines(input_path, output_path, threshold):
        img = cv2.imread(input_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.blur(gray_img, (3,3))
        edges = cv2.Canny(gray_img, 50, 150,None, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold,None,0,0)
        line_img = np.zeros_like(img)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(line_img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        cv2.imwrite(output_path, line_img)
        popup.destroy()

    # Create a button to submit the parameters
    submit_button = tk.Button(popup, text="Submit", command=lambda: detect_lines(inputfilepath,outputfilepath, int(entry1.get())))
    submit_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
    popup.wait_window()

def random_rotate_image(input_path, output_path):
    img = cv2.imread(input_path)
    max_angle = random.randint(-360, 360)
    height, width = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    cv2.imwrite(output_path, rotated_img)
    print(f"Random rotation result saved to {output_path}")

def random_flip_image(input_path, output_path):
    img = cv2.imread(input_path)
    flip_code = random.randint(-1, 1)
    flipped_img = cv2.flip(img, flip_code)
    cv2.imwrite(output_path, flipped_img)
    print(f"Random flip result saved to {output_path}")

def random_brightness(input_path, output_path):
    img = cv2.imread(input_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness_adjustment = random.uniform(0.5, 1.5)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * brightness_adjustment, 0, 255)
    result_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, result_img)
    print(f"Random brightness adjustment result saved to {output_path}")

def random_noise(input_path, output_path):
    img = cv2.imread(input_path)
    noise_scale = random.uniform(0, 32)
    noise = np.random.normal(scale=noise_scale, size=img.shape)
    noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, noisy_img)
    print(f"Random noise addition result saved to {output_path}")

def random_crop(input_path, output_path):
    img = cv2.imread(input_path)
    height, width, _ = img.shape
    crop_size = random.randint(1, min(width, height))
    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    bottom = top + crop_size
    right = left + crop_size
    cropped_img = img[top:bottom, left:right]
    cv2.imwrite(output_path, cropped_img)
    print(f"Random cropped result saved to {output_path}")

def process_image(method):
    global processedImage
    global caption
    if(processedImage != None):
        print("Debugging label displaying....")
        processedImage.image = None
    if(caption != None):
        caption.place_forget()
    print(inputfilepath)
    outputfilepath = inputfilepath[:-4] + method + ".png"
    outputfilepath.replace(" ", "")
    print(inputfilepath,method,outputfilepath)

    caption_text = None;
    if(method == "Grayscale"):
        grayscale(inputfilepath,outputfilepath)
        caption_text = "Grayscaled"
    elif(method == "Resize"):
        resize_popup(inputfilepath,outputfilepath)
        caption_text = "Resized"
    elif(method == "Crop"):
        crop_popup(inputfilepath,outputfilepath)
        caption_text ="Cropped"
    elif(method == "Blur"):
        blur_popup(inputfilepath,outputfilepath)
        caption_text ="Blurred"
    elif(method == "Detect edges"):
        detect_edges_popup(inputfilepath,outputfilepath)
        caption_text ="Detected edges"
    elif(method == "Detect lines"):
        detect_lines_popup(inputfilepath,outputfilepath)
        caption_text = "Detected lines"
    elif(method == "Random rotate"):
        caption_text ="Random rotated"
        random_rotate_image(inputfilepath,outputfilepath)
    elif(method == "Random flip"):
        caption_text ="Random flipped"
        random_flip_image(inputfilepath,outputfilepath)
    elif(method == "Random brightness"):
        caption_text ="Random brightened"
        random_brightness(inputfilepath,outputfilepath)
    elif(method == "Random noise"):
        caption_text ="Random noised"
        random_noise(inputfilepath,outputfilepath)
    elif(method == "Random crop"):
        caption_text = "Random cropped"
        random_crop(inputfilepath,outputfilepath)

    
    image_file = Image.open(outputfilepath)
    tk_image = ImageTk.PhotoImage(image_file)
    
    # Create a label to display the image
    
    processedImage = tk.Label(root, image=tk_image)
    processedImage.place(relx=1.0, x=-100, y=100, anchor="ne")
    print("Debugging image display...")
    
    caption = tk.Label(root, text=caption_text)
    caption_width = caption.winfo_reqwidth();
    image_width = tk_image.width()
    x_pos = (image_width - caption_width)/2
    caption.place(in_=processedImage,x=x_pos,y = tk_image.height() - 10,anchor='s')
    processedImage.image = tk_image

root = tk.Tk()
inputfilepath = None
image = None
processedImage = None
caption = None
caption_left = None

button = tk.Button(root, text="Select Image", command=select_image)
button.place(x=50,y=50)

methods = ["Grayscale", "Resize", "Crop","Blur","Detect edges","Detect lines","Random rotate","Random flip","Random brightness","Random noise","Random crop"]

selected_method = tk.StringVar(root)
selected_method.set(methods[0])  

dropdown = tk.OptionMenu(root, selected_method, *methods,command=lambda value: process_image(selected_method.get()))
dropdown.place(relx=1.0, x=-50, y=50, anchor="ne")



root.geometry("1600x900")
root.mainloop()

