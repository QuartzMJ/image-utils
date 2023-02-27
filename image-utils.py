import sys
import cv2
import numpy as np
import random

def grayscale(input_path, output_path):
    # Read the image from the input path
    img = cv2.imread(input_path)
    # Convert the image to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Save the grayscale image
    cv2.imwrite(output_path, grayscale_img)
    print(f"Grayscale image saved to {output_path}")

def resize_image(input_path, output_path, new_width, new_height):
    img = cv2.imread(input_path)
    new_size = (int(new_width), int(new_height))
    resized_img = cv2.resize(img, new_size)
    cv2.imwrite(output_path, resized_img)
    print(f"Resized image saved to {output_path}")

def crop_image(input_path, output_path,width, height,x,y):
    img = cv2.imread(input_path)
    cropped_img = img[y:y+height, x:x+width]
    cv2.imwrite(output_path, cropped_img)
    print(f"Cropped image saved to {output_path}")


def gaussian_blur(input_path, output_path, kernel_size, sigma):
    img = cv2.imread(input_path)
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    cv2.imwrite(output_path, blurred_img)
    print(f"Blurred image saved to {output_path}")

def detect_edges(input_path, output_path, threshold1, threshold2):
    img = cv2.imread(input_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, threshold1, threshold2)
    cv2.imwrite(output_path, edges)

def random_rotate_image(input_path, output_path, max_angle):
    img = cv2.imread(input_path)
    height, width = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    cv2.imwrite(output_path, rotated_img)
    print(f"Random rotation result saved to {output_path}")


def detect_lines(input_path, output_path, threshold=200):
    img = cv2.imread(input_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(output_path, line_img)

def random_flip_image(input_path, output_path):
    # Read the image from the input path
    img = cv2.imread(input_path)
    # Choose a random value to determine whether to flip horizontally or vertically
    flip_code = random.randint(-1, 1)
    # Flip the image
    flipped_img = cv2.flip(img, flip_code)
    # Save the resulting image
    cv2.imwrite(output_path, flipped_img)
    print(f"Random flip result saved to {output_path}")

def random_brightness(input_path, output_path):
    # Read the image from the input path
    img = cv2.imread(input_path)
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Choose a random value to adjust the brightness
    brightness_adjustment = random.uniform(0.5, 1.5)
    # Apply the brightness adjustment to the value channel
    hsv[:,:,2] = np.clip(hsv[:,:,2] * brightness_adjustment, 0, 255)
    # Convert the image back to BGR color space
    result_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Save the resulting image
    cv2.imwrite(output_path, result_img)
    print(f"Random brightness adjustment result saved to {output_path}")

def random_noise(input_path, output_path):
    # Read the image from the input path
    img = cv2.imread(input_path)
    # Choose a random value to adjust the noise
    noise_scale = random.uniform(0, 255)
    # Generate random noise of the same size as the image
    noise = np.random.normal(scale=noise_scale, size=img.shape)
    # Add the noise to the image
    noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    # Save the resulting image
    cv2.imwrite(output_path, noisy_img)
    print(f"Random noise addition result saved to {output_path}")

def random_crop(input_path, output_path):
    # Read the image from the input path
    img = cv2.imread(input_path)
    # Get the dimensions of the image
    height, width, _ = img.shape
    # Generate random crop size
    crop_size = random.randint(1, min(width, height))
    # Generate random crop coordinates
    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    bottom = top + crop_size
    right = left + crop_size
    # Crop the image using the generated coordinates
    cropped_img = img[top:bottom, left:right]
    # Save the resulting image
    cv2.imwrite(output_path, cropped_img)
    print(f"Random cropped result saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python image-utils.py option [input_path] [output_path] ....")
    else:
        input_path = sys.argv[2]
        output_path = sys.argv[3]
        if(sys.argv[1] == "grayscale"):
            grayscale(input_path,output_path)
        elif(sys.argv[1] == "resize"):
            new_width = int(sys.argv[4])
            new_height = int(sys.argv[5])
            resize_image(input_path,output_path,new_width,new_height)
        elif(sys.argv[1] == "crop"):
            width = int(sys.argv[4])
            height = int(sys.argv[5])
            x = int(sys.argv[6])
            y = int(sys.argv[7])
            crop_image(input_path,output_path,width,height,x,y)
        elif(sys.argv[1] == "blur"):
            kernel_size = int(sys.argv[4])
            sigma = int(sys.argv[5])
            gaussian_blur(input_path,output_path,kernel_size,sigma)
        elif(sys.argv[1] == "detect"):
            threshold1 = int(sys.argv[4])
            threshold2 = int(sys.argv[5])
            detect_edges(input_path,output_path,threshold1,threshold2)
        elif(sys.argv[1] == "lines"):
            threshold = int(sys.argv[4])
            detect_lines(input_path,output_path,threshold)
        elif(sys.argv[1] == "rotate"):
            angle = int(sys.argv[4])
            random_rotate_image(input_path,output_path,angle)
        elif(sys.argv[1] == "flip"):
            random_flip_image(input_path,output_path)
        elif(sys.argv[1] == "brightness"):
            random_brightness(input_path,output_path)
        elif(sys.argv[1] == "noise"):
            random_noise(input_path,output_path)
        elif(sys.argv[1] == "random_crop"):
            random_crop(input_path,output_path)






