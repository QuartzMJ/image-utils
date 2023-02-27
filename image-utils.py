import sys
import cv2
import numpy as np

def grayscale(input_path, output_path):
    # Read the image from the input path
    img = cv2.imread(input_path)
    # Convert the image to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Save the grayscale image
    cv2.imwrite(output_path, grayscale_img)
    print(f"Grayscale image saved to {output_path}")

def resize_image(input_path, output_path, new_width, new_height):
    # Read the image from the input path
    img = cv2.imread(input_path)
    # Resize the image
    new_size = (int(new_width), int(new_height))
    resized_img = cv2.resize(img, new_size)
    # Save the resized image
    cv2.imwrite(output_path, resized_img)
    print(f"Resized image saved to {output_path}")

def crop_image(input_path, output_path,width, height,x,y):
    # Read the image from the input path
    img = cv2.imread(input_path)
    # Crop the image using the given coordinates
    cropped_img = img[y:y+height, x:x+width]
    # Save the cropped image
    cv2.imwrite(output_path, cropped_img)
    print(f"Cropped image saved to {output_path}")


def gaussian_blur(input_path, output_path, kernel_size, sigma):
    # Read the image from the input path
    img = cv2.imread(input_path)
    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    # Save the blurred image
    cv2.imwrite(output_path, blurred_img)
    print(f"Blurred image saved to {output_path}")

def detect_edges(input_path, output_path, threshold1, threshold2):
    # Read the image from the input path
    img = cv2.imread(input_path)
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray_img, threshold1, threshold2)
    # Save the resulting image
    cv2.imwrite(output_path, edges)

def detect_lines(input_path, output_path, threshold=200):
    # Read the image from the input path
    img = cv2.imread(input_path)
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply edge detection using Canny
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    # Apply the Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    # Draw the detected lines on a new image
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
    # Save the resulting image
    cv2.imwrite(output_path, line_img)

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




