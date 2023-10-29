import cv2
import os
import numpy as np

input_folder = r"C:\Users\abely\OneDrive\Desktop\tomography\tom_snimki"
output_folder = r"C:\Users\abely\OneDrive\Desktop\tomography\tom_snimki_id"

# Функция для нахождения ROI на изображении с помощью неоднородного серого круга
def find_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        roi = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(roi)
        return (int(x), int(y), int(radius))
    return None

image_files = os.listdir(input_folder)

for file in image_files:
    image = cv2.imread(os.path.join(input_folder, file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = 0
            cY = 0
        
        roi = find_roi(image)
        if roi is not None:
            x, y, radius = roi
            roi_mean_brightness = np.mean(gray[y-int(radius/2):y+int(radius/2), x-int(radius/2):x+int(radius/2)])
            gray_color = int(roi_mean_brightness)
            
            s = 0
            e = 25
            width = 25
            height = 25
            # Вырезаем квадратную область изображения
            square = image[e:e+height, s:s+width, :]
            # Преобразуем в оттенки серого
            gray_black = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            # Вычисляем среднее значение яркости
            average_brightness = np.mean(gray_black)
            print("Average Brightness:", average_brightness)
            
            
            new_image = np.zeros_like(image)
            cv2.rectangle(new_image, (0, 0), (image.shape[1], image.shape[0]), (average_brightness, average_brightness, average_brightness), -1)
            cv2.circle(new_image, (x, y), int(radius), (gray_color, gray_color, gray_color), -1)
            cv2.imwrite(os.path.join(output_folder, file), new_image)
    
cv2.destroyAllWindows()