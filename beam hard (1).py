#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# In[30]:


img = cv2.imread(r'C:\Users\abely\OneDrive\Desktop\tomography\tom_snimki\3660.20-3660.95_01004.bmp')
#img = cv2.imread(r'C:\Users\abely\OneDrive\Desktop\tomography\tom_snimki\3660.20-3660.95_01011.bmp')

# Создаём копию изначального изображения
img_cont = img.copy()

# Переводим изначальное изображение img в серый канал (с этим методом
# мы познакомились выше) и сохраняем в переменной gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Производим изменение размерности в 4 раза (изменение размерности производится
# в пикселях) относительно изначальной картинки img и сохраняем полученное 
# изображение в переменной img_resize
img_resize = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))

# Далее идёт большой блок кода, в котором мы создаём 
# алгоритм детектирования краёв Canny Edge Detector (с этим методом
# мы познакомились выше) 
canny_1 = 200
canny_2 = 225
canny = cv2.Canny(img, canny_1, canny_2)
contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
min_black = 255
cnt_black = []

for cnt in contours:
    c_area = cv2.contourArea(cnt) + 1e-7
    if cv2.contourArea(cnt) + 1e-7 > 500:
        cv2.drawContours(img_cont, [cnt], -1, 3)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, (255,255,255), -1)
        temp_mask = cv2.bitwise_and(gray, mask)
        temp_col = np.sum(temp_mask).real/(cv2.contourArea(cnt)+1e-7)
        if (temp_col < min_black) or (len(cnt_black) == 0):
            cnt_black = cnt
            min_black = temp_col

if len(cnt_black)!=0:
    cv2.drawContours(img_cont, [cnt_black], -1, (0,0,255), 3)


# In[1]:


import cv2

# Считываем изображение
image = cv2.imread(r'C:\Users\abely\OneDrive\Desktop\tomography\tom_snimki\3660.20-3660.95_01004.bmp')

# Отображаем изображение
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


img = cv2.imread(r'C:\Users\abely\OneDrive\Desktop\tomography\tom_snimki\3660.20-3660.95_01004.bmp')
#img = cv2.imread(r'C:\Users\abely\OneDrive\Desktop\tomography\tom_snimki\3660.20-3660.95_01011.bmp')

# Создаём копию изначального изображения
img_cont = img.copy()

# Переводим изначальное изображение img в серый канал (с этим методом
# мы познакомились выше) и сохраняем в переменной gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Производим изменение размерности в 4 раза (изменение размерности производится
# в пикселях) относительно изначальной картинки img и сохраняем полученное 
# изображение в переменной img_resize
img_resize = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))

# Далее идёт большой блок кода, в котором мы создаём 
# алгоритм детектирования краёв Canny Edge Detector (с этим методом
# мы познакомились выше) 
canny_1 = 200
canny_2 = 225
canny = cv2.Canny(img, canny_1, canny_2)
contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
min_black = 255
cnt_black = []

for cnt in contours:
    c_area = cv2.contourArea(cnt) + 1e-7
    if cv2.contourArea(cnt) + 1e-7 > 500:
        cv2.drawContours(img_cont, [cnt], -1, 3)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, (255,255,255), -1)
        temp_mask = cv2.bitwise_and(gray, mask)
        temp_col = np.sum(temp_mask).real/(cv2.contourArea(cnt)+1e-7)
        if (temp_col < min_black) or (len(cnt_black) == 0):
            cnt_black = cnt
            min_black = temp_col

if len(cnt_black)!=0:
    cv2.drawContours(img_cont, [cnt_black], -1, (0,0,255), 3)


# In[31]:


# Сохранение результатов работы нашей программы в папку


# Сохранение изображения с детектированными контурами
cv2.imwrite('img_contour_1_1.jpg', img_cont)

# Сохранение изображения в сером цветовом канале
#cv2.imwrite('img_gray_channel_1_1.png', gray)

# Сохранение уменьшенного в 4 раза изображения 
#cv2.imwrite('img_resize_1_1.png', img_resize)

# Вывод на экран изначального изображения
cv2.imshow('Basic image', img)

# Вывод на экран изображения с детектированными контурами
cv2.imshow('Contour image', img_cont)

# Вывод на экран изображения в сером цветовом канале
#cv2.imshow('Gray channel image', gray)

# Вывод на экран уменьшенного в 4 раза изображения 
#cv2.imshow('Resize image', img_resize)


# Режим ожидания нажатия кнопки
cv2.waitKey(0)


# In[26]:


import cv2
import numpy as np
# Считываем изображение
image = cv2.imread(r'C:\Users\abely\OneDrive\Desktop\tomography\tom_snimki\3660.20-3660.95_01004.bmp')


# Преобразуем изображение в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применяем бинаризацию для выделения контура
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Ищем контуры в изображении
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Выбираем самый большой контур
contour = max(contours, key=cv2.contourArea)

# Создаем маску с нулями, такой же формы, как и исходное изображение
mask = np.zeros_like(image)

# Рисуем контур на маске
cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)

# Применяем битовую маску для обрезки изображения по контуру
cropped_image = cv2.bitwise_and(image, mask)

# Отображаем изображение и обрезанное изображение
cv2.imshow('Image', image)
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Вычисляем гистограмму яркости
histogram = cv2.calcHist([cropped_image], [0], threshold, [256], [0, 256])
#histogram = cv2.calcHist([cropped_image],[0],None,[256],[0,256])

# Нормализуем гистограмму
histogram_normalized = histogram / histogram.sum()

# Создаем массив значений яркости
brightness = np.arange(0, 256)

# Построение графика распределения яркости
plt.plot(brightness, histogram_normalized, color='black')
plt.xlabel('Значение яркости')
plt.ylabel('Частота')
plt.title('Распределение яркости')
plt.xlim([0, 255])
plt.ylim([0, 1])
plt.show()



# In[28]:


import cv2
import numpy as np
import matplotlib.pyplot as plt



# Рассчитываем расстояние каждого пикселя до центра изображения
h, w = img.shape[:2]
center = (w // 2, h // 2)
y, x = np.indices((h, w))
radii = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

# Рассчитываем среднюю яркость для каждого радиуса
unique_radii = np.unique(radii.astype(int))
brightness = []
for radius in unique_radii:
    mask = radii.astype(int) == radius
    average_brightness = np.mean(img[mask])
    brightness.append(average_brightness)

# Построение графика распределения яркости от радиуса
plt.plot(unique_radii, brightness, color='black')
plt.xlabel('Радиус')
plt.ylabel('Средняя яркость')
plt.title('Распределение яркости от радиуса')
plt.show()


# In[25]:


unique_radii


# In[41]:


import cv2
import numpy as np
import matplotlib.pyplot as plt



# Преобразуем изображение в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Применяем детектор кругов Хафа
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                           param1=50, param2=30, minRadius=0, maxRadius=0)

if circles is not None:
    # Преобразуем координаты и радиусы окружностей в целочисленные значения
    circles = np.round(circles[0, :]).astype(int)

    # Находим окружность с максимальным радиусом
    max_radius = 0
    max_circle = None
    for circle in circles:
        radius = circle[2]
        if radius > max_radius:
            max_radius = radius
            max_circle = circle

    if max_circle is not None:
        # Извлекаем координаты и радиус максимальной окружности
        center_x, center_y = max_circle[0], max_circle[1]
        radius = max_circle[2]

        # Рисуем окружность на изображении
        cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), 2)

        # Выводим информацию о центре и радиусе окружности
        print('Центр окружности: x =', center_x, ', y =', center_y)
        print('Радиус окружности:', radius)
    else:
        print('На изображении не обнаружено окружностей')
else:
    print('На изображении не обнаружено окружностей')

# Отображаем изображение с нарисованной окружностью
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[37]:


import cv2
import numpy as np
import matplotlib.pyplot as plt



# Рассчитываем расстояние каждого пикселя до центра изображения
h, w = cropped_image.shape[:2]
center = (w // 2, h // 2)
y, x = np.indices((h, w))
radii = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

# Рассчитываем среднюю яркость для каждого радиуса
unique_radii = np.unique(radii.astype(int))
brightness = []
for radius in unique_radii:
    mask = radii.astype(int) == radius
    average_brightness = np.mean(cropped_image[mask])
    brightness.append(average_brightness)

    
# Построение графика распределения яркости от радиуса
plt.plot(unique_radii, brightness, color='black')
plt.xlabel('Радиус')
plt.ylabel('Средняя яркость')
plt.title('Распределение яркости от радиуса')
plt.show()



# In[5]:


import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
img = cv2.imread(r'C:\Users\abely\OneDrive\Desktop\tomography\tom_snimki\3660.20-3660.95_01004.bmp',0) #open image
hist,bins = np.histogram(cropped_image.flatten(), 256, [0, 256]) #histogram
plt.hist(img.flatten(), 256, [0,256], color = 'r')
plt.xlim([0, 256])
plt.show()
hist,bins = np.histogram(cropped_image.flatten(), 256, [0, 256]) #histogram
plt.hist(cropped_image.flatten(), 256, [0,256], color = 'r')
plt.xlim([0, 256])
plt.show()
sns.distplot(cropped_image, hist=True, kde=False, 
             bins=int(180/5), color = 'blue',
             hist_kws={'edgecolor':'black'})


# In[ ]:





# In[8]:


cv2.imshow('Contour image', threshold)


# In[ ]:




