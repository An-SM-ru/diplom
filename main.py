#import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras import backend as K
from keras.initializers.initializers_v2 import RandomNormal
from keras.optimizers.optimizer_v2 import Adadelta, Adam, SGD, Adagrad, RMSprop, Ftrl, Adamax

from keras.models.model_v2 import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Flatten
from keras.layers import add, Convolution2D, concatenate, Input, MaxPooling2D, Conv2D
from keras.layers import Activation, LeakyReLU, Average, Maximum, Subtract, Multiply
from keras.layers import Conv2DTranspose, UpSampling2D
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import LambdaCallback
#from sklearn.model_selection import train_test_split
from PIL import Image
from IPython.core.display import clear_output
#from scipy.constants import hp

#import tensorflow_addons as tfa
from tensorflow_addons.layers.normalizations import InstanceNormalization


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import time
import os
import gc

import sys
#sys.path.append('D:\ASmirnov\AI\diplom\venv')
#import module_func
#from menpo.shape import bounding_box


def grafik(to_array, tip=''):
    # Функция для отображения истории после обучения модели
    # Вход: вектор историй обучения
    # выводятся только val_accuracy и val_loss
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(18, 3)
    )

    fig.set(facecolor='Yellow')
    # 1
    if tip != '':
        ax1.plot(to_array.history[tip],
                 label='Доля верных ответов на обучающем наборе')
        ax1.plot(to_array.history['val_' + tip],
                 label='Доля верных ответов на проверочном наборе')
        ax1.set_xlabel('Эпоха обучения ' + tip)
        ax1.set_ylabel('Доля верных ответов')
        ax1.legend()
        ax1.grid(axis='both', linewidth=1)
    # 2
    ax2.plot(to_array.history['loss'],
             label='Ошибка на обучающем наборе')
    ax2.plot(to_array.history['val_loss'],
             label='Ошибка на проверочном наборе')
    ax2.set_xlabel('Эпоха обучения loss')
    ax2.set_ylabel('Доля ошибок')
    ax2.legend()
    ax2.grid(axis='both', linewidth=1)
    # Выводим графики
    plt.show()
    return


def grafik_callback(to_array, *tip):
    # Функция для отображения истории во время обучения модели
    # Вход: вектор историй обучения
    # выводятся только val_accuracy и val_loss
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(18, 3)
    )

    fig.set(facecolor='Yellow')
    # 1
    ax1.plot(to_array[tip[0]],
             label='Доля верных ответов на обучающем наборе')
    ax1.plot(to_array[tip[1]],
             label='Доля верных ответов на проверочном наборе')
    ax1.set_xlabel('Эпоха обучения ' + tip[1])
    ax1.set_ylabel('Доля верных ответов')
    ax1.legend()
    ax1.grid(axis='both', linewidth=1)
    # 2
    ax2.plot(to_array[tip[2]],
             label='Ошибка на обучающем наборе')
    ax2.plot(to_array[tip[3]],
             label='Ошибка на проверочном наборе')
    ax2.set_xlabel('Эпоха обучения ' + tip[3])
    ax2.set_ylabel('Доля ошибок')
    ax2.legend()
    ax2.grid(axis='both', linewidth=1)

    plt.show()
    return


def grafik_sc(yy, pred, min_p=-1000, max_p=1000):
    # График Scetter и Гистограмма ошибок
    # yy - Подается yTrain, pred - Предсказания model.predict()
    # min_p, max_p - указание интервала для отображения значений на графике
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(18, 3)
    )

    fig.set(facecolor='Yellow')
    # Scetter
    ax1.scatter(yy, pred)
    ax1.set_xlabel('Правильные значение')
    ax1.set_ylabel('Предсказания')
    ax1.axis('equal')
    ax1.set_xlim(ax1.set_xlim())
    ax1.set_ylim(ax1.set_ylim())
    ax1.plot([min_p, max_p], [min_p, max_p])
    ax1.grid(axis='both', linewidth=1)

    # Гистограмма ошибок
    delta = pred - yy  # Вычитаем от Предсказанной правильную Цену
    ax2.hist(abs(delta).flatten(), bins=30)
    ax2.set_xlabel("Значение ошибки")
    ax2.set_ylabel("Количество")
    ax2.grid(axis='both', linewidth=1)

    # Выводим графики
    plt.show()
    return


def timezapusk(tip, *times):
    # Функция отображения времени исполнения
    # Вход: start - сохранение, возврат текущего времени
    # stop, times - остановка и печать использованного времени
    if tip == 'start':
        # print('Время обработки: 0')
        return time.time()
    if tip == 'stop':
        print('Время обработки: ', round(time.time() - times[0], 2), 'c', sep='')
    return round(time.time() - times[0], 2)


def my_graf(epoch, logs):
    # Выводит график ошибки
    global time_cur, time_tmp, time_close
    global rez, logs_
    global model
    global logs_, epoch_all, metrika, rez

    clear_output(wait=True)  # Очищаем экран
    # metrika = 'accuracy'
    rez.append([logs[metrika], logs['val_' + metrika], logs['loss'],
                logs['val_loss']])  # Собираем таблицу результатов для вывода на экран
    logs_ = pd.DataFrame(rez, columns=[metrika, 'val_' + metrika, 'loss',
                                       'val_loss'])  # преобразуем в пандас для рисования графика
    # Вывод графика
    grafik_callback(logs_, metrika, 'val_' + metrika, 'loss', 'val_loss')  # Выводим график ошибки
    print('Эпоха ', epoch + 1, ' из ', epoch_all, '   Текущий LR -',
          K.get_value(modelUnet.optimizer.lr))  # Выводим кол-во эпох тек - всего
    time_close = timezapusk('stop', time_tmp)  # Фиксируем время на конце эпохи
    time_cur += time_close  # Суммируем прошедшее время с начала всех эпох
    time_all = round(epoch_all * (time_cur / (epoch + 1)), 1)  # Вычисляем общее время, как среднее * кол-во эпох
    time_ost = round(time_all - time_cur, 1)  # Вычисляем сколько времени осталось до конца расчета
    print('Прошло ', round(time_cur, 1),
          'сек. | осталось ', time_ost,
          'сек. | из ', time_all, 'сек.')
    print('Последнее значение обучения\n', logs_[-5:])
    print('Максимальное значение обучения val_', metrika, '\n', logs_['val_' + metrika].max())


def my_time(epoch, logs):
    # Callback - выводит текущую эпоху, и сколько всего эпох
    # Выводит текущее время и сколько осталось
    global time_cur, time_tmp, time_close
    global epoch_all, metrika, rez

    clear_output(wait=True)
    print('Эпоха ', epoch + 1, ' из ', epoch_all)  # Выводим кол-во эпох тек - всего
    time_close = timezapusk('stop', time_tmp)  # Фиксируем время на конце эпохи
    time_cur += time_close  # Суммируем прошедшее время с начала всех эпох
    time_all = round(epoch_all * (time_cur / (epoch + 1)), 1)  # Вычисляем общее время, как среднее * кол-во эпох
    time_ost = round(time_all - time_cur, 1)  # Вычисляем сколько времени осталось до конца расчета
    print('Прошло ', round(time_cur, 1),
          'сек. | осталось ', time_ost,
          'сек. | из ', time_all, 'сек.')
    # print(stroka_vyvod)


def my_time_start(epoch, logs):
    global time_cur, time_tmp, time_close
    time_close = 0
    time_tmp = timezapusk('start')  # Запоминаем текущее время на начале эпохи


def my_init(logs):
    global time_cur, time_tmp, time_close
    global logs_, epoch_all, metrika, rez
    hist = []
    metrika = 'dice_coef'
    epoch_all = 400
    rez = []
    logs_ = pd.DataFrame(rez, columns=[metrika, 'val_' + metrika, 'loss', 'val_loss'])
    time_cur = 0
    time_tmp = time.time()
    print('Запуск обучения', time_cur)
    clear_output(wait=True)  # Очищаем экран


# Параметры для функции callback
# Прерывание обучение при не изменной ошибке
stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                     patience=39, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
# Измененение шага для оптимизатора при стагнации ошибки
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                              patience=33, verbose=1,
                                              mode='auto', cooldown=1, min_lr=1e-15)

# Коллбэки
pltGraf = LambdaCallback(on_epoch_end=my_graf)  # Конец эпохи
time_start = LambdaCallback(on_epoch_begin=my_time_start)  # Начало эпохи
time_stop = LambdaCallback(on_epoch_end=my_time)  # Начало эпохи
init = LambdaCallback(on_train_begin=my_init)  # Начало обучения, инициализация параметров

# Загрузка снимков

# Пример снимка ОПТГ, разбитых по категориям.
load_img(f'orig/I74485_.png', color_mode = 'grayscale')

# Пример созданных масок, разбитых по категориям.
# - голубой - леченый зуб
# - красный - нормальный зуб
# - розовый - коронка на зубе
# - сиреневый - мостовидный протез
# - бирюзовый - кариес на зубе
# - коричневый - имплант

load_img(f'mask/I74485.png')

# Задаем первоначальные параметры
img_width = 400
img_height = 200

## Загружаем снимки в базу

# Читаем каталог с изображениями масок, и загружаем такой же файл из основной категории
# Помещаем в два массива
X_image = []
X_image_mask = []
for filename in os.listdir(f'mask/'):
    filename = filename[:-4]  # Получаем только имя файла
    # Получаем снимок ОПТГ ,target_size=(img_height, img_width)
    optg = (np.asarray(keras.utils.load_img(f'orig/jpg/{filename}_.jpg',
                                target_size=(img_height, img_width),
                                color_mode='grayscale')) / 256)
    # optg = Image.open(f'orig/{filename}_.png')
    # Получаем маску снимка ОПТГ
    mask = (np.asarray(load_img(f'mask/{filename}.png',
                                target_size=(img_height, img_width)))).astype('uint8')
    # Записываем в массив
    X_image.append(optg)
    X_image_mask.append(mask)
# преобразовываем в numpy массив и делим на обучающую и тестовую
# X_image = np.stack(X_image).astype('uint8') # .astype('uint16') - добавил чтобы точно был этот тип, не 32
# X_image_mask = np.stack(X_image_mask).astype('uint8')
X_image = np.asarray(X_image)
X_image_mask = np.asarray(X_image_mask)

X_image_copy = X_image.copy()  # Создаем архив на всякий

### Удаляем фон на изображениях

# Суммируем все каналы цвета в один
# mask_tmp = (X_image_mask[:,:,:,0] + X_image_mask[:,:,:,1] + X_image_mask[:,:,:,2])

# X_image = X_image * np.around(mask_tmp / (mask_tmp + 1)) # Вырезаем фон на основном изображении

### Несколько примеров из базы

X_image = X_image_copy

X_image.shape

X_image[0].shape

X_image[0]

plt.imshow(X_image[1], cmap='gray')

X_image_mask[1]

plt.imshow(X_image_mask[1], cmap='gray')

X_image[4]

X_image_mask[4]

X_image.shape

X_image_mask.shape

# Показываем несколько примеров
ttt = 7
plt.imshow(X_image[ttt], cmap='gray')  # Здесь есть импланты, мосты, леченые, нормальные
plt.show()
plt.imshow(X_image_mask[ttt])
plt.show()

ttt = 9
plt.imshow(X_image[ttt], cmap='gray')  # Здесь есть импланты, мосты, леченые, нормальные
plt.show()
plt.imshow(X_image_mask[ttt])
plt.show()

### Определяем цветовые категории

ttt = 8
plt.imshow(X_image[ttt], cmap='gray')  # Здесь есть мосты, леченые, нормальные, коронки
plt.show()
plt.imshow(X_image_mask[ttt])  # Вывод маски для примера в цвете
plt.show()

# Вывод уникальных значений цветов по всем маскам и сохраняем в массив
img_ = 8  # анализируем 6 изображение, тут есть все категории, по цветам
colors_ = []  # будем хранить значения по каналам и сумму всех каналов
num_classes_ = []  # сумму цветных каналов, для сегментации
for h in range(img_height):
    for w in range(img_width):
        r = X_image_mask[img_][h, w, 0]  # красный канал
        g = X_image_mask[img_][h, w, 1]  # зеленый канал
        b = X_image_mask[img_][h, w, 2]  # синий канал
        if (np.unique(num_classes_) == r + g + b).any() == False and (r > 0 or g > 0 or b > 0):
            num_classes_.append([r + g + b])  # суммируем каналы и сохраняем в массиве
            colors_.append([r, g, b, r + g + b])  # запоминаем по отдельности значение канала и сумму для сравнения
            # colors_.append([str(r) + '/' + str(g) + '/'+ str(b) + '/'+ str(r + g + b)])
colors_ = pd.DataFrame(colors_)

colors_

num_classes_

## Построение выборок для моделей

val_cnt = 30  # процент от всей базы для проверочной выборки
val_cnt = int(X_image.shape[0] * (val_cnt / 100))  # количество элементов для проверочной выборки
val_cnt

train_images = X_image[:-val_cnt]  # Массив оригинальных изображений обучающей выборки
val_images = X_image[-val_cnt:]  # Массив оригинальных изображений проверочной выборки

train_segments = X_image_mask[:-val_cnt]  # Массив сегментированных изображений обучающей выборки
val_segments = X_image_mask[-val_cnt:]  # Массив сегментированных изображений проверочной выборки


# Функция преобразования пикселя сегментированного изображения в индекс (6 классов)
def color2index(color):
    color = color[0] + color[1] + color[2]
    index = 6  # Фон
    if (color == num_classes_[0]):
        index = 0  # тип 1
    elif (color == num_classes_[1]):
        index = 1  # тип 2
    elif (color == num_classes_[2]):
        index = 2  # тип 3
    elif (color == num_classes_[3]):
        index = 3  # тип 4
    elif (color == num_classes_[4]):
        index = 4  # тип 5
    elif (color == num_classes_[5]):
        index = 5  # тип 6
    return index


# Функция преобразования индекса в цвет пикселя
def index2color(index2):
    index = np.argmax(index2)  # Получаем индекс максимального элемента
    color = [0, 0, 0]  # Фон
    if index == 0:
        color = [31, 120, 180]  # тип 1 num_classes_[1]
    elif index == 1:
        color = [177, 89, 40]  # тип 2
    elif index == 2:
        color = [233, 180, 245]  # тип 3
    elif index == 3:
        color = [36, 207, 195]  # тип 4
    elif index == 4:
        color = [106, 61, 154]  # тип 5
    elif index == 5:
        color = [227, 26, 28]  # тип 6
    return color  # Возвращаем цвет пикслея


# Функция перевода индекса пикселя в to_categorical
def rgbToohe(y, num_classes):
    y2 = y.copy()  # Создаем копию входного массива
    y = y.reshape(y.shape[0] * y.shape[1], 3)  # Решейпим в двумерный массив
    yt = []  # Создаем пустой лист
    for i in range(len(y)):  # Проходим по всем пикселям изображения
        yt.append(utils.to_categorical(color2index(y[i]),
                                       num_classes=num_classes))  # Переводим пиксели в индексы и преобразуем в OHE
    yt = np.array(yt)  # Преобразуем в numpy
    yt = yt.reshape(y2.shape[0], y2.shape[1], num_classes)  # Решейпим к исходныму размеру
    return yt  # Возвращаем сформированный массив

print('ФОРМИРУЕМ yTrain')
# Функция формирования yTrain
def yt_prep(data, num_classes):
    yTrain = []  # Создаем пустой список под карты сегметации
    for seg in range(len(data)):  # Пробегаем по всем файлам набора с сегментированными изображениями
        # y = img_to_array(seg) # Переводим изображение в numpy-массив размерностью: высота - ширина - количество каналов
        y = rgbToohe(data[seg], num_classes)  # Получаем OHE-представление сформированного массива
        yTrain.append(y)  # Добавляем очередной элемент в yTrain
        # if len(yTrain) % 100 == 0: # Каждые 100 шагов
        #  print(len(yTrain)) # Выводим количество обработанных изображений
    return np.array(yTrain)  # Возвращаем сформированный yTrain


xTrain = []  # Создаем пустой список под обучающую выборку
for img in train_images:  # Проходим по всем изображениям из train_images
    # x = image.img_to_array(img) # Переводим изображение в numpy-массив размерностью: высота - ширина - количество каналов
    xTrain.append(img)  # Добавляем очередной элемент в xTrain
xTrain = np.array(xTrain)  # Переводим в numpy

xVal = []  # Создаем пустой список под проверочную выборку
for img in val_images:  # Проходим по всем изображениям из val_images
    # x = image.img_to_array(img) # Переводим изображение в numpy-массив размерностью: высота - ширина - количество каналов
    xVal.append(img)  # Добавляем очередной элемент в xTrain
xVal = np.array(xVal)  # Переводим в numpy

print('Размерность обучающей выборки   ', xTrain.shape)  # Размерность обучающей выборки
print('Размерность проверочной выборки ', xVal.shape)  # Размерность проверочной выборки

cur_time = time.time()  # Засекаем текущее время
yTrain = yt_prep(train_segments, len(num_classes_) + 1)  # Создаем yTrain
print('Время обработки: ', round(time.time() - cur_time, 2), 'c')  # Выводим время работы

cur_time = time.time()  # Засекаем текущее время
yVal = yt_prep(val_segments, len(num_classes_) + 1)  # Создаем yVal
print('Время обработки: ', round(time.time() - cur_time, 2), 'c')  # Выводим время работы

# Tuner настройки

#pip
#install - U
#keras - tuner

#from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization

## Создание пилотной модели

'''
  Функция создания сети
    Входные параметры:
    - num_classes - количество классов
    - input_shape - размерность карты сегментации
'''

print('TESTING MODEL...\n')

def unet(num_classes=7, input_shape=(img_height, img_width, 3),
         act_f=False, act='elu', cnt_max=0, do=False, dozn=0.5, ax=1,
         lrl=True, alpha=0.3, psp=False, bn=False, instN=True,
         j_con=True, kern_1=7, kern_2=5, kern_3=3, kern_4=2):
    #activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])

    init_k = RandomNormal(mean=0.0, stddev=0.02, seed=42)

    img_input = Input(input_shape)  # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv2D(128, (kern_1, kern_1), kernel_initializer=init_k,
               padding='same',
               dilation_rate=(3, 3),
               name='block1_conv1')(img_input)  # Добавляем Conv2D-слой с 64-нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax,
                                        center=True,
                                        scale=True,
                                        beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    x = Conv2D(128, (kern_1, kern_1), kernel_initializer=init_k,
               padding='same',
               dilation_rate=(3, 3),
               name='block1_conv2')(x)  # Добавляем Conv2D-слой с 64-нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax,
                                        center=True,
                                        scale=True,
                                        beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)

    block_1_out = Activation(act)(x)  # Добавляем слой Activation и запоминаем в переменной block_1_out
    # x = MaxPooling2D()(block_1_out)                                        # Добавляем слой MaxPooling2D
    x = Conv2D(256, (kern_2, kern_2), kernel_initializer=init_k,
               padding='same',
               strides=(2, 2))(block_1_out)

    # Block 2
    x = Conv2D(256, (kern_2, kern_2), kernel_initializer=init_k,
               padding='same',
               dilation_rate=(3, 3),
               name='block2_conv1')(x)  # Добавляем Conv2D-слой с 128-нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax, center=True, scale=True, beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    x = Conv2D(256, (kern_2, kern_2), kernel_initializer=init_k,
               padding='same',
               dilation_rate=(3, 3),
               name='block2_conv2')(x)  # Добавляем Conv2D-слой с 128-нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax, center=True, scale=True, beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)

    block_2_out = Activation(act)(x)  # Добавляем слой Activation и запоминаем в переменной block_2_out
    # x = MaxPooling2D()(block_2_out)                                        # Добавляем слой MaxPooling2D
    x = Conv2D(512, (kern_3, kern_3), kernel_initializer=init_k,
               padding='same',
               strides=(2, 2))(block_2_out)

    # Block 3
    x = Conv2D(512, (kern_3, kern_3), kernel_initializer=init_k,
               dilation_rate=(3, 3),
               padding='same', name='block3_conv1')(x)  # Добавляем Conv2D-слой с 256-нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax, center=True, scale=True, beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    x = Conv2D(512, (kern_3, kern_3), kernel_initializer=init_k,
               dilation_rate=(3, 3),
               padding='same', name='block3_conv2')(x)  # Добавляем Conv2D-слой с 256-нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax, center=True, scale=True, beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    x = Conv2D(512, (kern_3, kern_3), kernel_initializer=init_k,
               dilation_rate=(3, 3),
               padding='same', name='block3_conv3')(x)  # Добавляем Conv2D-слой с 256-нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax, center=True, scale=True, beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)

    block_3_out = Activation(act)(x)  # Добавляем слой Activation и запоминаем в переменной block_3_out
    # x = MaxPooling2D()(block_3_out)                                        # Добавляем слой MaxPooling2D
    x = Conv2D(1024, (kern_3, kern_3), kernel_initializer=init_k,
               padding='same',
               strides=(2, 2))(block_3_out)

    # Block 4
    x = Conv2D(1024, (kern_4, kern_4), kernel_initializer=init_k,
               dilation_rate=(3, 3),
               padding='same', name='block4_conv1')(x)  # Добавляем Conv2D-слой с 512-нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax, center=True, scale=True, beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    x = Conv2D(1024, (kern_4, kern_4), kernel_initializer=init_k,
               dilation_rate=(3, 3),
               padding='same', name='block4_conv2')(x)  # Добавляем Conv2D-слой с 256-нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax, center=True, scale=True, beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    x = Conv2D(1024, (kern_4, kern_4), kernel_initializer=init_k,
               dilation_rate=(3, 3),
               padding='same', name='block4_conv3')(x)  # Добавляем Conv2D-слой с 256-нейронами
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax, center=True, scale=True, beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    # ------------- ПЕРЕВОРОТ -------------
    block_4_out = Activation(act)(x)  # Добавляем слой Activation и запоминаем в переменной block_4_out
    x = block_4_out
    # ------------- ПЕРЕВОРОТ -------------

    if psp:
        # ----------------- PSPnet
        pool_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(
            block_1_out)  # Добавляем слой MaxPooling2D
        pool_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(
            block_1_out)  # Добавляем слой MaxPooling2D
        pool_3 = MaxPooling2D(pool_size=(3, 3), strides=(4, 4), padding='same')(
            block_1_out)  # Добавляем слой MaxPooling2D
        pool_6 = MaxPooling2D(pool_size=(3, 3), strides=(8, 8), padding='same')(
            block_1_out)  # Добавляем слой MaxPooling2D

        # pool_1
        pool_1 = Conv2D(512, (5, 5), kernel_initializer=init_k,
                        padding='same')(pool_1)  # Добавляем Conv2D-слой с 128-нейронами
        # pool_1 = Dropout(dozn)(pool_1)
        pool_1 = BatchNormalization()(pool_1)  # Добавляем слой BatchNormalization
        pool_1 = Activation(act)(pool_1)  # Добавляем слой Activation

        pool_1 = Conv2D(256, (4, 4), kernel_initializer=init_k,
                        padding='same')(pool_1)  # Добавляем Conv2D-слой с 128-нейронами
        # pool_1 = Dropout(dozn)(pool_1)
        pool_1 = BatchNormalization()(pool_1)  # Добавляем слой BatchNormalization
        pool_1 = Activation(act)(pool_1)  # Добавляем слой Activation

        pool_1 = Conv2D(128, (3, 3), kernel_initializer=init_k,
                        padding='same')(pool_1)  # Добавляем Conv2D-слой с 128-нейронами
        # pool_1_ = Dropout(dozn)(pool_1)
        pool_1 = BatchNormalization()(pool_1)  # Добавляем слой BatchNormalization
        pool_1_ = Activation(act)(pool_1)  # Добавляем слой Activation и запоминаем в переменной block_2_out

        # pool_2
        pool_2 = Conv2D(512, (5, 5), kernel_initializer=init_k,
                        padding='same')(pool_2)  # Добавляем Conv2D-слой с 128-нейронами
        # pool_2 = Dropout(dozn)(pool_2)
        pool_2 = BatchNormalization()(pool_2)  # Добавляем слой BatchNormalization
        pool_2 = Activation(act)(pool_2)  # Добавляем слой Activation

        pool_2 = Conv2D(256, (4, 4), kernel_initializer=init_k,
                        padding='same')(pool_2)  # Добавляем Conv2D-слой с 128-нейронами
        # pool_2 = Dropout(dozn)(pool_2)
        pool_2 = BatchNormalization()(pool_2)  # Добавляем слой BatchNormalization
        pool_2 = Activation(act)(pool_2)  # Добавляем слой Activation

        pool_2 = Conv2D(128, (3, 3), kernel_initializer=init_k,
                        padding='same')(pool_2)  # Добавляем Conv2D-слой с 128-нейронами
        # pool_2_ = Dropout(dozn)(pool_2)
        pool_2 = BatchNormalization()(pool_2)  # Добавляем слой BatchNormalization
        pool_2_ = Activation(act)(pool_2)  # Добавляем слой Activation и запоминаем в переменной block_2_out

        # pool_3
        pool_3 = Conv2D(512, (5, 5), kernel_initializer=init_k,
                        padding='same')(pool_3)  # Добавляем Conv2D-слой с 128-нейронами
        # pool_3 = Dropout(dozn)(pool_3)
        pool_3 = BatchNormalization()(pool_3)  # Добавляем слой BatchNormalization
        pool_3 = Activation(act)(pool_3)  # Добавляем слой Activation

        pool_3 = Conv2D(256, (4, 4), kernel_initializer=init_k,
                        padding='same')(pool_3)  # Добавляем Conv2D-слой с 128-нейронами
        # pool_3 = Dropout(dozn)(pool_3)
        pool_3 = BatchNormalization()(pool_3)  # Добавляем слой BatchNormalization
        pool_3 = Activation(act)(pool_3)  # Добавляем слой Activation

        pool_3 = Conv2D(128, (3, 3), kernel_initializer=init_k,
                        padding='same')(pool_3)  # Добавляем Conv2D-слой с 128-нейронами
        # pool_3_ = Dropout(dozn)(pool_3)
        pool_3 = BatchNormalization()(pool_3)  # Добавляем слой BatchNormalization
        pool_3_ = Activation(act)(pool_3)  # Добавляем слой Activation и запоминаем в переменной block_2_out

        # pool_6
        pool_6 = Conv2D(512, (5, 5), kernel_initializer=init_k,
                        padding='same')(pool_6)  # Добавляем Conv2D-слой с 128-нейронами
        # pool_6 = Dropout(dozn)(pool_6)
        pool_6 = BatchNormalization()(pool_6)  # Добавляем слой BatchNormalization
        pool_6 = Activation(act)(pool_6)  # Добавляем слой Activation

        pool_6 = Conv2D(256, (4, 4), kernel_initializer=init_k,
                        padding='same')(pool_6)  # Добавляем Conv2D-слой с 128-нейронами
        # pool_6 = Dropout(dozn)(pool_6)
        pool_6 = BatchNormalization()(pool_6)  # Добавляем слой BatchNormalization
        pool_6 = Activation(act)(pool_6)  # Добавляем слой Activation

        pool_6 = Conv2D(128, (3, 3), kernel_initializer=init_k,
                        padding='same')(pool_6)  # Добавляем Conv2D-слой с 128-нейронами
        # pool_6_ = Dropout(dozn)(pool_6)
        pool_6 = BatchNormalization()(pool_6)  # Добавляем слой BatchNormalization
        pool_6_ = Activation(act)(pool_6)  # Добавляем слой Activation и запоминаем в переменной block_2_out

        # UPSAMPLING
        pool_1_out = UpSampling2D(size=(1, 1))(pool_1_)
        pool_2_out = UpSampling2D(size=(2, 2))(pool_2_)
        pool_3_out = UpSampling2D(size=(4, 4))(pool_3_)
        pool_6_out = UpSampling2D(size=(8, 8))(pool_6_)
        # pool_1_out = Conv2DTranspose(256, (2, 2), strides=(1, 1), padding='same')(pool_1_)
        # pool_2_out = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(pool_2_)
        # pool_3_out = Conv2DTranspose(256, (2, 2), strides=(4, 4), padding='same')(pool_3_)
        # pool_6_out = Conv2DTranspose(256, (2, 2), strides=(8, 8), padding='same')(pool_6_)

        # ----------------- PSPnet

        x = concatenate([x, pool_6_])

    # UP 2
    # x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(512, (kern_3, kern_3), kernel_initializer=init_k,
                        strides=2, padding='same')(x)  # Добавляем слой Conv2DTranspose с 256 нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax,
                                        center=True,
                                        scale=True,
                                        beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    if j_con: x = concatenate([x, block_3_out])  # Объединем текущий слой со слоем block_3_out

    x = Conv2D(512, (kern_3, kern_3), kernel_initializer=init_k,
               # dilation_rate = (3, 3),
               padding='same',
               name='up3_conv1')(x)  # Добавляем слой Conv2D с 256 нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax,
                                        center=True,
                                        scale=True,
                                        beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    x = Conv2D(512, (kern_3, kern_3), kernel_initializer=init_k,
               # dilation_rate = (3, 3),
               padding='same',
               name='up3_conv2')(x)
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax,
                                        center=True,
                                        scale=True,
                                        beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    if cnt_max > 1:
        for _ in range(cnt_max):
            if psp:
                y = Activation('softmax')(y)
            else:
                x = Activation('softmax')(x)
        if psp:
            y = Dropout(dozn)(y)
        else:
            x = Dropout(dozn)(x)

    if psp: x = concatenate([x, pool_3_])

    # UP 3
    # x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(256, (kern_2, kern_2), kernel_initializer=init_k,
                        strides=2, padding='same')(x)  # Добавляем слой Conv2DTranspose с 128 нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax,
                                        center=True,
                                        scale=True,
                                        beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    if j_con: x = concatenate([x, block_2_out])  # Объединем текущий слой со слоем block_2_out

    x = Conv2D(256, (kern_2, kern_2), kernel_initializer=init_k,
               # dilation_rate = (3, 3),
               padding='same',
               name='up2_conv1')(x)  # Добавляем слой Conv2D с 128 нейронами
    if do: x = Dropout(dozn)(x)
    if instN: x = InstanceNormalization(axis=ax,
                                        center=True,
                                        scale=True,
                                        beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    x = Conv2D(256, (kern_2, kern_2), kernel_initializer=init_k,
               # dilation_rate = (3, 3),
               padding='same',
               name='up2_conv2')(x)  # Добавляем слой Conv2D с 128 нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax,
                                        center=True,
                                        scale=True,
                                        beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    if cnt_max > 1:
        for _ in range(cnt_max):
            if psp:
                y = Activation('softmax')(y)
            else:
                x = Activation('softmax')(x)
        if psp:
            y = Dropout(dozn)(y)
        else:
            x = Dropout(dozn)(x)

    if psp: x = concatenate([x, pool_2_])

    # UP 4
    # x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(128, (kern_1, kern_1), kernel_initializer=init_k,
                        strides=2, padding='same')(x)  # Добавляем слой Conv2DTranspose с 64 нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax,
                                        center=True,
                                        scale=True,
                                        beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    # Concat
    # Подключает PSPnet
    if psp: y = concatenate([pool_1_out, pool_2_out, pool_3_out, pool_6_out])
    if cnt_max > 1:
        for _ in range(cnt_max):
            if psp:
                y = Activation('softmax')(y)
            else:
                x = Activation('softmax')(x)
        if psp:
            y = Dropout(dozn)(y)
        else:
            x = Dropout(dozn)(x)

    if psp: x = concatenate([x, y])

    if j_con: x = concatenate([x, block_1_out])  # Объединем текущий слой со слоем block_1_out

    x = Conv2D(128, (kern_1, kern_1), kernel_initializer=init_k,
               # dilation_rate = (3, 3),
               padding='same',
               name='up1_conv1')(x)  # Добавляем слой Conv2D с 64 нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax,
                                        center=True,
                                        scale=True,
                                        beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    x = Conv2D(128, (kern_1, kern_1), kernel_initializer=init_k,
               # dilation_rate = (3, 3),
               padding='same',
               name='up1_conv2')(x)  # Добавляем слой Conv2D с 64 нейронами
    if do: x = Dropout(dozn)(x)
    if bn: x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    if instN: x = InstanceNormalization(axis=ax,
                                        center=True,
                                        scale=True,
                                        beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform")(x)
    if lrl: x = LeakyReLU(alpha=alpha)(x)
    if act_f: x = Activation(act)(x)  # Добавляем слой Activation

    # Concat
    if psp: y = concatenate([pool_1_out, pool_2_out, pool_3_out, pool_6_out, block_1_out])
    if cnt_max > 1:
        for _ in range(cnt_max):
            if psp:
                y = Activation('softmax')(y)
            else:
                x = Activation('softmax')(x)
        if psp:
            y = Dropout(dozn)(y)
        else:
            x = Dropout(dozn)(x)
    if psp: x = concatenate([x, y])

    # Выходной слой
    x = Conv2D(num_classes, (2, 2),
               activation='softmax',
               padding='same')(x)  # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

    model = Model(img_input, x)  # Создаем модель с входом 'img_input' и выходом 'x'

    return model  # Возвращаем сформированную модель


def unet_new(num_classes=7, input_shape=(img_height, img_width, 3),
             act_f=False, act='elu', cnt_max=0, do=False, dozn=0.5, ax=1,
             lrl=True, alpha=0.3, psp=False, bn=False, instN=True,
             j_con=True):
    inputs = Input(img_height, img_width, 3)

    init_k = RandomNormal(mean=0.0, stddev=0.02, seed=42)

    conv1 = Conv2D(64, (3, 3), activation=act, padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation=act, padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation=act, padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation=act, padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation=act, padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation=act, padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation=act, padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation=act, padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation=act, padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation=act, padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), pool4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation=act, padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation=act, padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), pool3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation=act, padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation=act, padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), pool2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation=act, padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation=act, padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), pool1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation=act, padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation=act, padding='same')(conv9)

    conv10 = Conv2D(num_classes, (2, 2), activation='softmax', padding='same')(conv9)

    # model = Model(inputs=[inputs], outputs=[conv10])

    # Выходной слой
    # x = Conv2D(num_classes, (2, 2),
    #            activation='softmax',
    #            padding='same')(x)  # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

    model = Model(inputs, conv10)  # Создаем модель с входом 'img_input' и выходом 'x'

    return model  # Возвращаем сформированную модель


'''
  Собственная функция метрики, обрабатывающая пересечение двух областей
'''


def dice_coef(y_true, y_pred):
    # tf.keras.backend.flatten()
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1.0)
    # return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.) # Возвращаем площадь пересечения деленную на площадь объединения двух областей


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


### Тестируем

# Создание модели
'''
len(num_classes_)+1,        # Кол-во классов
(img_height, img_width, 1), # Входной размер картинки
act_f = True,               # Использовать функцию Activation, yes/no, True
act = 'selu',               # Функция активации, 'elu'
cnt_max = 0,                # Кол-во блоков активации идущих подряд, 0
do = False,                 # Использовать Dropout, y/n, False
dozn = 0.7,                 # Значение функции Dropout, 0.7
lrl = False,                # Использовать LeakyReLU, y/n, False
alpha = 0.7,                # Значение alpha LeakyReLU, 0.7
psp = True,                 # Использовать сеть PSPnet, y/n, False
bn = True,                  # Использовать BatchNormalization, y/n, False
instN = True,               # Использовать InstanceNormalization, y/n, True
ax = 3,                     # Значение axis InstanceNormalization, 3
j_con = True                # Использовать concatenate, y/n, True
'''
# Создаем модель unet
modelUnet = unet(len(num_classes_) + 1, (img_height, img_width, 1),
                 act_f=True, act='elu',
                 cnt_max=0,
                 do=False, dozn=0.2,
                 lrl=False, alpha=0.4,
                 psp=True,
                 bn=True,
                 instN=True, ax=3,
                 j_con=True,
                 kern_1=13, kern_2=13, kern_3=5, kern_4=3)

'''
----- Последний вариант
                 act_f = True, act = 'selu', \
                 cnt_max = 0, \
                 do = False, dozn = 0.2, \
                 lrl = True, alpha = 0.2, \
                 psp = True, \
                 bn = True, \
                 instN = True, ax = 3, \
                 j_con = False
------ Качественный вариант
modelUnet = unet(len(num_classes_)+1, (img_height, img_width, 1), \
                 act_f = True, act = 'selu', \
                 cnt_max = 0, \
                 do = False, dozn = 0.2, \
                 lrl = True, alpha = 0.2, \
                 psp = False, \
                 bn = True, \
                 instN = True, ax = 3, \
                 j_con = True,
                 kern_1 = 3, kern_2 = 5, kern_3 = 5, kern_4 = 3)
----- Результат не плохой, но много шума
                 act_f = True, act = 'selu', \
                 cnt_max = 0, \
                 do = False, dozn = 0.2, \
                 lrl = False, alpha = 0.2, \
                 psp = True, \
                 bn = True, \
                 instN = True, ax = 3, \
                 j_con = True)
----- Результат Отличный, не хватает примеров
                 act_f = True, act = 'selu', \
                 cnt_max = 0, \
                 do = False, dozn = 0.2, \
                 lrl = False, alpha = 0.8, \
                 psp = False, \
                 bn = True, \
                 instN = True, ax = 3, \
                 j_ave = True, \
                 j_con = False
----- Результат хороший только для коронок
                 act_f = True, act = 'selu', \
                 cnt_max = 0, \
                 do = False, dozn = 0.7, \
                 lrl = True, alpha = 0.7, \
                 psp = False, \
                 bn = True, \
                 instN = True, ax = 3, \
                 j_ave = False, \
                 j_con = True
----- Мосты, коронки
                 act_f = True, act = 'selu', \
                 cnt_max = 0, \
                 do = False, dozn = 0.7, \
                 lrl = True, alpha = 0.7, \
                 psp = False, \
                 bn = True, \
                 instN = True, ax = 3, \
                 j_ave = False, \
                 j_con = True
----- Результата НЕТ
                 act_f = True, act = 'elu', \
                 cnt_max = 7, \
                 do = True, dozn = 0.3, \
                 lrl = False, alpha = 0.7, \
                 psp = False, \
                 bn = False, \
                 instN = False, ax = 3, \
                 j_ave = True, \
                 j_con = True
----- Приемленный результат, Импланты, Мосты, но есть мусор
                 act_f = True, act = 'elu', \
                 cnt_max = 0, \
                 do = False, dozn = 0.7, \
                 lrl = True, alpha = 0.7, \
                 psp = True, \
                 bn = True, \
                 instN = True, ax = 3, \
                 j_ave = False, \
                 j_con = True
------ Рез-т на 4ку
                 act_f = True, act = 'selu', \
                 cnt_max = 0, \
                 do = False, dozn = 0.3, \
                 lrl = False, alpha = 0.8, \
                 psp = True, \
                 bn = True, \
                 instN = True, ax = 3, \
                 j_ave = False, \
                 j_con = True)
------ Рез-та НЕТ
                 act_f = True, act = 'selu', \
                 cnt_max = 0, \
                 do = False, dozn = 0.3, \
                 lrl = False, alpha = 0.8, \
                 psp = False, \
                 bn = True, \
                 instN = False, ax = 3, \
                 j_ave = False, \
                 j_con = False)
'''

### Рисунок модели

# tf.keras.utils.plot_model(modelUnet, show_shapes=True, rankdir='TB')

### **Компилируем и тестируем**

# Компилируем модель - Adam
modelUnet.compile(optimizer=Adam(lr=0.4),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
# Компилируем модель - RMSprop
# modelUnet.compile(optimizer=Adam(lr = 0.1),
#                  loss=dice_coef_loss,
#                  metrics=[dice_coef])
# Компилируем модель - RMSprop
# modelUnet.compile(optimizer=RMSprop(lr = 0.1),
#                  loss=dice_coef_loss,
#                  metrics=[dice_coef])
# modelUnet.compile(optimizer=SGD(lr = 0.1),
#                   loss='categorical_crossentropy',
#                   metrics=[dice_coef])
# modelUnet.compile(optimizer=RMSprop(lr = 0.1),
#                  loss='categorical_crossentropy',
#                  metrics=[dice_coef])
modelUnet.summary()

epoch_all = 400
result = modelUnet.fit(xTrain,
                       yTrain,
                       epochs=epoch_all,
                       batch_size=2,
                       verbose=1,
                       callbacks=[reduce_lr, stop, init, time_start, time_stop, pltGraf],
                       validation_data=(xVal, yVal))  # Обучаем модель на выборке по классам
grafik(result, 'dice_coef')

#### Распознавание

# Сохраняем модель и веса
modelUnet.save('/content/drive/My Drive/_AI/_diplom/modelUnet_.h5')
modelUnet.save_weights('/content/drive/My Drive/_AI/_diplom/weightsUnet_.h5')


# modelUnet_ = keras.models.load_model('/content/drive/My Drive/_AI/_diplom/modelUnet_')
# modelUnet_.load('/content/drive/My Drive/_AI/_diplom/modelUnet_.h5')

# Функция визуализации сегментированных изображений
def processImage(model, n_classes=7, idx_=0):
    idx = idx_
    print('image', xVal[idx].reshape(1, img_height, img_width, 1).shape)
    predict = np.array(model.predict(xVal[idx].reshape(1, img_height, img_width, 1)))  # Предиктим картинку
    pr = predict[0]  # Берем нулевой элемент из предикта
    # print('pedict', pr.shape)
    # print('pr', pr.shape)
    # print('средняя', pr.mean())
    # print('max', pr.max())

    pr1 = []  # Пустой лист под сегментированную картинку из predicta
    pr2 = []  # Пустой лист под сегменитрованную картинку из yVal
    pr = pr.reshape(-1, n_classes)  # Решейпим предикт
    # print('pr', pr.shape)
    # print('----', pr[64312])
    # print(np.argmax(pr[64312]))
    yr = yVal[idx].reshape(-1, n_classes)  # Решейпим yVal
    print('pr-', pr.shape)
    print('yr-', yr.shape)
    for k in range(len(pr)):  # Проходим по всем уровням (количество классов)
        pr1.append(index2color(pr[k]))  # Переводим индекс в пиксель
        pr2.append(index2color(yr[k]))  # Переводим индекс в пиксель

    print(index2color(pr[10000]))
    pr1 = np.array(pr1)  # Преобразуем в numpy
    print('pr1-', pr1.shape)
    pr1 = pr1.reshape(img_height, img_width, 3)  # Решейпим к размеру изображения
    pr2 = np.array(pr2)  # Преобразуем в numpy
    print('pr2-', pr2.shape)
    pr2 = pr2.reshape(img_height, img_width, 3)  # Решейпим к размеру изображения
    plt.imshow(Image.fromarray(pr1.astype('uint8')))  # Отображаем на графике в первой линии
    plt.show()
    plt.imshow(Image.fromarray(
        pr2.astype('uint8')))  # Отображаем на графике во второй линии сегментированное изображение из yVal
    plt.show()
    plt.imshow(Image.fromarray(xVal[idx]),
               cmap='gray')  # Отображаем на графике в третьей линии оригинальное изображение
    plt.show()


xVal[0].shape

# Проверяем на xVal
processImage(modelUnet, len(num_classes_) + 1, 0)

processImage(modelUnet, len(num_classes_) + 1, 1)

processImage(modelUnet, len(num_classes_) + 1, 2)

processImage(modelUnet, len(num_classes_) + 1, 3)


### Проверка неизвестного снимка

# from keras.models import load_model

# modelUnet_ = load_model('/content/drive/My Drive/_AI/_diplom/modelUnet_.h5')

# modelUnet = modelUnet_
# modelUnet.load('/content/drive/My Drive/_AI/_diplom/modelUnet_.h5')

# Функция визуализации сегментированных изображений
def NewImage(model, n_classes=7, path=''):
    global newimg
    delit = 1
    if np.asarray(load_img(path, grayscale=True)).max() > 256:
        delit = 256
    newimg = (np.asarray(load_img(path,
                                  target_size=(img_height, img_width),
                                  grayscale=True)) / delit).astype('uint8')
    print('image', newimg.reshape(1, img_height, img_width, 1).shape)
    predict = np.array(model.predict(newimg.reshape(1, img_height, img_width, 1)))  # Предиктим картинку
    pr = predict[0]  # Берем нулевой элемент из предикта
    # print('predict', predict.shape)
    # print('pr', pr.shape)
    # print('средняя', pr.mean())
    # print('max', pr.max())

    pr1 = []  # Пустой лист под сегментированную картинку из predicta
    pr = pr.reshape(-1, n_classes)  # Решейпим предикт
    # print('pr', pr.shape)
    # print('----', pr[99382])
    # print(np.argmax(pr[14312:59000]))
    for k in range(len(pr)):  # Проходим по всем уровням (количесвто классов)
        pr1.append(index2color(pr[k]))  # Переводим индекс в пиксель
    # print('-- ', len(pr1))
    pr1 = np.array(pr1)  # Преобразуем в numpy
    # print('max     - ', pr1[0:130100].max())
    # print('---', pr1.shape)
    pr1 = pr1.reshape(img_height, img_width, 3)  # Решейпим к размеру изображения
    # print('---', pr1.shape)
    # print(pr1.mean())
    plt.imshow(Image.fromarray(pr1.astype('uint8')))  # Отображаем на графике в первой линии
    plt.show()
    plt.imshow(Image.fromarray(newimg.astype('uint8')),
               cmap='gray')  # Отображаем на графике в третьей линии оригинальное изображение
    plt.show()


ppp = f'/content/drive/My Drive/_AI/_diplom/test/I105049.png'
NewImage(modelUnet, 7, ppp)

# Из интернета
ppp = f'/content/drive/My Drive/_AI/_diplom/test/001.jpg'
NewImage(modelUnet, 7, ppp)

# Из интернета
ppp = f'/content/drive/My Drive/_AI/_diplom/test/002.jpg'
NewImage(modelUnet, 7, ppp)

# Из интернета
ppp = f'/content/drive/My Drive/_AI/_diplom/test/003.jpg'
NewImage(modelUnet, 7, ppp)

# Из интернета
ppp = f'/content/drive/My Drive/_AI/_diplom/test/004.jpg'
NewImage(modelUnet, 7, ppp)

# Из интернета
ppp = f'/content/drive/My Drive/_AI/_diplom/test/005.jpg'
NewImage(modelUnet, 7, ppp)

# Из интернета
ppp = f'/content/drive/My Drive/_AI/_diplom/test/006.jpg'
NewImage(modelUnet, 7, ppp)

# Из интернета
ppp = f'/content/drive/My Drive/_AI/_diplom/test/007.jpg'
NewImage(modelUnet, 7, ppp)

# Из интернета
ppp = f'/content/drive/My Drive/_AI/_diplom/test/008.jpg'
NewImage(modelUnet, 7, ppp)

# Случайный снимок из интернета
import requests
from io import BytesIO

ppp = 'https://x-rdv.ru/wp-content/plugins/phastpress/phast.php/https-3A-2F-2Fx-2Drdv.ru-2Fwp-2Dcontent-2Fuploads-2F2020-2F04-2Fortopanotomogramma-2Doptg-2Dscaled.jpg/service=images/cacheMarker=1589361037-2D473236/token=a87a65571c11609a/__p__.jpg'
response = requests.get(ppp)
img = Image.open(BytesIO(response.content))
img.save(fp="\img.jpg")
NewImage(modelUnet, 7, "\img.jpg")

ppp = 'https://smile-at-once.ru/data/images/video_gallery/babk-snimok-do.jpg'
response = requests.get(ppp)
img = Image.open(BytesIO(response.content))
img.save(fp="\img.jpg")
NewImage(modelUnet, 7, "\img.jpg")