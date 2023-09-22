import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm as normal_distr
from scipy.stats import poisson, gamma

import tkinter as tk
from tkinter import filedialog


img_path = None
img = None
crop = None


def calc_distr_params(noise, distr_type):
    if distr_type == 'norm':
        distr = stats.norm
    if distr_type == 'gamma':
        distr = stats.gamma
        
    params = distr.fit(noise.reshape(-1))

    return params


def calc_contrast(img):
    contrast = (img.ravel().max() - img.ravel().min()) / (img.ravel().max() + img.ravel().min())
    return np.round(contrast, 4)


def calc_psnr(noisy_crop, noise):
    clean = noisy_crop - noise
    psnr = cv2.PSNR(clean, noisy_crop)
    return psnr


def show_image(img):
    global crop
    fig, ax = plt.subplots(1, 1)

    toolbar = fig.canvas.manager.toolbar
    fig.set_tight_layout(True)
    plt.imshow(img, cmap='gray')
    plt.title('С помощью инструмента лупа выделите кусок только с фоном\n\
              После выбора окно НУЖНО закрыть, выбранный участок сохранится для анализа')
    plt.axis('off')
    plt.show()
    plt.close()

    old_x_lim = ax.get_xlim()
    old_y_lim = ax.get_ylim()

    crop = img[int(old_y_lim[1]):int(old_y_lim[0]), int(old_x_lim[0]):int(old_x_lim[1])]
    process_crop(crop)


def select_img():
    global img_path, img, img_name
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.tif *.tiff *.png *.jpg *.jpeg")])
    img_name = img_path.split('/')[-1][:-4]
    
    if img_path:
        img = cv2.imread(img_path, 0).astype(np.float32)
        show_image(img)


def interpret_results(a, scale, std_noise, contrast, psnr):
    if a > 70:
        a_bool = True
    else:
        a_bool = False

    if scale > 0 and scale < 1:
        scale_bool = True
    else:
        scale_bool = False

    if std_noise < 9:
        std_noise_bool = True
    else:
        std_noise_bool = False

    if contrast < 0.7:
        contrast_bool = True
    else:
        contrast_bool = False

    if psnr > 29:
        psnr_bool = True
    else:
        psnr_bool = False


    if a_bool and scale_bool and std_noise_bool and contrast_bool and psnr:
        return 'Уровень шума удовлетворительный'
    
    else:
        return 'Уровень шума неудовлетворительный'


def process_crop(crop):
    mode = stats.mode(crop.ravel())[0]
    noise = crop - mode
    params_gamma_noise_crop = calc_distr_params(noise, 'gamma') # a, loc, scale
    a = params_gamma_noise_crop[0]
    scale = params_gamma_noise_crop[2]
    std_noise = np.round(noise.std(), 4)
    contrast = calc_contrast(img)
    psnr_phr = calc_psnr(crop, noise)
    
    print('Для изображения {}\nПараметры гамма распределения: a = {:.2f}, scale = {:.2f}\nstd in crop PhR = {:.2f},\
          image contrast = {:.2f}\n\
          psnr in crop = {:.2f}'.format(img_name, a, scale, std_noise, contrast, psnr_phr))
    
    print(interpret_results(a, scale, std_noise, contrast, psnr_phr), '\n')


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Анализ изображения")
    load_button = tk.Button(root, text=f"   Upload image    ", command=select_img)
    load_button.pack(pady=100)

    root.mainloop()