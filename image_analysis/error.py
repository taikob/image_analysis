from PIL import Image
import numpy as np
import os

def load_img(path, size=None):
    im = Image.open(path)
    if size is not None: im = im.resize(size)
    im = np.asarray(im).astype(np.float32) / float(255)  # true image(BGR)
    return im

def error(path, cmppath,cmp = None,savepath=None,size=None):

    img = load_img(path)
    if cmp is None: cmp = load_img(cmppath,size=size)

    er = img - cmp

    erA = np.average(np.abs(er[:, :, :] ** 2))
    erB = np.average(np.abs(er[:, :, 0] ** 2))
    erG = np.average(np.abs(er[:, :, 1] ** 2))
    erR = np.average(np.abs(er[:, :, 2] ** 2))
    erdev = np.std(np.array([erR, erG, erB]))
    if savepath is not None:
        if os.path.exists(savepath):
            os.makedirs(savepath)
        np.savetxt(savepath + '/error.csv', np.array([[erA, erR, erG, erB, erdev]]), delimiter=',')
    else: np.savetxt('error.csv', np.array([[erA, erR, erG, erB, erdev]]), delimiter=',')


if __name__ == '__main__':
    path = 'test_18y_0.jpg'
    cmppath = 'rotsnake_ctrl.jpg'
    error(path, cmppath,size=[256,256])