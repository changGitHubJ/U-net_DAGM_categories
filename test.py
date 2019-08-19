import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

NUM = 30
IMG_SIZE = 256
OUTPUT_SIZE = 256*256
CATEGORY = 2

def readImages(filename):
    images = np.zeros((NUM*CATEGORY, IMG_SIZE*IMG_SIZE))
    fileImg = open(filename)
    for k in range(NUM*CATEGORY):
        line = fileImg.readline()
        if(not line):
            break
        val = line.split(',')
        for i in range(IMG_SIZE*IMG_SIZE):
            images[k, i] = float(val[i + 1])
    return images

def readLabels(filename):
    labels = np.zeros((NUM*CATEGORY, OUTPUT_SIZE*CATEGORY))
    fileImg = open(filename)
    for k in range(NUM*CATEGORY):
        line = fileImg.readline()
        if(not line):
            break
        val = line.split(',')
        for i in range(OUTPUT_SIZE*CATEGORY):
            labels[k, i] = float(val[i + 1])
    return labels

if __name__=='__main__':
    tst_image = readImages('./data/testImage256.txt')
    tst_label = readLabels('./data/testLABEL256.txt')
    label = tst_label.reshape([-1, IMG_SIZE, IMG_SIZE, CATEGORY])

    for i in range(NUM*CATEGORY):
        plt.figure(figsize=[17, 3])
        plt.subplot(1, 5, 1)
        fig = plt.imshow(tst_image[i, :].reshape([IMG_SIZE, IMG_SIZE]), vmin=0, vmax=255, cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)    
        
        plt.subplot(1, 5, 2)
        fig = plt.imshow(label[i, :, :, 0], cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        
        plt.subplot(1, 5, 3)
        fig = plt.imshow(label[i, :, :, 1], cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        
        rslt = np.loadtxt('./result/' + str(i) + '_0.txt')
        max_val = np.max(rslt)
        min_val = np.min(rslt)
        rslt = rslt - min_val
        rslt = rslt / (max_val - min_val)
        plt.subplot(1, 5, 4)
        fig = plt.imshow(rslt, vmin=0.95, cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        rslt = np.loadtxt('./result/' + str(i) + '_1.txt')
        max_val = np.max(rslt)
        min_val = np.min(rslt)
        rslt = rslt - min_val
        rslt = rslt / (max_val - min_val)
        plt.subplot(1, 5, 5)
        fig = plt.imshow(rslt, vmin=0.95, cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        #plt.show()
        plt.savefig('./result/' + str(i) + '.png')
