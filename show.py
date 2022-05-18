import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import os
from light_simulation import tube_light_generation_by_func, simple_add
from skimage import util
import cv2
import seaborn as sns

import requests

# 显示softmax柱状图
def drawbar(outputs):
    # 标题处加label和confidence
    labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    fig = plt.figure()
    data = list(zip(labels, outputs))
    data.sort(key=lambda x: x[1])
    data.reverse()
    labels, outputs = zip(*data)

    x = ['1', '1', '1', '1']
    y = [0.1, 0.1, 0.2, 0.1]
    for i in range(4):
        x[i] = labels[i]
        y[i] = outputs[i]
    plt.barh(x, y, color='green')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=18)
    # plt.ylabel('class', fontsize=15)
    plt.xlabel('softmax output', fontsize=15)
    plt.show()

# transform
test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))
    ])
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 对img进行攻击，成功后标记为suc+1
def attack(img, net, suc, path):
    delay_threhold = 20
    Q = np.asarray([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 1, 0, 0],
                    [1, 0, 1, 0],
                    [1, 0, 0, 1],
                    [0, 1, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 1]
                    ])

    clean_image = test_transform(img).unsqueeze(0)
    inputs = Variable(clean_image.cuda(), requires_grad=True)
    outputs = net(inputs)

    outputs = outputs.data.cpu()
    outputs = outputs.numpy()
    outputs = outputs[0]
    outputs = outputs - np.max(outputs)
    outputs = np.exp(outputs) / np.sum(np.exp(outputs))

    org_pred_label = np.argmax(outputs)
    min_confidence = outputs[org_pred_label]

    print("confidence:", min_confidence)
    print("pred label:", org_pred_label)

    adv_image = np.asarray(img)
    save_img = adv_image
    save_outputs = outputs

    cur_pred_label = org_pred_label
    correct_adv = True

    params_list = []  # 参数列表
    for i in range(25): # 迭代 20次
        init_v_it = [np.random.randint(380, 750), np.random.randint(0, 180), np.random.randint(0, 400),
                     np.random.randint(10, 1600)]
        params_list.append(init_v_it)# k-restart 随机初始化

    for init_v in params_list:

        for search_i in range(delay_threhold):
            q_id = np.random.randint(len(Q))
            q = Q[q_id]
            step_size = np.random.randint(1, 20)
            q = q * step_size
            for a in [-1, 1]:
                temp_q = init_v + a * q
                temp_q = np.clip(temp_q, [380, 0, 0, 10],
                                 [750, 180, 400, 1600])  # 裁剪数组到规定范围内 .clip(a,min,max,out=None)

                radians = math.radians(temp_q[1])
                k = round(math.tan(radians), 2)

                tube_light = tube_light_generation_by_func(k, temp_q[2], alpha=1, beta=temp_q[3],
                                                           wavelength=temp_q[0])
                tube_light = tube_light * 255.0
                img_with_light = simple_add(adv_image, tube_light, 1.0)
                img_with_light = np.clip(img_with_light, 0.0, 255.0).astype('uint8')
                img_with_light = Image.fromarray(img_with_light)  # array转换成image
                # print(img_with_light)
                save_img = img_with_light
                save_light = tube_light
                # img_with_light.save(os.path.join("./save/adv/eg.png"))

                # img_with_light = img_with_light.resize((224, 224), Image.BILINEAR)
                img_with_light = test_transform(img_with_light).unsqueeze(0)
                adv_inputs = Variable(img_with_light.cuda(), requires_grad=True)
                adv_outputs = net(adv_inputs)

                adv_outputs = adv_outputs.data.cpu()
                adv_outputs = adv_outputs.numpy()
                adv_outputs = adv_outputs[0]
                adv_outputs = adv_outputs - np.max(adv_outputs)
                adv_outputs = np.exp(adv_outputs) / np.sum(np.exp(adv_outputs))
                save_outputs = adv_outputs

                cur_confidence = adv_outputs[org_pred_label]
                cur_pred_label = np.argmax(adv_outputs)
                # print("adv confidence:", cur_confidence)
                # print("adv label:", cur_pred_label)

                if cur_confidence < min_confidence:
                    min_confidence = cur_confidence
                    init_v = temp_q
                    break

            if cur_pred_label != org_pred_label:
                correct_adv = False
                break

        if cur_pred_label != org_pred_label:
            correct_adv = False
            break

    if correct_adv:
        print('attack failed')
    else:
        print('attack success')
        #plt.imshow(save_img)
        #plt.title("{}: {:.3f}".format(classes[np.argmax(save_outputs)], np.max(save_outputs)), fontsize=20)
        #plt.show()

        suc = suc + 1
        '''save_img.save("{}{}-{}.png".format(path, suc, org_pred_label))
        save_light = np.clip(save_light, 0.0, 255.0).astype('uint8')
        save_light = Image.fromarray(save_light)  # array转换成image

        plt.imshow(save_light)
        plt.show()'''
        # save_img.save("./save/Imagecrop/{}-{}.png".format(suc, org_pred_label))
        # save_img.save("{}{}-{}.png".format(path, suc, org_pred_label))
        # img.save("./save/origin/{}-{}.png".format(suc, org_pred_label))
        '''print("adv confidence:", save_outputs)
        print("adv label:", np.argmax(save_outputs))
        drawbar(save_outputs)'''
    return suc

# 特定激光束参数攻击
def sim_attack(net, dataset,r,b,alpha,beta,wavelength):
    count=0
    suc=0
    for img, label in dataset:

        adv_image = np.asarray(img)
        radians = math.radians(r)
        k = round(math.tan(radians), 2)
        #def tube_light_generation_by_func(k, b, alpha, beta, wavelength, w = 400, h = 400):
        tube_light = tube_light_generation_by_func(k, b, alpha, beta,wavelength)
        tube_light = tube_light * 255.0
        img_with_light = simple_add(adv_image, tube_light, 1.0)
        img_with_light = np.clip(img_with_light, 0.0, 255.0).astype('uint8')
        img_with_light = Image.fromarray(img_with_light)  # array转换成image
        '''if count == 6:
            plt.imshow(img_with_light)
            plt.title("r={};b={};w={};λ={}".format(r,b,beta,wavelength), fontsize=20)
            plt.show()'''
        outputs=pre_img(img_with_light, net)
        if np.argmax(outputs) != label:
            suc = suc + 1
        count = count + 1
        #print(count)
        if count > 57:
            print("success rate:{:.3f}".format(suc/count))
            return suc/count


def predata():

    net = torch.load("../models/densenet10.pth")
    net.cuda()
    dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True)
    dataset_path = "../data/Imagenet"
    '''dataset_path = "../data/Imagenet_resize"
    dataset_path = "../data/iSUN"
    dataset_path = "../data/LSUN/LSUN"
    dataset_path = "../data/LSUN_resize"
    dataset = torchvision.datasets.ImageFolder(dataset_path)'''
    count = -1
    suc = 0
    for img, _ in dataset:
        # OOD 数据集中的label都是0
        count = count+1
        print(count)
        #if count <= 295: continue
        suc = attack(img, net, suc, " ")
        if count >= 97:
            return suc
# 改变图片像素值
def chg_pixels(img,num):
    img = np.array(img)
    row,col,dim = img.shape
    for i in range(num):
        x = np.random.randint(0, row)
        y = np.random.randint(0, col)
        z = np.random.randint(0, dim)
        # img[x, y, z] = img[x, y, z]/2
        img[x, y, z] = np.random.randint(0,255)

    img = Image.fromarray(img)  # array转换成image
    return img
# 计算两张灰度图像的ssim
def cal_ssim(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim

# 与随机攻击方法比较
def compare():
    path_list = os.listdir("./save/origin")
    min = 0.9

    for path in path_list:
        if path != ".ipynb_checkpoints":
            print(path)
            ori_path = "./save/origin/"+path
            pix_path = "./save/origin_pix/"+ path
            img = Image.open(ori_path).convert('RGB')
            ssim = 1
            n=200
            while ssim >= min:
                n=n+1
                p_img = chg_pixels(img, n)
                im = img.convert('L')
                pim = p_img.convert('L')
                ssim = cal_ssim(im, pim)
            print(ssim)


# 给plt image添加噪声
def addnoise(img,path):
    img = np.array(img)
    # noise_img = util.random_noise(img, mode='gaussian')
    # noise_img = util.random_noise(img, mode='salt')
    # noise_img = util.random_noise(img, mode='pepper')
    # noise_img = util.random_noise(img, mode='s&p')
    noise_img = util.random_noise(img, mode='speckle')

    noise_img = np.clip(noise_img*255, 0.0, 255.0).astype('uint8')
    # RGB图像数据若为浮点数则范围为[0, 1], 若为整型则范围为[0, 255]。

    noise_img = Image.fromarray(noise_img)  # array转换成image
    # noise_img.save("./save/noise_adv/{}".format(path))

    return noise_img

# 预测图像img的输出confidence，并显示图片，返回预测的输出
def pre_img(img, net):
    image = test_transform(img).unsqueeze(0)
    inputs = Variable(image.cuda(), requires_grad=True)
    outputs = net(inputs)

    outputs = outputs.data.cpu()
    outputs = outputs.numpy()
    outputs = outputs[0]
    outputs = outputs - np.max(outputs)
    outputs = np.exp(outputs) / np.sum(np.exp(outputs))

    '''plt.imshow(img)
    label = np.argmax(outputs)
    plt.title("{}: {:.3f}".format(classes[label], outputs[label]), fontsize=20)
    plt.show()'''

    return outputs

#测试鲁棒性
def rub():
    path_list = os.listdir("./save/origin")
    net = torch.load("../models/densenet10.pth")
    net.cuda()

    for path in path_list:
        if path != ".ipynb_checkpoints":
            '''label = path.split('-')[1]
            label = int(label.split('.')[0])
            print(label)'''

            #org_img = Image.open(os.path.join("./save/origin", path).encode("utf-8")).convert('RGB')
            #pre_img(org_img, net)

            img = Image.open(os.path.join("./save/adv", path).encode("utf-8")).convert('RGB')
            #outputs = pre_img(img, net)
            #drawbar(outputs)

            noise_img = addnoise(img, path)
            #outputs = pre_img(noise_img, net)
            #drawbar(outputs)



if __name__ == '__main__':
    # compare()
    suc = predata()
    print("suc = {:.3f}".format(suc/97))
    # rub()
    #net = torch.load("../models/densenet10.pth")
    #net.cuda()
    #dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True)
    #url = 'https://api.day.app/yvLcN8BZg5un8gyNXyhtdY/OOD_layout/'
    '''suc=np.zeros((180, 400))
    f = open("./layout.txt", 'r')
    for r in range(180):
        #if r<=129:continue
        for b in range(400):
            #if r==130 and b<=342:continue
            line = f.read()
            suc[r][b]=float(line.split('/')[0])
            suc[r][b]=sim_attack(net, dataset, r, b,alpha=1,beta=20,wavelength=400)*100
            f = open("./layout.txt", 'a')
            f.write("{}/{}/{}\n".format(suc[r][b],r,b))
            f.close()

    ax = sns.heatmap(suc,linewidths=0, xticklabels=50,yticklabels=30, cmap="YlGnBu")
    ax.set_xlabel("b", fontsize = 12)
    ax.set_ylabel("r", fontsize = 12)
    s = ax.get_figure()
    s.savefig("layout.jpg", dpi=100, bbox_inches='tight')
    plt.show()'''
    
    requests.post("https://api.day.app/yvLcN8BZg5un8gyNXyhtdY/^_^completed_suc={:.3f}".format(suc/97))