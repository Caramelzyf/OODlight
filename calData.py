import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
from PIL import Image
import os
# from scipy import misc
# 数据集处理
import addPer
import show

transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
])
criterion = nn.CrossEntropyLoss()

def drawhist(score, title):
    plt.figure(figsize=(10, 5))
    plt.title(title,fontsize='large')
    plt.hist(score, bins=100, edgecolor='k')
    '''plt.xticks(bins, bins)
    for num, bin in zip(nums, bins):
        plt.annotate(num, xy=(bin, num), xytext=(bin + 1.5, num + 0.5))'''
    plt.show()

def testData(net, CUDA_DEVICE, datasetIn, datasetOOD, dataName, noiseMagnitude1, temper):
    t0 = time.time()
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')
    N = 113
    min = 0.99
    print("Processing in-distribution images")
########################################In-distribution###########################################
    count = -1
    IDscore = []

    for img, label in datasetIn:
        count += 1
        # if count < 1000:continue
        # attack:
        # img = addPer.addPerturbation(img, label, net)

        img = Image.open(os.path.join("./save/adv", img).encode("utf-8")).convert('RGB')

        # 添加噪声
        # img = show.addnoise(img," ")
        '''ssim = 1
        n = 100
        p_img = img
        while ssim >= min:
            n = n + 5
            p_img = show.chg_pixels(img, n)
            im = img.convert('L')
            pim = p_img.convert('L')
            ssim = show.cal_ssim(im, pim)
            image = transform(p_img).unsqueeze(0)'''

        image = transform(img).unsqueeze(0)
        inputs = Variable(image.cuda(CUDA_DEVICE), requires_grad=True)
        outputs = net(inputs)

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)  # 添加扰动，tempInputs是一个张量tensor

        outputs = net(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        # np.max(nnOutputs) 作为OOD score
        IDscore.append(np.max(nnOutputs))

        if count % 10 == 9:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count+1, N, time.time()-t0))
            t0 = time.time()
        
        if count == N - 1:
            f1.close()
            g1.close()
            break
    title = "ID样本的OOD分数分布直方图"
    #drawhist(IDscore, title)

    t0 = time.time()
    print("Processing out-of-distribution images")
###################################Out-of-Distributions#####################################
    count = -1
    OODscore = []

    for img in datasetOOD:
        count += 1
        # if count < 1000:continue
        img = Image.open(os.path.join("./save/Imagecrop", img).encode("utf-8")).convert('RGB')
        # 添加噪声
        # img = show.addnoise(img," ")
        '''ori_path = "../data/Imagenet/test/" + img
        img = Image.open(ori_path).convert('RGB')
        ssim = 1
        n = 100
        p_img = img
        while ssim >= min:
            n = n + 1
            p_img = show.chg_pixels(img, n)
            im = img.convert('L')
            pim = p_img.convert('L')
            ssim = show.cal_ssim(im, pim)
            image = transform(p_img).unsqueeze(0)'''
        # print(ssim)

        image = transform(img).unsqueeze(0)
        inputs = Variable(image.cuda(CUDA_DEVICE), requires_grad=True)
        outputs = net(inputs)

        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temper
  
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        # np.max(nnOutputs) 作为OOD score
        OODscore.append(np.max(nnOutputs))

        if count % 10 == 9:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count+1, N, time.time()-t0))
            t0 = time.time()

        if count == N-1:
            f2.close()
            g2.close()
            break
    title = "OOD样本的OOD分数分布直方图"
    #drawhist(OODscore, title)



def testGaussian(net, CUDA_DEVICE, datasetIn, noiseMagnitude1, temper):
    t0 = time.time()
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')
    N = 100
########################################In-Distribution###############################################
    print("Processing in-distribution images")
    count = -1

    for img, label in datasetIn:
        count += 1
        # if count < 1000:continue
        # attack:
        # img = Image.open(os.path.join("./save/adv", img).encode("utf-8")).convert('RGB')

        image = transform(img).unsqueeze(0)
        inputs = Variable(image.cuda(CUDA_DEVICE), requires_grad=True)
        outputs = net(inputs)

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
        gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
        gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)  # 添加扰动，tempInputs是一个张量tensor

        outputs = net(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        if count % 10 == 9:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count + 1, N, time.time() - t0))
            t0 = time.time()

        if count == N - 1:
            f1.close()
            g1.close()
            break

    
    
########################################Out-of-Distribution######################################
    print("Processing out-of-distribution images")
    count= -1
    while count <= N:
        count=count+1
        noise = np.random.normal(0.5, 1,(32,32,3))
        img = np.clip(noise * 255, 0.0, 255.0).astype('uint8')
        img = Image.fromarray(img)  # array转换成image
        #attack:
        # images = show.attack(img, net, 0," ")

        images= transform(img).unsqueeze(0)
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = net(inputs)

        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temper
        
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        
        if count % 10 == 9:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count+1, N, time.time()-t0))
            t0 = time.time()

        if count== N-1:
            f2.close()
            g2.close()
            break




def testUniforam(net, CUDA_DEVICE, datasetIn,  noiseMagnitude1, temper):
    t0 = time.time()
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')
########################################In-Distribution###############################################
    N = 100
    print("Processing in-distribution images")
    count = -1

    for img, label in datasetIn:
        count += 1
        # if count < 1000:continue
        # attack:
        img = Image.open(os.path.join("./save/adv", img).encode("utf-8")).convert('RGB')

        image = transform(img).unsqueeze(0)
        inputs = Variable(image.cuda(CUDA_DEVICE), requires_grad=True)
        outputs = net(inputs)

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
        gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
        gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)  # 添加扰动，tempInputs是一个张量tensor

        outputs = net(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        if count % 10 == 9:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count + 1, N, time.time() - t0))
            t0 = time.time()

        if count == N - 1:
            f1.close()
            g1.close()
            break


########################################Out-of-Distribution######################################
    print("Processing out-of-distribution images")
    count = -1
    while count <= N:
        count = count + 1
        noise = np.random.rand(32, 32, 3)

        img = np.clip(noise * 255, 0.0, 255.0).astype('uint8')
        img = Image.fromarray(img)  # array转换成image
        # attack:
        images = show.attack(img, net, 0," ")

        images = transform(images).unsqueeze(0)
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad=True)
        outputs = net(inputs)

        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
        gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
        gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
        outputs = net(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        if count % 10 == 9:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count + 1, N, time.time() - t0))
            t0 = time.time()

        if count == N - 1:
            f2.close()
            g2.close()
            break










