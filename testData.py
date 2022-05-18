import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import time
from addPer import addPerturbation
# 测试阶段：进行OOD检测

def test(net, ID_loader, OOD_loader, magnitude, temperature, OOD_dataset):
    criterion = nn.CrossEntropyLoss()
    t0 = time.time()
    # 分段存储；改变N值
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')
    N = 1
    if OOD_dataset == "iSUN":
        N = 8925

    print("Processing in-distribution images")
    for j, data in enumerate(ID_loader):
        # if j < 2000: continue
        images, target = data   # input(tensor)
        print("{}:  label:{}\t".format(j, target))

        inputs = Variable(images.cuda(), requires_grad=True)
        outputs = net(inputs)

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temperature, magnitude, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temperature

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)    # target

        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        '''# Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
        gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
        gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)'''
        # Adding small perturbations to images
        # tempInputs = torch.add(inputs.data, -magnitude, gradient)  # 添加扰动，inputs.data、tempInputs都是tensor

        tempInputs = addPerturbation(images, maxIndexTemp, net, j)
        outputs = net(Variable(tempInputs))
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        g1.write("{}, {}, {}\n".format(temperature, magnitude, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j + 1 - 1000, N - 1000, time.time() - t0))
            f1.close()
            g1.close()
            f1 = open("./softmax_scores/confidence_Base_In.txt", 'a')
            g1 = open("./softmax_scores/confidence_Our_In.txt", 'a')
            t0 = time.time()

        if j == N - 1:
            break

    f1.close()
    g1.close()

    t0 = time.time()
    print("Processing out-of-distribution images")
    for j, data in enumerate(OOD_loader):
        # if j < 2000: continue
        images, target = data
        print("{}:  label:{}\t".format(j, target))

        inputs = Variable(images.cuda(), requires_grad=True)
        outputs = net(inputs)

        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temperature, magnitude, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temperature

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        '''# Normalizing the gradient to binary in {0, 1}
        gradient = (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
        gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
        gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)'''
        # Adding small perturbations to images
        # tempInputs = torch.add(inputs.data, -magnitude, gradient)
        tempInputs = addPerturbation(images, maxIndexTemp, net, j)
        outputs = net(Variable(tempInputs))
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temperature, magnitude, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j + 1 - 1000, N - 1000, time.time() - t0))
            f2.close()
            g2.close()
            f2 = open("./softmax_scores/confidence_Base_Out.txt", 'a')
            g2 = open("./softmax_scores/confidence_Our_Out.txt", 'a')
            t0 = time.time()

        if j == N - 1:
            break

    f2.close()
    g2.close()


def testGau(net, ID_loader, magnitude, temperature):
    criterion = nn.CrossEntropyLoss()
    t0 = time.time()
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')

    N = 10000
    print("Processing in-distribution images")
    for j, data in enumerate(ID_loader):

        if j < 1000:
            continue
        images, target = data

        inputs = Variable(images.cuda(), requires_grad=True)
        outputs = net(inputs)

        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temperature, magnitude, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temperature

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        tempInputs = addPerturbation(images, maxIndexTemp, net, j)
        outputs = net(Variable(tempInputs))
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))

        g1.write("{}, {}, {}\n".format(temperature, magnitude, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j + 1 - 1000, N - 1000, time.time() - t0))
            f1.close()
            g1.close()
            f1 = open("./softmax_scores/confidence_Base_In.txt", 'a')
            g1 = open("./softmax_scores/confidence_Our_In.txt", 'a')
            t0 = time.time()
    f1.close()
    g1.close()

    print("Processing out-of-distribution images")
    for j, data in enumerate(ID_loader):
        if j < 1000:
            continue

        images = torch.randn(1, 3, 32, 32) + 0.5
        images = torch.clamp(images, 0, 1)
        images[0][0] = (images[0][0] - 125.3 / 255) / (63.0 / 255)
        images[0][1] = (images[0][1] - 123.0 / 255) / (62.1 / 255)
        images[0][2] = (images[0][2] - 113.9 / 255) / (66.7 / 255)

        inputs = Variable(images.cuda(), requires_grad=True)
        outputs = net(inputs)

        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temperature, magnitude, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temperature

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Adding small perturbations to images
        tempInputs = addPerturbation(images, maxIndexTemp, net) # 有改动
        outputs = net(Variable(tempInputs))
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temperature, magnitude, np.max(nnOutputs)))

        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j + 1 - 1000, N - 1000, time.time() - t0))
            f2.close()
            g2.close()
            f2 = open("./softmax_scores/confidence_Base_Out.txt", 'a')
            g2 = open("./softmax_scores/confidence_Our_Out.txt", 'a')
            t0 = time.time()

        if j == N - 1:
            break
    f2.close()
    g2.close()

def testUni(net, ID_loader, magnitude, temperature):
    criterion = nn.CrossEntropyLoss()
    t0 = time.time()
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')

    N = 10000
    print("Processing in-distribution images")
    for j, data in enumerate(ID_loader):

        if j < 1000:
            continue
        images, target = data

        inputs = Variable(images.cuda(), requires_grad=True)
        outputs = net(inputs)

        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temperature, magnitude, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temperature

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        tempInputs = addPerturbation(images, maxIndexTemp, net)  # 有改动
        outputs = net(Variable(tempInputs))
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))

        g1.write("{}, {}, {}\n".format(temperature, magnitude, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j + 1 - 1000, N - 1000, time.time() - t0))
            f1.close()
            g1.close()
            f1 = open("./softmax_scores/confidence_Base_In.txt", 'a')
            g1 = open("./softmax_scores/confidence_Our_In.txt", 'a')
            t0 = time.time()
    f1.close()
    g1.close()

    print("Processing out-of-distribution images")
    for j, data in enumerate(ID_loader):
        if j < 1000:
            continue

        images = torch.rand(1, 3, 32, 32)
        images[0][0] = (images[0][0] - 125.3 / 255) / (63.0 / 255)
        images[0][1] = (images[0][1] - 123.0 / 255) / (62.1 / 255)
        images[0][2] = (images[0][2] - 113.9 / 255) / (66.7 / 255)

        inputs = Variable(images.cuda(), requires_grad=True)
        outputs = net(inputs)

        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temperature, magnitude, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temperature

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Adding small perturbations to images
        tempInputs = addPerturbation(images, maxIndexTemp, net)  # 有改动
        outputs = net(Variable(tempInputs))
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temperature, magnitude, np.max(nnOutputs)))

        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j + 1 - 1000, N - 1000, time.time() - t0))
            f2.close()
            g2.close()
            f2 = open("./softmax_scores/confidence_Base_Out.txt", 'a')
            g2 = open("./softmax_scores/confidence_Our_Out.txt", 'a')
            t0 = time.time()

        if j == N - 1:
            break
    f2.close()
    g2.close()

