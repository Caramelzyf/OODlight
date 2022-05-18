import torch
import numpy as np
import math
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from light_simulation import tube_light_generation_by_func, simple_add

delay_threhold = 3
Q = np.asarray([[1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [1,1,0,0],
        [1,0,1,0],
        [1,0,0,1],
        [0,1,1,0],
        [0,1,0,1],
        [0,0,1,1]
        ])

# 数据集处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
])


def addPerturbation(image, label, model):  # 为单一样本添加扰动, 输入要处理的img，输出攻击后的img
    clean_image = transform(image).unsqueeze(0)
    inputs = Variable(clean_image.cuda(), requires_grad=True)
    outputs = model(inputs)

    outputs = outputs.data.cpu()
    outputs = outputs.numpy()
    outputs = outputs[0]
    outputs = outputs - np.max(outputs)
    outputs = np.exp(outputs) / np.sum(np.exp(outputs))

    min_confidence = outputs[label]
    org_pred_label = np.argmax(outputs)

    # print('org_pred_label:', org_pred_label)

    adv_image = np.asarray(image)
    save_img = adv_image
    # adv_image.save("./save/origin/{}.png".format(index))

    cur_pred_label = org_pred_label
    correct_adv = org_pred_label == label

    # cur_search = 0
    params_list = []  # 参数列表
    for i in range(10):
        init_v_it = [np.random.randint(380, 750), np.random.randint(0, 180), np.random.randint(0, 400),
                     np.random.randint(10, 1600)]
        params_list.append(init_v_it)

    for init_v in params_list:

        for search_i in range(delay_threhold):  # 迭代
            q_id = np.random.randint(len(Q))  # k-restart 随机初始化
            q = Q[q_id]
            step_size = np.random.randint(1, 20)
            q = q * step_size
            for a in [-1, 1]:
                # cur_search += 1
                # print(a*q)
                temp_q = init_v + a * q
                temp_q = np.clip(temp_q, [380, 0, 0, 10], [750, 180, 400, 1600])  # 裁剪数组到规定范围内 .clip(a,min,max,out=None)

                radians = math.radians(temp_q[1])
                k = round(math.tan(radians), 2)

                tube_light = tube_light_generation_by_func(k, temp_q[2], alpha=1, beta=temp_q[3], wavelength=temp_q[0])
                tube_light = tube_light * 255.0
                img_with_light = simple_add(adv_image, tube_light, 1.0)
                img_with_light = np.clip(img_with_light, 0.0, 255.0).astype('uint8')
                # RGB图像数据若为浮点数则范围为[0, 1], 若为整型则范围为[0, 255]。
                img_with_light = Image.fromarray(img_with_light)  # array转换成image
                # if args.save:
                # img_with_light.save("./save/adv/eg.png")
                save_img = img_with_light

                img_with_light = transform(img_with_light).unsqueeze(0)
                adv_inputs = Variable(img_with_light.cuda(), requires_grad=True)
                adv_outputs = model(adv_inputs)

                adv_outputs = adv_outputs.data.cpu()
                adv_outputs = adv_outputs.numpy()
                adv_outputs = adv_outputs[0]
                adv_outputs = adv_outputs - np.max(adv_outputs)
                adv_outputs = np.exp(adv_outputs) / np.sum(np.exp(adv_outputs))
                save_outputs = adv_outputs

                cur_confidence = adv_outputs[label]
                cur_pred_label = np.argmax(adv_outputs)

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

    return save_img

