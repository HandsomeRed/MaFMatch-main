# 加载.h5图片数据

## import
import os
from PIL import Image
import numpy as np
import h5py as h5

## functions
def save_image(file, groups):
    '''
    将.h5/group下的array转换为灰度图像并存储
    :param group: .h5文件包中的组名(key值)
    :param save_path: 图像存储的路径
    :return: None
    '''
    for group in groups:
        print(group + " 组：")
        image_path = input("请输入第一组图片要存储的路径(不存储输入N)：")
        counter = 0
        for i in file[group][:]:
            if image_path == "N":
                continue
            else:
                makedir(image_path)
                save_path = image_path + str(counter) + ".png"
                image = np.array(i)
                image *= 255  # 变换为0-255的灰度值
                image = Image.fromarray(image)
                image = image.convert('L')  # 灰度为L，彩色为RGB’
                image.save(save_path)
                print(counter + 1)
                counter += 1
    print("done!")


def makedir(dir_path):
    '''
    创建文件夹
    :param dir_path: 文件夹路径
    :return: None
    '''
    isExists = os.path.exists(dir_path)
    if not isExists:  # 判断如果文件不存在,则创建
        os.makedirs(dir_path)


def load_h5(file_path):
    '''
    加载.h5数据
    :param file_path: .h5文件路径
    :return: groups, file
    '''
    file = h5.File(file_path, "r")
    groups = [key for key in file.keys()]
    print("该文件共有以下几组：", groups)
    return groups, file

## main
if __name__ == "__main__":
    h5_file = input("输入.h5文件路径：")
    groups, file = load_h5(h5_file)
    save_image(file, groups)