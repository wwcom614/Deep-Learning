import os
#数据集路径
IMAGES_DIR = 'D:/ideaworkspace/TensorFlow/Captcha_MultiTask/captcha/images/'

#获取所有验证码图片全路径名+文件名存放在list中
def _get_path_imagename_list(IMAGES_DIR):
    path_imagename_list = []
    for filename in os.listdir(IMAGES_DIR):
        #获取文件路径
        path = os.path.join(IMAGES_DIR, filename)
        path_imagename_list.append(path)
    return path_imagename_list

path_imagename = _get_path_imagename_list(IMAGES_DIR)[0]

labels = path_imagename.split('/')[-1]

print(labels)