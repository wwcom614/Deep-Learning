# coding: utf-8

# 先要安装captcha：pip install captcha。如果远端拒绝需要翻墙
from captcha.image import ImageCaptcha

import random
import sys

#验证码候选列表，生成验证码可以是纯数字或者加字母
number = ['0','1','2','3','4','5','6','7','8','9']
# alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

#code_list是验证码候选列表，如果带字符，code_list = number + alphabe + ALPHABET
#captcha_size是生成的一张验证码图片中的字符个数，默认4个
def random_captcha_text(code_list = number, captcha_size = 4):
    captcha_text_list = []
    for i in  range(captcha_size):
        #随机从验证码候选列表code_list中选取字符，放入captcha_size位的验证码列表captcha_text_list中
        captcha_text_list.append(random.choice(code_list))
    return captcha_text_list

#将验证码文本captcha_text_list生成验证码图片
def gen_captcha_image():
    image = ImageCaptcha()
    #获取随机生成的验证码
    captcha_text_list = random_captcha_text()
    #将列表转换成字符串
    captcha_text = ''.join(captcha_text_list)
    #生成验证码图片
    image.generate(captcha_text)
    #手工先将captchaimages文件夹创建好
    image.write(captcha_text, 'captcha\\images\\' + captcha_text + '.jpg')

#主函数，调用gen_captcha_image生成随机验证码
#因为随机生成会有重复，生成时会有覆盖
num = 10000
if __name__ == '__main__':
    for i in range(num):
        gen_captcha_image()
        sys.stdout.write('\r>> Creating image %d/%d' % (i+1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    print("\n随机验证码生成完毕！")