import os
from SinGAN.run_train import functions
from SinGAN.run_train.manipulate import SinGAN_generate
from SinGAN.run_train.training import train
from SinGAN.run_train.config import get_arguments


'''
SinGAN，这是一个可以从单张自然图像学习的非条件性生成式模型。模型可以捕捉给定图像中各个小块内的内在分布，接着就能够生成带有和给定图像中的视觉内容相同的高质量且多样的新图像。
SinGAN的结构是多个全卷积GANs组成的金字塔，这些全卷积GANs都负责学习图像中的某个小块中的数据分布，不同的GANs学习的小块的大小不同。
这种设计可以让它生成具有任意大小和比例的新图像，这些新图像在具有给定的训练图像的全局结构和细节纹理的同时，还可以有很高的可变性。
与此前的从单张图像学习GAN的研究不同的是，这个方法不仅仅可以学习图像中的纹理，而且是一个非条件性模型（也就是说它是从噪声生成图像的）。
作者们做实验让人分辨原始图像和生成的图像，结果表明很难区分两者
'''

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='../Input/Images')
    parser.add_argument('--input_name', help='input image name', default='food.jpg')
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    #
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        # 将图片读取成torch版的数据
        real = functions.read_image(opt)
        # 将图片适配尺寸
        functions.adjust_scales2image(real, opt)
        # 开始训练模型 opt 手动输入的参数
        train(opt, Gs, Zs, reals, NoiseAmp)
        # 根据模型生成图片  生成具有任意大小和比例的新图像
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
