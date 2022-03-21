import random
from moviepy.editor import VideoFileClip
# import matplotlib.pyplot as plt
# import matplotlib.image as mplimg
import numpy as np
import cv2
import math
# from skimage import data, filters, color
# import matplotlib.pyplot as plt

blur_ksize = 3  # Gaussian blur kernel size
canny_lthreshold = 90  # Canny edge detection low threshold
canny_hthreshold = 180  # Canny edge detection high threshold

# Hough transform parameters
rho = 1  # rho的步长，即直线到图像原点(0,0)点的距离
theta = np.pi / 180  # theta的范围
threshold = 15  # 累加器中的值高于它时才认为是一条直线
min_line_length = 10  # 线的最短长度，比这个短的都被忽略
max_line_gap = 20  # 两条直线之间的最大间隔，小于此值，认为是一条直线


def roi_mask(img, vertices):  # img是输入的图像，verticess是兴趣区的四个点的坐标（三维的数组）
    try:
        mask = np.zeros_like(img)  # 生成与输入图像相同大小的图像，并使用0填充,图像为黑色
        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            mask_color = (255,) * channel_count  # 如果 channel_count=3,则为(255,255,255)
        else:
            mask_color = 255
        cv2.fillPoly(mask, vertices, mask_color)  # 使用白色填充多边形，形成蒙板
        masked_img = cv2.bitwise_and(img, mask)  # img&mask，经过此操作后，兴趣区域以外的部分被蒙住了，只留下兴趣区域的图像

        return masked_img
    except BaseException:
        print('0-0')


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap,ret):
    try:

        size = img.shape
        height = size[0]
        length = size[1]

        mid = int(length / 2)

        leftImage = img[0:height, 0:mid]

        rightIamge = img[0:height, mid:length]

        leftpot=choose_pot(leftImage,ret)
        rightpot = choose_pot(rightIamge, ret)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)  # 生成绘制直线的绘图板，黑底
        doc = 0
        draw_lanes(line_img, leftpot, doc)
        doc = 1
        draw_lanes(line_img, rightpot, doc)
        # plt.imshow(line_img)
        # plt.show()
        return line_img
    except BaseException:
        print('0-0')

def choose_pot(img,ret):
    try:
        pot=[]
        for i in range(320):
            for j in range(360):
                if img[i][j] > ret:
                    pot.append((i, j))
                j += 1
            i += 1
        return pot
    except BaseException:
        print('0-0')

def draw_roi(img, vertices):
    try:
        cv2.polylines(img, vertices, True, [255, 0, 0], thickness=2)
    except BaseException:
        print('0-0')


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    except BaseException:
        print('0-0')


def draw_lanes(img, pot, tt,color=[255, 0, 0], thickness=8):
    try:
        # 使用RANSAC算法估算模型
        # 迭代最大次数，每次得到更好的估计会优化iters的数值
        iters = 1000
        # 数据和模型之间可接受的差值
        sigma = 0.25
        # 最好模型的参数估计和内点数目
        best_a = 0
        best_b = 0
        pretotal = 0
        # 希望的得到正确模型的概率
        P = 0.99
        i = len(pot)
        for qq in range(iters):
            # 随机在数据中红选出两个点去求解模型
            sample_index = random.sample(range(i * 2), 2)
            x_1 = pot[sample_index[0]]
            x_2 = pot[sample_index[1]]
            y_1 = pot[sample_index[0]]
            y_2 = pot[sample_index[1]]

            # y = ax + b 求解出a，b
            a = (y_2 - y_1) / (x_2 - x_1)
            b = y_1 - a * x_1

            # 算出内点数目
            total_inlier = 0
            for index in range(i * 2):
                y_estimate = a * pot[index] + b
                if abs(y_estimate - pot[index]) < sigma:
                    total_inlier = total_inlier + 1

            # 判断当前的模型是否比之前估算的模型好
            if total_inlier > pretotal:
                iters = math.log(1 - P) / math.log(1 - pow(total_inlier / (i * 2), 2))
                pretotal = total_inlier
                best_a = a
                best_b = b

            # 判断是否当前模型已经符合超过一半的点
            if total_inlier > i:
                break

        # 用我们得到的最佳估计画图
        Y = []
        NUM = 0
        for x in pot:
            y = x * best_a + best_b
            yy = int(y)
            if tt == 1:
                x = x + 320
            Y.append((x, yy))
            NUM += 1
        cv2.line(img, Y[0], Y[NUM - 1], color, thickness)  # 画出直线
        # plt.imshow(img)
        # plt.show()

    except BaseException:
        print('0-0')

def process_an_image(img):
    try:
        roi_vtx = np.array(
            [[[125, 310], [235, 259], [325, 259], [435, 310]]])  # 目标区域的四个点坐标，roi_vtx是一个三维的
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 图像转换为灰度图
        # plt.imshow(gray)
        # plt.show()
        # 使用高斯模糊去噪声
        blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
        # 利用多级图像阈值OTSU方法二值化
        # ret:阈值，th:处理后的图片
        ret, th = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # plt.imshow(th3)
        # plt.show()
        # print(ret3)
        # roi_edges = roi_mask(th, roi_vtx)  # 对边缘检测的图像生成图像蒙板，去掉不感兴趣的区域，保留兴趣区
        line_img = hough_lines(th, rho, theta, threshold, min_line_length, max_line_gap,ret)  # 使用霍夫直线检测，并且绘制直线
        res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)  # 将处理后的图像与原图做融合
        # plt.imshow(res_img)
        # plt.show()
        return res_img
    except BaseException:
        print('0-0')

print("start to process the video....")
output = 'Alan2.mp4'#ouput video
clip = VideoFileClip("Alan_.mp4")#input video
out_clip = clip.fl_image(process_an_image)#对视频的每一帧进行处理
out_clip.write_videofile(output, audio=True)#将处理后的视频写入新的视频文件
