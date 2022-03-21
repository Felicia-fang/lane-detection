from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import numpy as np
import cv2
'''
姓名：吴方
学号：19281022
现在可运行的为Alan_.mp4
将部分注释内容调换后可运行tunnel.mp4
'''


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


def draw_roi(img, vertices):
    try:cv2.polylines(img, vertices, True, [255, 0, 0], thickness=2)#cv2.polylines()方法用于绘制任何图像上的多边形。
    except BaseException:
        print('0-0')
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    except BaseException:
        print('0-0')


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    try:
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)  # 函数输出的直接就是一组直线点的坐标位置（每条直线用两个点表示[x1,y1],[x2,y2]）
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)  # 生成绘制直线的绘图板，黑底
        # draw_lines(line_img, lines)
        draw_lanes(line_img, lines)
        return line_img
    except BaseException:
        print('0-0')


def draw_lanes(img, lines, color=[255, 255, 0], thickness=2):
    try:
        left_lines, right_lines = [], []  # 用于存储左边和右边的直线
        global ll1, ll2, rr1, rr2
        for line in lines:  # 对直线进行分类
            for x1, y1, x2, y2 in line:
                k = (y2 - y1) / (x2 - x1)
                #tunnel
                # if k < 0:
                #     if x1 < 100 and x1 > -40:
                #         if k>-2 and k<-0.4:
                #             if k > -0.9 and k < -0.7:
                #                 left_lines.append(line)
                # elif k > 0:
                #     if (x1 > 360 and x1 < 680):
                #         if(k>0.7 and k<0.9):
                #             right_lines.append(line)
                #alan
                if k < 0:
                    if x1 < 280 and x1 > -40:
                        if k>-3.8 and k<-0.4:
                            if k > -3.8 and k < -0.4:
                                left_lines.append(line)
                elif k > 0:
                    if (x1 > 360 and x1 < 680):
                        if(k>0.4 and k<3.8):
                            right_lines.append(line)
                else:
                    continue
        global num
        if (len(left_lines) <= 0 or len(right_lines) <= 0):
            num = num + 1
            # print(num)
            if num > 100:
                return img
            else:
                cv2.line(img, ll1, ll2, color, thickness)  # 画出直线
                cv2.line(img, rr1, rr2, color, thickness)  # 画出直线
                return img
        else:
            num = 0
            # print(num)

        clean_lines(left_lines, 0.01)  # 弹出左侧不满足斜率要求的直线
        clean_lines(right_lines, 0.01)  # 弹出右侧不满足斜率要求的直线
        left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]  # 提取左侧直线族中的所有的第一个点
        left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]  # 提取左侧直线族中的所有的第二个点
        right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]  # 提取右侧直线族中的所有的第一个点
        right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]  # 提取右侧侧直线族中的所有的第二个点

        left_vtx = calc_lane_vertices(left_points, 275, img.shape[0])  # 拟合点集，生成直线表达式，并计算左侧直线在图像中的两个端点的坐标
        right_vtx = calc_lane_vertices(right_points, 300, img.shape[0])  # 拟合点集，生成直线表达式，并计算右侧直线在图像中的两个端点的坐标
        # # left_vtx = calc_lane_vertices(left_points, 100, img.shape[0])  # 拟合点集，生成直线表达式，并计算左侧直线在图像中的两个端点的坐标
        # right_vtx = calc_lane_vertices(right_points, 100, img.shape[0])  # 拟合点集，生成直线表达式，并计算右侧直线在图像中的两个端点的坐标
        ll1 = left_vtx[0]
        ll2 = left_vtx[1]
        rr1 = right_vtx[0]
        rr2 = right_vtx[1]
        cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)  # 画出直线
        cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)  # 画出直线
    except BaseException:
        print('0-0')


# 将不满足斜率要求的直线弹出
def clean_lines(lines, threshold):
    try:
        slope = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                k = (y2 - y1) / (x2 - x1)
                slope.append(k)
        while len(lines) > 0:
            mean = np.mean(slope)  # 计算斜率的平均值，因为后面会将直线和斜率值弹出
            diff = [abs(s - mean) for s in slope]  # 计算每条直线斜率与平均值的差值
            idx = np.argmax(diff)  # 计算差值的最大值的下标
            if diff[idx] > threshold:  # 将差值大于阈值的直线弹出
                slope.pop(idx)  # 弹出斜率
                lines.pop(idx)  # 弹出直线
            else:
                break
    except BaseException:
        print('0-0')


# 拟合点集，生成直线表达式，并计算直线在图像中的两个端点的坐标
def calc_lane_vertices(point_list, ymin, ymax):
    try:
        x = [p[0] for p in point_list]  # 提取x
        y = [p[1] for p in point_list]  # 提取y
        fit = np.polyfit(y, x, 1)  # 用一次多项式x=a*y+b拟合这些点，fit是(a,b)
        fit_fn = np.poly1d(fit)  # 生成多项式对象a*y+b

        xmin = int(fit_fn(ymin))  # 计算这条直线在图像中最左侧的横坐标
        xmax = int(fit_fn(ymax))  # 计算这条直线在图像中最右侧的横坐标

        return [(xmin, ymin), (xmax, ymax)]
    except BaseException:
        print(0-0)


def process_an_image(img):
    try:
        roi_vtx = np.array(
            [[[125, 324], [235, 259], [325, 259], [435, 324]]])  # 目标区域的四个点坐标，roi_vtx是一个三维的,Alan_
        # roi_vtx = np.array(
        #     [[[40,240], [133, 112], [336, 136], [475,228]]])  # 目标区域的四个点坐标，roi_vtx是一个三维的,tunnel
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 图像转换为灰度图
        #tunnel
        #ret3, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
        # 使用高斯模糊去噪声
        blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
        # 使用Canny进行边缘检测

        edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
        # 对边缘检测的图像生成图像蒙板，保留兴趣区
        roi_edges = roi_mask(edges, roi_vtx)
        # plt.imshow( roi_edges)
        # plt.show()
        line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)  # 使用霍夫直线检测，并且绘制直线
        res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)  # 将处理后的图像与原图做融合
        return res_img
    except BaseException:
        print('0-0')



num=0
print("start to process the video....")
output = 'Alan1.mp4'#ouput video
clip = VideoFileClip("Alan_.mp4")#input video
# output = 'ttt.mp4'#ouput video
# clip = VideoFileClip("tunnel.mp4")#input video
out_clip = clip.fl_image(process_an_image)#对视频的每一帧进行处理
out_clip.write_videofile(output, audio=True)#将处理后的视频写入新的视频文件