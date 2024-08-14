"""
http://dlib.net/
"""
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import os

# 初始化dlib的人脸检测器（基于HOG），并创建面部特征点预测器和面部对齐器
detector = dlib.get_frontal_face_detector()

# You can get the .dat at http://dlib.net/
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

# 输入和输出目录
input_dir = '../data/EmotioNet/img'
output_dir = '../data/EmotioNet/img-dlib'

# 错误输出文件路径
error_output_path = 'Emotio_error.txt'

# 遍历输入目录中的每个图片文件
for filename in os.listdir(input_dir):
    img_path = os.path.join(input_dir, filename)

    print("处理图片：" + img_path)

    # 加载输入图像，调整大小并转换为灰度图
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 在灰度图像中检测人脸
    rects = detector(gray, 2)

    # 如果没有检测到人脸
    if len(rects) == 0:
        # 将没有检测到人脸的图片路径写入到错误输出文件中
        with open(error_output_path, 'a') as error_file:
            error_file.write(img_path + '\n')
        continue  # 继续处理下一张图片

    # 循环遍历每个检测到的人脸
    for rect in rects:
        # 使用面部特征点进行人脸对齐
        faceAligned = fa.align(image, gray, rect)

        # 构造对齐后人脸图像的输出路径
        output_path = os.path.join(output_dir, filename)

        # 保存对齐后的人脸图像
        cv2.imwrite(output_path, faceAligned)
