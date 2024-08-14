import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os


def process_images(input_txt_path, output_dir):
    # 初始化MTCNN检测器
    detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker=4, accurate_landmark=False)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开输入txt文件
    with open(input_txt_path, 'r') as file:
        for img_path in file:
            img_path = img_path.strip()  # 移除路径两端的空白字符

            print("处理图片：" + img_path)

            # 加载图像
            img = cv2.imread(img_path)

            # 运行检测器
            results = detector.detect_face(img)

            if results is not None:
                total_boxes, points = results

                # 提取对齐后的人脸图像
                chips = detector.extract_image_chips(img, points, 256, 0.37)
                for i, chip in enumerate(chips):
                    # 构造输出路径并保存处理后的图像
                    filename = os.path.basename(img_path)  # 提取文件名
                    output_path = os.path.join(output_dir, filename)
                    cv2.imwrite(output_path, chip)
            else:
                # 图片检测失败，将失败的图片路径写入错误日志文件
                with open('Emotio_error2.txt', 'a') as error_file:
                    error_file.write(img_path + '\n')


def main():

    input_txt_path = r'../data/EmotioNet/img_list.txt'  # 你的输入txt文件路径
    output_dir = '../data/EmotioNet/img-mtcnn'  # 你的输出目录路径

    process_images(input_txt_path, output_dir)


if __name__ == '__main__':
    main()
