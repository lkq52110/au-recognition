import urllib.request
import pandas as pd
import os
import requests
from sklearn.model_selection import KFold


def download_image(url, path):
    """尝试下载并保存图片到指定路径。"""
    try:
        request = urllib.request.Request(url)
        response = urllib.request.urlopen(request)
        if response.getcode() == 200:
            with open(path, 'wb') as f:
                f.write(response.read())
            return True
        else:
            print(f"Error downloading {url}: HTTP Response Code {response.getcode()}")
    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")
    return False


def main():
    # 加载数据
    data_path = '../data/EmotioNet/emotionet_facs_24600.xlsx'
    data = pd.read_excel(data_path)

    # 准备目录
    base_dir = '../data/EmotioNet'  # 设置基本目录为当前工作目录
    img_dir = os.path.join(base_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    # 设置3折交叉验证
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(kf.split(data))

    # 准备AU列
    au_columns = ['AU 1', 'AU 2', 'AU 4', 'AU 5', 'AU 6', 'AU 9', 'AU 12', 'AU 17', 'AU 20', 'AU 25', 'AU 26', 'AU 43']
    au_columns = [f"'{col}'" for col in au_columns]  # 调整列名

    # 处理每个fold
    for fold_index, (_, test_idx) in enumerate(splits, start=1):
        fold_data = data.iloc[test_idx]
        img_path_file = os.path.join(base_dir, f'img_path_fold{fold_index}.txt')
        label_file = os.path.join(base_dir, f'label_fold{fold_index}.txt')

        with open(img_path_file, 'w') as img_f, open(label_file, 'w') as label_f:
            for _, row in fold_data.iterrows():
                img_url = row['URL'].strip().rstrip("'")  # 清除两端空格和末尾的单引号
                img_name = img_url.split('/')[-1]
                local_img_path = os.path.join(img_dir, img_name)

                # 下载图片
                if download_image(img_url, local_img_path):
                    img_f.write(f"{local_img_path}\n")
                    img_f.flush()  # 强制刷新写入文件
                    label_line = ' '.join(str(row[col]) for col in au_columns)
                    label_f.write(f"{label_line}\n")
                    label_f.flush()  # 强制刷新写入文件


if __name__ == "__main__":
    main()
