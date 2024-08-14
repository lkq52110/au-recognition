def merge_parts(part1, part2, part3, fold_num):
    # 合并训练集
    train_img_paths = []
    train_labels = []
    if fold_num == 1:
        train_img_paths.extend(part1[0] + part2[0])
        train_labels.extend(part1[1] + part2[1])
    elif fold_num == 2:
        train_img_paths.extend(part1[0] + part3[0])
        train_labels.extend(part1[1] + part3[1])
    elif fold_num == 3:
        train_img_paths.extend(part2[0] + part3[0])
        train_labels.extend(part2[1] + part3[1])

    # 合并测试集
    test_img_paths = part3[0] if fold_num == 1 else (part2[0] if fold_num == 2 else part1[0])
    test_labels = part3[1] if fold_num == 1 else (part2[1] if fold_num == 2 else part1[1])

    return train_img_paths, train_labels, test_img_paths, test_labels

def write_to_file(img_paths, labels, img_file, label_file):
    with open(img_file, 'w') as img_f, open(label_file, 'w') as label_f:
        for img_path, label in zip(img_paths, labels):
            img_f.write(f"{img_path}\n")
            label_f.write(f"{label}\n")

# 读取各部分数据
part1_img_paths = ['../data/EmotioNet/list/img_path_fold1.txt']
part1_labels = ['../data/EmotioNet/list/label_fold1.txt']
part2_img_paths = ['../data/EmotioNet/list/img_path_fold2.txt']
part2_labels = ['../data/EmotioNet/list/label_fold2.txt']
part3_img_paths = ['../data/EmotioNet/list/img_path_fold3.txt']
part3_labels = ['../data/EmotioNet/list/label_fold3.txt']

# 读取每个part的图片路径文件与对应标签文件，并将数据存入相应的列表中

# 合并数据生成fold
for fold_num in range(1, 4):
    train_img_paths, train_labels, test_img_paths, test_labels = merge_parts((part1_img_paths, part1_labels),
                                                                             (part2_img_paths, part2_labels),
                                                                             (part3_img_paths, part3_labels),
                                                                             fold_num)

    # 写入训练集文件
    train_img_file = f'Emotio_train_img_path_fold{fold_num}.txt'
    train_label_file = f'Emotio_train_label_fold{fold_num}.txt'
    write_to_file(train_img_paths, train_labels, train_img_file, train_label_file)

    # 写入测试集文件
    test_img_file = f'Emotio_test_img_path_fold{fold_num}.txt'
    test_label_file = f'Emotio_test_label_fold{fold_num}.txt'
    write_to_file(test_img_paths, test_labels, test_img_file, test_label_file)
