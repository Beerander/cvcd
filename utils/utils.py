# -*-coding:gb2312-*-
import random
import os
import shutil
import pandas as pd


def CopyFile(imageDir, test_rate, save_test_dir, save_train_dir):
    """
    ��Թ̶���������split
    imageDir: �ض���������ͼ���ڼ�����е�λ��
    test_rate: copy��ͼƬ��Ŀ��ռ�ܵı���
    save_xxx_dir: �ƶ���ͼƬ�����λ��
    """
    image_number = len(imageDir)  # ͼƬ����Ŀ
    test_number = int(image_number * test_rate)  # Ҫ�ƶ���ͼƬ��Ŀ
    print(f'{save_train_dir}����:{image_number}, test:{test_number}, train:{image_number - test_number}')
    test_samples = random.sample(imageDir, test_number)  # �����ȡ�б�imageDir����ĿΪtest_number��Ԫ��
    # copyͼ��Ŀ���ļ���
    if os.path.exists(save_test_dir) or os.path.exists(save_train_dir):
        print("data already exists, reloading...")
        shutil.rmtree(save_test_dir)
        shutil.rmtree(save_train_dir)
    os.makedirs(save_test_dir)
    os.makedirs(save_train_dir)
    for i, test_sample in enumerate(test_samples):
        shutil.copy(test_sample, save_test_dir + test_sample.split("/")[-1])  # x.split("/")[-1]ȡ�ļ���
    for train_img in imageDir:
        if train_img not in test_samples:
            shutil.copy(train_img, save_train_dir + train_img.split("/")[-1])
    return image_number, test_number


def train_test_split(file_path, save_train_root, save_test_root, ratio=0.2):
    """
    file_path: ԭʼ·��
    test_rate: �ָ����
    """
    file_dirs = os.listdir(file_path)
    origion_paths = []
    save_test_dirs = []
    save_train_dirs = []
    for path in file_dirs:
        origion_paths.append(file_path + "/" + path + "/")
        save_train_dirs.append(save_train_root + "/train/" + path + "/")
        save_test_dirs.append(save_test_root + "/test/" + path + "/")
    for i, origion_path in enumerate(origion_paths):
        image_list = os.listdir(origion_path)  # ���ԭʼ·���µ�����ͼƬ��name��Ĭ��·���¶���ͼƬ��
        image_Dir = []
        for x, y in enumerate(image_list):
            image_Dir.append(os.path.join(origion_path, y))
        CopyFile(image_Dir, ratio, save_test_dirs[i], save_train_dirs[i])
    print("done")


def arrays_to_csv(csv_path, train_accs, train_losss, test_accs, test_losss, train_times):
    df = pd.DataFrame(columns=['epochs', 'train loss', 'train acc',  'val loss', 'val acc', 'time(s)'])
    df.to_csv(csv_path, index=False)
    for i in range(len(train_accs)):
        sing_metric = [f'{i}', f'{train_losss[i]}', f'{train_accs[i]}',
                       f'{test_losss[i]}', f'{test_accs[i]}', f'{train_times[i]}']
        data = pd.DataFrame([sing_metric])
        data.to_csv(csv_path, mode='a', header=False, index=False)


def plot_result(net_name, train_accs, train_losss, test_accs, test_losss):
    plt.subplot(1, 2, 1)
    plt.title('Train and Validation Accuracy')
    plt.plot(np.arange(num_epochs), np.array(test_accs))
    plt.plot(np.arange(num_epochs), np.array(train_accs))
    plt.subplot(1, 2, 2)
    plt.title('Train and Validation Loss')
    plt.plot(np.arange(num_epochs), np.array(test_losss))
    plt.plot(np.arange(num_epochs), np.array(train_losss))
    plt.savefig(f'./logs/{net_name}.png')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('../logs/resnext101.csv')
    df_array = df.to_numpy()
    for j, i in enumerate(df_array.T[-1]):
        df_array.T[-1][j] = eval(i)[j]
    df = pd.DataFrame(df_array)
    df.to_csv('../logs/resnext101.csv', index=False)
