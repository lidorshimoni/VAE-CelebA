import os
import os.path
import cv2
import numpy as np
import pandas as pd


class Dataset(object):
    sample_size = None
    batch_size = None
    crop = None
    filter = None
    dims = None
    shape = [None, None, None]
    image_size = None
    data_dir = None
    ignore_image_description = None
    y_dim = None  # number of facial features
    data_file = None
    data = None
    data_y = None

    def load_data(self): raise NotImplementedError

    def get_next_batch(self): raise NotImplementedError

    def save(self, dir):
        np.save(dir + 'data.npy', self.data)
        if not self.ignore_image_description:
            np.save(dir + 'data_y.npy', self.data_y)

    def load(self, dir):
        self.data = np.load(dir + 'data.npy')
        if not self.ignore_image_description:
            self.data_y = np.load(dir + 'data_y.npy')


class CelebA(Dataset):

    def __init__(self, output_size=64, channel=3, sample_size=2e4, batch_size=64, crop=True, filter=True,
                 ignore_image_description=False,
                 data_dir='/home/lidor/Desktop/FDGAN/celebA/'):
        # self.dataname = 'CelebA'
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.crop = crop
        self.filter = filter
        self.dims = output_size * output_size
        self.shape = [output_size, output_size, channel]
        self.image_size = output_size
        self.data_dir = data_dir
        self.ignore_image_description = ignore_image_description
        self.y_dim = 5  # number of facial features
        self.data_file = 'list_attr_celeba.csv'
        self.data = None
        self.data_y = None

    def load_data(self):

        images_dir = os.path.join(self.data_dir, 'img_align_celeba/img_align_celeba')

        X = []
        y = []

        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        data = pd.read_csv(os.path.join(self.data_dir, self.data_file))

        i = 0
        count = 0
        print('\n===LOADING DATA===')
        while count < self.sample_size:
            img = data['image_id'][i]
            print('\rLoading: {}- {}/{}'.format(img, count, self.sample_size), end='')

            image = cv2.imread(os.path.join(images_dir, img))
            if self.crop:
                h, w, c = image.shape
                # crop 4/6ths of the image
                cr_h = h // 6
                cr_w = w // 6
                crop_image = image[cr_h:h - cr_h, cr_w:w - cr_w]
                image = crop_image
            image = cv2.resize(image, (self.image_size, self.image_size)).astype(np.float32)
            face = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
            if type(face) is np.ndarray:
                if not self.ignore_image_description:
                    features = np.zeros(self.y_dim)
                    features[0] = int(data['Black_Hair'][i])  # Black hair
                    features[1] = int(data['Brown_Hair'][i])  # Brown hair
                    features[2] = int(data['Blond_Hair'][i])  # Blonde hair
                    features[3] = int(data['Male'][i])  # Male
                    features[4] = int(
                        data['No_Beard'][i]) * -1  # Beard (invert because in dataset, positive 1 represents no beard)
                    features = list(map(lambda x: x if x != -1 else 0, features))
                    if sum([1 for i in features[:3] if i == 1]) == 1:
                        X.append(image)
                        y.append(features)
                        count += 1
                else:
                    X.append(image)
                    count += 1
            i += 1
        seed = 547
        if not self.ignore_image_description:
            print('\n\n===DATA STATS===')
            print('Black Hair: ', sum([1 for i in y if i[0] == 1]))
            print('Brown Hair: ', sum([1 for i in y if i[1] == 1]))
            print('Blonde Hair: ', sum([1 for i in y if i[2] == 1]))
            print('Male: ', sum([1 for i in y if i[3] == 1]))
            print('Beard: ', sum([1 for i in y if i[4] == 1]))
            np.random.seed(seed)
            np.random.shuffle(y)
            y = np.array(y)

        X = np.array(X)
        np.random.seed(seed)
        np.random.shuffle(X)

        self.data = X / 255
        self.data_y = y

    def get_next_batch(self, iter_num):
        ro_num = self.sample_size // self.batch_size - 1

        if iter_num % ro_num == 0:
            length = len(self.data)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.data = np.array(self.data)
            self.data = self.data[perm]
            self.data_y = np.array(self.data_y)
            self.data_y = self.data_y[perm]
        if not self.ignore_image_description:
            return self.data[int(iter_num % ro_num) * self.batch_size: int(iter_num % ro_num + 1) * self.batch_size],\
                   self.data_y[int(iter_num % ro_num) * self.batch_size: int(iter_num % ro_num + 1) * self.batch_size]

        return self.data[int(iter_num % ro_num) * self.batch_size: int(iter_num % ro_num + 1) * self.batch_size]

    def text_to_vector(self, text):
        text = text.lower()
        key_words = ['black',
                     'brown',
                     'blonde',
                     'male',
                     'beard']
        vec = np.ones(self.y_dim) * -1
        for i, key in enumerate(key_words, 0):
            if key in text:
                vec[i] = 1
        batch_vector = np.tile(vec, (self.batch_size, 1))
        return batch_vector

    def save(self, dir):
        return super().save(dir)

    def load(self, dir):
        return super().load(dir)


# TODO:

class Anime(Dataset):
    def __init__(self, output_size=64, channel=3, sample_size=2e4, batch_size=64, crop=True, filter=True,
                 ignore_image_description=True,
                 data_dir='/home/lidor/Desktop/FDGAN/anime/data'):
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.filter = filter
        self.dims = output_size * output_size
        self.shape = [output_size, output_size, channel]
        self.image_size = output_size
        self.data_dir = data_dir
        self.ignore_image_description = ignore_image_description
        self.data = None

    def load_data(self):

        images_dir = self.data_dir

        X = []
        count = 1
        print('\n===LOADING DATA===')
        err = 0
        while count < self.sample_size+1:
            print('\rLoading: {}- {}/{}'.format(count + err, count, self.sample_size), end='\r')
            try:
                image = cv2.imread(os.path.join(images_dir, str(count + err) + ".png"))
                if image is None:
                    raise Exception
            except Exception as e:
                err += 1
                continue
            X.append(image)
            count += 1

        seed = 547
        X = np.array(X)
        np.random.seed(seed)
        np.random.shuffle(X)

        self.data = X / 255


    def get_next_batch(self, iter_num):
        ro_num = self.sample_size // self.batch_size - 1

        if iter_num % ro_num == 0:
            length = len(self.data)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.data = np.array(self.data)
            self.data = self.data[perm]

        return self.data[int(iter_num % ro_num) * self.batch_size: int(iter_num % ro_num + 1) * self.batch_size]

    def save(self, dir):
        return super().save(dir)

    def load(self, dir):
        return super().load(dir)
