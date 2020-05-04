# -*- coding: utf-8 -*-

# Загрузка модулей
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d, Rbf
from skimage.draw import line
from skimage.transform import rescale
import numpy as np
import tensorflow as tf


class DataGen:
    # Конструктор
    def __init__(self,
                 # стартовый(!) размер изобаражения, который потом увеличивается
                 # в 2 ** (image_updownsampling[0] - image_updownsampling[1])
                 # раза
                 image_size=(128, 128),
                 # режим (тип эталонных данных),
                 # доступны одно или несколько значений из множества:
                 # {'scalar', 'gradient', 'binary', 'vector', 'none'}
                 mode='binary',
                 # число повышений и последующих понижений разрешения в 2 раза
                 image_updownsampling=(4, 1),
                 batch_size=4,      # размер пакета
                 frame_gap=0.2,     # отступ от краёв
                 edges_num=(2, 3),  # диапазон числа границ на одном изображении
                 # диапазон числа сглаживаемых точек для одной границы
                 edge_points_num=(0, 1),
                 # диапазон числа    угловых   точек для одной границы
                 edge_corner_num=(0, 1),
                 # диапазон контрастности границ в ключевых точках
                 edges_contrast=(0.8, 1),
                 # диапазон долей шума в изображении
                 noise_ratio=(0.0, 0.2),
                 # диапазон долей текстурных флуктуаций в изображении
                 texture_ratio=(0.0, 0.5),
                 ):
        self.image_size = image_size
        self.mode = mode
        self.image_updownsampling = image_updownsampling
        self.batch_size = batch_size
        self.frame_gap = frame_gap
        self.edges_num = edges_num
        self.edge_points_num = edge_points_num
        self.edge_corner_num = edge_corner_num
        self.edges_contrast = edges_contrast
        self.noise_ratio = noise_ratio
        self.texture_ratio = texture_ratio

    # Свойство "Размер изображения"
    @property
    def image_size(self):
        return self.__image_size

    @image_size.setter
    def image_size(self, image_size_val):
        image_size = np.ones(2, dtype=int) * np.array(image_size_val)
        self.__image_size = image_size.astype(np.int)

    # Свойство "Тип эталонных данных"
    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, mode_val):
        none_values = (None, 'none')
        if mode_val in none_values:
            self.__mode = (None,)
            return
            
        mode_tuple = (mode_val,) if isinstance(mode_val, str) \
            else tuple(mode_val)
        for mode in mode_tuple:
            if mode in none_values:
                self.__mode = (None,)
                return
        self.__mode = mode_tuple

    # Свойство "Зазор" - размер отступа случайных точек от краёв изображения
    @property
    def frame_gap(self):
        return self.__frame_gap

    # Установка зазора
    @frame_gap.setter
    def frame_gap(self, frame_gap_val):
        # Кновертация в массив numpy:
        if type(frame_gap_val) in [list, tuple]:
            gap = np.array(frame_gap_val, dtype=np.float).flatten()
        elif type(frame_gap_val) != np.ndarray:
            gap = np.array([frame_gap_val], dtype=np.float)
        else:
            gap = frame_gap_val.astype(np.float).flatten()
        if len(gap) not in [1, 2]:
            raise ValueError('Неверный тип данных frame_gap.',
                             'len(frame_gap) = %s' % (len(gap)))

        # Конвертация в НАТУРАЛЬНЫЕ значения
        for i in range(len(gap)):
            if gap[i] < 1:
                gap[i] = gap[i] * self.image_size[i]
        gap = gap.astype(np.int)

        # Расширение в случае одномерности входных данных
        if len(gap) < 2:
            gap = gap * np.ones(2, dtype=np.int)

        self.__frame_gap = gap

    @staticmethod
    def gray2rgb(images):
        images = images.copy()
        if images.shape[-1] == 1:
            return np.repeat(images, 3, -1)
        elif images.shape[-1] == 2:
            if len(images.shape) > 3:
                images = np.squeeze(images, -2)
            images_ch3 = np.zeros(list(images.shape[:-1])+[1])
            images = np.concatenate([images, images_ch3], -1)
            return images
        elif images.shape[-1] == 3:
            return images
        else:
            return gray2rgb(np.expand_dims(images,-1))
    
    @staticmethod
    def batch2vstack(batch):
        return np.reshape(batch, [-1] + list(batch.shape[2:]))

    @staticmethod
    def image_norm(image, zeros2middle=True):
        if np.any(np.isnan(image)):
            return image
        
        if zeros2middle:
            abs_image_max = np.absolute(image).max()
            if abs_image_max > 0:
                image /= abs_image_max * 2
            image += 0.5
        else:
            image -= image.min()
            image_max = image.max()
            if image_max > 0:
                image /= image_max
        
        return image
        
    @classmethod
    def batch_norm(cls, batch, zeros2middle=True):
        if zeros2middle:
            for ind in range(len(batch)):
                sub_batch = batch[ind, ...]
                sub_batch = cls.image_norm(sub_batch, np.any(sub_batch < 0))
                batch[ind, ...] = sub_batch
        else:
            for ind in range(len(batch)):
                batch[ind, ...] = cls.image_norm(batch[ind, ...], zeros2middle)
        return batch
    
    # Отображение сгенерированных данных
    def show(self, data=None):
        if data is None:
            if self.mode[0] is None:
                title = 'images only'
            else:
                title = 'images | ' + ' | '.join(self.mode)
            print(title)
            data = self.__next__()
        else:
            title = None

        if isinstance(data, (tuple, list)):
            # Если GT Есть
            images = data[0]
            several_gts = isinstance(data[1], (tuple, list))
            gts = data[1] if several_gts else [data[1]]
        else:
            # Если GT нет
            images = data
            gts = []
            
        im2show = self.batch2vstack(images)
        im2show = self.gray2rgb(im2show)
        for gt in gts:
            if isinstance(gt, (list, tuple)):
                pass
            else:
                gt2show = gt
                gt2show = self.gray2rgb(gt2show)
                gt2show = self.batch_norm(gt2show, False)
                gt2show = self.batch2vstack(gt2show)
                im2show = np.hstack([im2show, gt2show])
        plt.figure(figsize=(20, 200))
        plt.imshow(im2show, cmap='gray')
        plt.axis('off')
        plt.title(title)
        plt.show()

        return im2show

    # Отображение основной информации о генераторе
    def summary(self):
        str_with = 65
        print('=' * str_with + '\n' +
              ' ' * (str_with // 2 - 4) + 'Генератор\n' +
              '=' * str_with)
        print('                        Стартовый размер изображения :',
              self.image_size)
        print('                                Тип эталонных данных :',
              self.mode[0], end='')
        for mode in self.mode[1:]:
            print(',\n' + ' ' * 54, mode, end='')
        if len(self.mode) > 1:
            print('.')
        else:
            print('')
        print('Порядок увеличения и последующего понижения масштаба :',
              np.array(self.image_updownsampling))
        print('                                       Размер пакета :',
              self.batch_size)
        print('                                     Отступ от краёв :',
              np.array(self.frame_gap))
        print('          Диапазон числа границ на одном изображении :',
              np.array(self.edges_num))
        print(' Диапазон числа сглаживаемых точек для одной границы :',
              np.array(self.edge_points_num))
        print(' Диапазон числа    угловых   точек для одной границы :',
              np.array(self.edge_corner_num))
        print(' Диапазон  контрастности  границ  в  ключевых точках :',
              np.array(self.edges_contrast))
        print('                   Диапазон долей шума в изображении :',
              np.array(self.noise_ratio))
        print('  Диапазон долей текстурных флуктуаций в изображении :',
              np.array(self.texture_ratio))

        print('=' * str_with)

    # Генератор случайных узловых точек на изображении
    def make_curved_edge_points(self,
                                image_size=None,
                                num_curved_points=None,
                                num_coner_points=None,
                                low_high_contrast=None):
        if not image_size:
            image_size = self.image_size
        if not num_curved_points:
            num_curved_points = np.random.randint(self.edge_points_num[0],
                                                  self.edge_points_num[1] + 1)
        if not num_coner_points:
            num_coner_points = np.random.randint(self.edge_corner_num[0],
                                                 self.edge_corner_num[1] + 1)
        if not low_high_contrast:
            low_high_contrast = self.edges_contrast

        low = self.frame_gap
        high = image_size - self.frame_gap

        # Список индексов узловых точек
        num_points = num_curved_points + num_coner_points + 1
        corner_points_inds = np.arange(1, num_points)
        np.random.shuffle(corner_points_inds)
        corner_points_inds = corner_points_inds[:num_coner_points]
        corner_points_inds.sort()
        curved_lines_inds = np.zeros((2, num_coner_points + 1), int)
        curved_lines_inds[1, :-1] = corner_points_inds
        curved_lines_inds[0, 1:] = corner_points_inds
        curved_lines_inds[1, -1] = num_points
        curved_lines_inds[1, :] += 1

        # Список координат самих точек
        num_points += 1
        points = np.zeros((2, num_points))
        points[0, :] = np.random.uniform(low[0], high[0], num_points)
        points[1, :] = np.random.uniform(low[1], high[1], num_points)

        # Список значений контрастов
        contrasts = np.zeros(num_points)
        contrasts[1:-1] = np.random.uniform(low_high_contrast[0],
                                            low_high_contrast[1],
                                            num_points - 2)
        # contrasts[[0, -1]] = 0 # Обнуляем контрасты концов

        # Добавление элемента в центр отрезка, если точек всего 2
        if np.all(contrasts == 0) and True:
            curved_lines_inds += 1
            curved_lines_inds[0, 0] = 0

            new_point = points[:, 0] + \
                (points[:, 1] - points[:, 0]) * np.random.uniform()

            new_point = np.expand_dims(new_point, -1)
            points = np.hstack([points[:, :1], new_point, points[:, 1:]])

            contrasts = np.zeros(num_points + 1)
            contrasts[1] = np.random.uniform(low_high_contrast[0],
                                             low_high_contrast[1])
        return points, contrasts, curved_lines_inds

    # Генератор случайных узловых точек для всего пакета
    def make_curved_edge_points_batch(self):
        curved_edge_points_batch = []
        batch_edges_num = 0
        for batch_ind in range(self.batch_size):
            edges_num = np.random.randint(self.edges_num[0],
                                          self.edges_num[1] + 1)
            image_curved_edge_points = [self.make_curved_edge_points()
                                        for _ in range(edges_num)]
            curved_edge_points_batch.append(image_curved_edge_points)
            batch_edges_num += len(image_curved_edge_points)
        return curved_edge_points_batch, batch_edges_num

    # Создание интерполятора
    @staticmethod
    def make_interpolator(points):
        num_points = points.shape[-1]
        x = np.arange(num_points)
        if num_points > 3:
            kind = 'cubic'
        elif num_points > 2:
            kind = 'quadratic'
        else:
            kind = 'linear'
        return interp1d(x, points, kind=kind)

    # Интерполяция
    @staticmethod
    def interpolate(interpolator, steps, num_per_step=100):
        return interpolator(np.linspace(0, steps - 1, steps * num_per_step + 1))

    # Создание списка градиентов попиксельно для одной гладкой кривой
    @staticmethod
    def make_gradient_list(soft_points, soft_contrast, image_size, scale=1):
        # Размер изобажения
        image_size = image_size * scale
        # Нулевая точка кривой
        j, i = np.round(soft_points * scale).astype(int)
        # Максимальная длина списка
        d_len = np.prod(image_size)
        # Список всех градиентов попиксельно ([i, j] и [dy, dx])
        di = np.zeros(d_len, dtype=np.int)
        dj = np.zeros(d_len, dtype=np.int)
        dx = np.zeros(d_len, dtype=np.float)
        dy = np.zeros(d_len, dtype=np.float)
        # Индекс следующей записи в dxy
        d_ind = 0
        # Заполнение списка
        for ind in range(1, len(soft_contrast)):
            # Точки следующего сегмента границы
            ii, jj = line(i[ind - 1], j[ind - 1], i[ind], j[ind])
            cc = np.linspace(soft_contrast[ind - 1],
                             soft_contrast[ind],
                             len(ii) - 1, endpoint=False)
            r = soft_points[:, ind] - soft_points[:, ind - 1]
            r /= np.linalg.norm(r)
            len_line = len(cc)
            di[d_ind:d_ind + len_line] = ii[:-1]
            dj[d_ind:d_ind + len_line] = jj[:-1]
            dx[d_ind:d_ind + len_line] = r[1] * cc
            dy[d_ind:d_ind + len_line] = r[0] * cc
            d_ind += len_line
        # Удаление незаполненных элементов списка
        di = di[:d_ind]
        dj = dj[:d_ind]
        dx = dx[:d_ind]
        dy = dy[:d_ind]
        return di, dj, dx, dy

    # Создание списка градиентов попиксельно для всей кривой
    @classmethod
    def make_full_gradient_list(cls, points, contrasts,
                                curved_lines_inds,
                                image_size, scale=1):
        di_list = []
        dj_list = []
        dx_list = []
        dy_list = []
        for curved_line_inds in curved_lines_inds.T:
            key_points = points[:, curved_line_inds[0]:curved_line_inds[1]]
            key_contrast = contrasts[curved_line_inds[0]:curved_line_inds[1]]
            points_num = len(key_contrast)

            # Создаём сглаживатель последовательности точек
            points_interpolator = cls.make_interpolator(key_points)
            contrasts_interpolator = cls.make_interpolator(key_contrast)

            # Создаём кривые на основе сглаживателя
            soft_points = cls.interpolate(points_interpolator,
                                          points_num,
                                          10)

            soft_contrasts = cls.interpolate(contrasts_interpolator,
                                             points_num,
                                             10)

            di, dj, dx, dy = cls.make_gradient_list(soft_points,
                                                    soft_contrasts,
                                                    image_size,
                                                    scale)
            di_list.append(di)
            dj_list.append(dj)
            dx_list.append(dx)
            dy_list.append(dy)

        di = np.hstack(di_list)
        dj = np.hstack(dj_list)
        dx = np.hstack(dx_list)
        dy = np.hstack(dy_list)
        return di, dj, dx, dy

    # Создание градиентного поля
    def make_gradient_field(self, di, dj, dx, dy, scale=1):
        gf_size = list(self.image_size * scale) + [1, 2]
        gf = np.zeros(gf_size, dtype=np.float)
        for i, j, x, y in zip(di, dj, dx, dy):
            if (i >= 0) and (j >= 0) and (i < gf_size[0]) and (j < gf_size[1]):
                gf[i, j, :, 0] = y
                gf[i, j, :, 1] = x
        return gf

    # Создание карты краёв
    @staticmethod
    def make_edge_sides(gf):
        edge = np.zeros_like(gf)
        edge[:-1, :, :, 0] -= gf[1:, :, :, 0]
        edge[1:, :, :, 0] += gf[:-1, :, :, 0]
        edge[:, :-1, :, 1] += gf[:, 1:, :, 1]
        edge[:, 1:, :, 1] -= gf[:, :-1, :, 1]

        edge_s = edge ** 2  # Получаем квадраты
        # Получаем квадраты только положительных
        # или только отрицательных значений ...
        edge_p = edge_s * (edge > 0)
        edge_n = edge_s * (edge < 0)
        # ... чтобы получить нормы положительных и отрицательных значений.
        edge_p = edge_p.sum(-1) ** 0.5
        edge_n = edge_n.sum(-1) ** 0.5
        return edge_p - edge_n  # Возвращаем их разность

    # Масштабирование пакета:
    @staticmethod
    def batch_resize(batch, resize_scale):
        resize_scale = [resize_scale, resize_scale, 1]
        batch = np.transpose(batch, [1, 2, 3, 0])
        batch = rescale(batch, resize_scale, multichannel=True)
        batch = np.transpose(batch, [3, 0, 1, 2])
        return batch

    '''
    # Сглаживание изображений через tf
    @staticmethod
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.double),
                                  tf.TensorSpec(shape=None, dtype=tf.double),
                                  tf.TensorSpec(shape=None, dtype=tf.double),
                                  tf.TensorSpec(shape=None, dtype=tf.double),
                                  tf.TensorSpec(shape=None, dtype=tf.double),
                                  tf.TensorSpec(shape=(), dtype=tf.int64)])
    def tf_smooth(soft_edge, sharp_edge, pm, nm, em, max_iter):
        for _ in tf.range(max_iter, dtype=tf.int64):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(soft_edge)

                # Частные производные по x и y
                with tf.name_scope('Diff'):
                    dx = soft_edge[:, :-1, :, :] - soft_edge[:, 1:, :, :]
                    dy = soft_edge[:, :, :-1, :] - soft_edge[:, :, 1:, :]

                # Функция потерь как средний квадрат частных производных
                with tf.name_scope('Loss'):
                    loss = tf.reduce_mean(tf.square(dx), axis=[1, 2, 3])
                    loss += tf.reduce_mean(tf.square(dy), axis=[1, 2, 3])

            # Выделение и применение нормализованных градиентов
            grad = tape.gradient(loss, soft_edge)
            soft_edge = soft_edge - grad / tf.reduce_max(grad) / 2

            # Обновление остроты границ после размытия
            extended_soft_edge = tf.stack((soft_edge, sharp_edge), axis=-1)
            add = tf.reduce_max(extended_soft_edge, -1) * pm \
                + tf.reduce_min(extended_soft_edge, -1) * nm
            soft_edge = soft_edge * em + add
        return soft_edge

    # Сглаживание изображений
    @classmethod
    def smooth(cls, sharp_edge, soft_edge=(None,), steps=100):
        if np.any(soft_edge):
            # Увеличение разрешения изображений в 2 раза
            soft_edge = cls.batch_resize(soft_edge, 2)
        else:
            soft_edge = sharp_edge
        soft = tf.Variable(soft_edge)
        edge = tf.constant(sharp_edge)
        pm = tf.cast(edge > 0, tf.double)  # Маска положительных значений
        nm = tf.cast(edge < 0, tf.double)  # Маска отрицательных значений
        em = tf.cast(edge == 0, tf.double)  # Маска    нулевых    значений
        max_iter = tf.constant(steps, tf.int64)

        out = cls.tf_smooth(soft, edge, pm, nm, em, max_iter)
        return out.numpy()
    '''

    # Сглаживание изображений через tf
    @staticmethod
    @tf.function(input_signature=[tf.TensorSpec(shape=None,
                                                dtype=tf.double)] * 5)
    def tf_smooth(soft_edge, sharp_edge, pm, nm, em):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(soft_edge)

            # Частные производные по x и y
            with tf.name_scope('Diff'):
                dx = soft_edge[:, :-1, :, :] - soft_edge[:, 1:, :, :]
                dy = soft_edge[:, :, :-1, :] - soft_edge[:, :, 1:, :]

            # Функция потерь как средний квадрат частных производных
            with tf.name_scope('Loss'):
                loss = tf.reduce_mean(tf.square(dx), axis=[1, 2, 3])
                loss += tf.reduce_mean(tf.square(dy), axis=[1, 2, 3])

        # Выделение и применение нормализованных градиентов
        grad = tape.gradient(loss, soft_edge)
        soft_edge = soft_edge - grad / tf.reduce_max(grad) / 2

        # Обновление остроты границ после размытия
        extended_soft_edge = tf.stack((soft_edge, sharp_edge), axis=-1)
        add = tf.reduce_max(extended_soft_edge, -1) * pm \
            + tf.reduce_min(extended_soft_edge, -1) * nm
        soft_edge = soft_edge * em + add
        return soft_edge

    # Сглаживание изображений
    @classmethod
    def smooth(cls, sharp_edge, soft_edge=(None,), steps=100):
        if np.any(soft_edge):
            # Увеличение разрешения изображений в 2 раза
            soft_edge = cls.batch_resize(soft_edge, 2)
        else:
            soft_edge = sharp_edge
        soft = tf.Variable(soft_edge)
        edge = tf.constant(sharp_edge)
        pm = tf.cast(edge > 0, tf.double)  # Маска положительных значений
        nm = tf.cast(edge < 0, tf.double)  # Маска отрицательных значений
        em = tf.cast(edge == 0, tf.double)  # Маска    нулевых    значений
        for _ in range(steps):
            soft = cls.tf_smooth(soft, edge, pm, nm, em)
        return soft.numpy()
    
    # Генератор пары image, ground_truth
    def __next__(self):

        # Создание пакета точек
        curved_edge_points_batch, batch_edges_num = \
            self.make_curved_edge_points_batch()

        # Создание пары (граница, эталон границы)
        soft_edge_batch = None
        batch_full_gradient_list = [[] for _ in range(self.batch_size)]
        for image_upsampling_ind in range(self.image_updownsampling[0] + 1):

            scale = 2 ** image_upsampling_ind

            batch_sharp_edge = np.zeros((batch_edges_num,
                                         self.image_size[0] * scale,
                                         self.image_size[1] * scale,
                                         1))
            batch_gradient_field = np.zeros((batch_edges_num,
                                             self.image_size[0] * scale,
                                             self.image_size[1] * scale,
                                             1, 2))
            batch_edge_ind = 0
            
            # Перебор по всем семплам
            for batch_ind, image_curved_edge_points in \
                    enumerate(curved_edge_points_batch):
                
                # Перебор по всем кривым семпла
                for edge_ind, curved_edge_points in \
                        enumerate(image_curved_edge_points):
                    
                    # Строим градиентное поле:
                    full_gradient_list = \
                        self.make_full_gradient_list(*curved_edge_points,
                                                     self.image_size,
                                                     scale=scale)
                    gradient_field = \
                        self.make_gradient_field(*full_gradient_list,
                                                 scale=scale)
                    batch_gradient_field[batch_edge_ind, :, :, :, :] = \
                        gradient_field
                    
                    # Строим резкие границы
                    batch_sharp_edge[batch_edge_ind, :, :, :] = \
                        self.make_edge_sides(gradient_field)
                    batch_edge_ind += 1
                    
                    # Если масштаб соответствует итоговому:
                    if image_upsampling_ind == \
                            self.image_updownsampling[0] - \
                            self.image_updownsampling[1]:
                        batch_gradient_field_out = batch_gradient_field
                        batch_full_gradient_list[batch_ind] += \
                            [full_gradient_list]

            # Смягчаем границы
            soft_edge_batch = self.smooth(batch_sharp_edge,
                                          soft_edge_batch,
                                          steps=1000 // (scale ** 2))
            
        # Объединяем границы в изображения:
        scale = 2 ** self.image_updownsampling[0]
        batch_shape = (self.batch_size,
                       self.image_size[0] * scale,
                       self.image_size[1] * scale,
                       1)
        batch_images = np.zeros(batch_shape)
        
        batch_shape_out = [self.batch_size] + \
            list(batch_gradient_field_out.shape[1:4])
        batch_scalar = np.zeros(batch_shape_out)

        batch_gradient_field_out_ = batch_gradient_field_out
        batch_gradient_field_out = np.zeros(batch_shape_out + [2])

        ind_start = 0
        for batch_ind, image_curved_edge_points in \
                enumerate(curved_edge_points_batch):

            ind_delta = len(image_curved_edge_points)
            ind_end = ind_start + ind_delta
            
            batch_images[batch_ind, :, :, :] = \
                soft_edge_batch[ind_start:ind_end, :, :, :].sum(0)
            
            sub_batch_gradient_field_out = \
                batch_gradient_field_out_[ind_start:ind_end, :, :, :, :]
            
            tmp = sub_batch_gradient_field_out ** 2
            tmp = tmp.sum(-1) ** 0.5
            tmp = tmp.max(0)
            batch_scalar[batch_ind, :, :, :] = tmp
            
            batch_gradient_field_out[batch_ind, :, :, :, :] = \
                sub_batch_gradient_field_out.sum(0)

            ind_start += ind_delta

        # Понижение разрешения
        resize_scale = .5 ** self.image_updownsampling[1]
        batch_images = self.batch_resize(batch_images, resize_scale)

        # Нормализация и низкие частоты
        # bg_points_num = batch_images.shape[1] * batch_images.shape[2] // 100
        bg_points_num = 100
        bg_points = np.random.uniform(size=(self.batch_size, 2, bg_points_num))
        bg_points[:, 0, :] *= batch_images.shape[1]
        bg_points[:, 1, :] *= batch_images.shape[2]
        bg_values = np.random.uniform(size=(self.batch_size, bg_points_num))

        x = np.arange(batch_images.shape[2])
        y = np.arange(batch_images.shape[1])
        xx, yy = np.meshgrid(x, y)
        for batch_ind in range(self.batch_size):
            # Рассчёт долей
            noise_ratio = np.random.uniform(*self.noise_ratio)
            texture_ratio = np.random.uniform(*self.texture_ratio)
            signal_ratio = 1. - noise_ratio - texture_ratio
            if signal_ratio < 0:  # Если сигналу выпала отрицательная доля
                signal_ratio = 0
                noise_texture_ratio = texture_ratio + noise_ratio
                texture_ratio /= noise_texture_ratio
                noise_ratio /= noise_texture_ratio

            # Нормализация
            batch_images[batch_ind, :, :, :] -= \
                batch_images[batch_ind, :, :, :].min()
            cur_edges_max = batch_images[batch_ind, :, :, :].max()
            batch_scalar[batch_ind, :, :, :] /= cur_edges_max
            cur_signal_ratio = signal_ratio / cur_edges_max
            batch_images[batch_ind, :, :, :] *= cur_signal_ratio
            
            # Наложение фона
            bg_interpolator = Rbf(bg_points[batch_ind, 0, :],
                                  bg_points[batch_ind, 1, :],
                                  bg_values[batch_ind, :],
                                  function='quintic')
            bg = bg_interpolator(xx, yy)
            bg -= bg.min()
            bg *= texture_ratio / bg.max()
            batch_images[batch_ind, :, :, 0] += bg
            
            # Наложение шума
            noise = np.random.normal(noise_ratio/2,
                                     noise_ratio/6,
                                     batch_images.shape[1:-1])
            noise[noise < 0] = 0
            noise[noise > noise_ratio] = noise_ratio
            batch_images[batch_ind, :, :, 0] += noise

        out = []
        for mode in self.mode:
            # Модуль градиента
            if mode == 'scalar':
                out.append(batch_scalar)
            
            # Градиентное поле
            elif mode == 'gradient':
                out.append(batch_gradient_field_out)
            
            # Граница - 1, фон - 0
            elif mode == 'binary':
                batch_binary = batch_scalar.copy()
                batch_binary[batch_binary > 0] = 1
                out.append(batch_binary)
            
            # Карта сегментов
            elif mode == 'vector':
                out.append(batch_full_gradient_list)
                        
            elif mode in {None, 'none'}:
                out = []
                break
            else:
                raise('Неверный параметр', mode)
        
        if len(out) > 0:
            if len(out) == 1:
                out = out[0]
            return batch_images, out
        else:
            return batch_images


# ===================================================================
if __name__ == '__main__':
    print('TF ver:', tf.__version__)
    # Создаём экземпляр класса
    dg = DataGen(image_size=32,
                 frame_gap=.1,
                 image_updownsampling=(4, 2),
                 batch_size=6,
                 mode=('gradient', 'scalar', 'binary'))
    
    # Отобржаем детали
    dg.summary()

    # Приводим пример сгенерированных данных
    example_batch = next(dg)
    dg.show(example_batch)
