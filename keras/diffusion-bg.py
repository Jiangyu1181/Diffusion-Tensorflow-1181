# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Author: Jiangyu1181
# Github: https://github.com/Jiangyu1181
# ----------------------------------------------------------------------------------------------------------------------
# Description: "Diffusion model utils"
# Reference Paper: ""
# Reference Code: ""
# ----------------------------------------------------------------------------------------------------------------------

import math

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

channel_axis = 1 if K.image_data_format() == 'channels_first' else -1


def positional_embedding(t, filters):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    :param t: int
    :param filters:
    :return: (1, embedding_dim(dim))
    """
    half_dim = filters // 2
    emb = tf.math.log(10000.) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    emb = tf.cast(t, dtype=tf.float32) * emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=channel_axis)
    if filters % 2 == 1:
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    return emb


def t_emb_mlp(t_emb, filters):
    t_emb = Dense(units=filters)(t_emb)
    t_emb = gelu(x=t_emb)
    t_emb = Dense(units=filters)(t_emb)
    return t_emb


def gelu(x, approximate=False):
    if approximate:
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))


def cosine_beta_schedule(time_steps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = time_steps + 1
    x = tf.cast(tf.linspace(0, time_steps, steps), tf.float32)

    alphas_cumprod = tf.cos(((x / time_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return tf.clip_by_value(betas, 0, 0.999)


def get_linear_beta(time_steps, beta_min=1e-4, beta_max=2e-2):
    """
    tf.cast(tf.linspace(beta_min, beta_max, time_steps), tf.float32)
    :param time_steps:
    :param beta_min:
    :param beta_max:
    :return:
    """
    return tf.cast(tf.linspace(beta_min, beta_max, time_steps), tf.float32)


def get_alpha(beta):
    """
    1 - beta
    :param beta:
    :return:
    """
    return 1 - beta


def get_alpha_cumprod(alpha):
    """
    tf.math.cumprod(alpha, axis=0)
    :param alpha:
    :return:
    """
    return tf.math.cumprod(alpha, axis=0)


def get_sqrt_recip_alpha(alpha):
    """
    1.0 / tf.math.sqrt(alpha)
    :param alpha:
    :return:
    """
    return 1.0 / tf.math.sqrt(alpha)


def get_sqrt_alpha_cumprod(alpha_cumprod):
    """
    tf.math.sqrt(alpha_cumprod)
    :param alpha_cumprod:
    :return:
    """
    return tf.math.sqrt(alpha_cumprod)


def get_sqrt_one_minus_alpha_cumprod(alpha_cumprod):
    """
    tf.math.sqrt(1. - alpha_cumprod)
    :param alpha_cumprod:
    :return:
    """
    return tf.math.sqrt(1. - alpha_cumprod)


def get_alpha_cumprod_prev(alpha_cumprod):
    """
    tf.concat([tf.ones((1,)), alpha_cumprod[:-1]], axis=0)
    :param alpha_cumprod:
    :return:
    """
    return tf.concat([tf.ones((1,)), alpha_cumprod[:-1]], axis=0)


# def get_value_at_t(value, t, x_shape):
#     """
#     get_value_at_t
#     :param value:
#     :param t: type: tf.int64, shape:(batch, 1)
#     :param x_shape:
#     :return: shape as x_shape
#     """
#     batch_size = t.shape[0]
#     value_at_t = tf.gather(value, t)
#     value_at_t_as_x_shape = tf.reshape(value_at_t, (batch_size, *((1,) * (len(x_shape) - 1))))  # shape (b, 1, 1, 1)
#     return value_at_t_as_x_shape


def get_value_at_t(value, t, x_shape=None):
    """
    get_value_at_t
    :param value:
    :param t: type: int, shape:1
    :param x_shape:
    :return (1, 1, 1, 1)
    """
    value_at_t = tf.gather(value, t)
    # value_at_t = tf.reshape(value_at_t, (1, 1, 1, 1))
    return value_at_t


# def get_t_from_t_image(img_t, input_type, t_num=1):
#     def pipe(t_img):  # (H, W, C)
#         if t_num != 1:
#             raise NotImplementedError
#         if input_type == 'loss':
#             height, width, channel = t_img.shape
#         else:
#             _, height, width, channel = t_img.shape
#         assert channel == 1, t_img.shape
#         t_slice = tf.reshape(tensor=t_img, shape=(height * width * channel,))
#         t_slice = t_slice[0:t_num]
#         return t_slice
#
#     if input_type == 'loss':
#         batch_size = img_t.shape[0]
#         t_list = list()
#         for batch in range(batch_size):
#             t_list.append(
#                 tf.expand_dims(pipe(t_img=img_t[batch]), axis=0))  # expend dim for batch
#         t = tf.concat(values=t_list, axis=0)
#     elif input_type == 'net':
#         t = pipe(t_img=img_t)
#     else:
#         raise ValueError(input_type)
#
#     return t


# def diffusion_process(x0, t, time_steps, beta_min, beta_max, return_z_t, z_t=None):
#     """
#     x0 -> x_t
#     q_sample: add noise
#     x_t = sqrt(alpha_t的累积乘积) + sqrt(1-alpha_t的累积乘积) * Z(noise)
#     :param x0: y_ture image  (batch, h, w, c)
#     :param t: shape is (batch_size, 1)
#     :param time_steps: for generating beta
#     :param beta_min: for generating beta
#     :param beta_max: for generating beta
#     :param return_z_t:
#     :param z_t: noise ~ N(0., 1.) at t time
#     :return: x_t or x_t, z_t
#     """
#     beta = get_linear_beta(time_steps, beta_min=beta_min, beta_max=beta_max)
#     alpha = get_alpha(beta=beta)
#     alpha_cumprod = get_alpha_cumprod(alpha=alpha)
#     sqrt_alpha_cumprod = get_sqrt_alpha_cumprod(alpha_cumprod=alpha_cumprod)
#     sqrt_one_minus_alpha_cumprod = get_sqrt_one_minus_alpha_cumprod(alpha_cumprod=alpha_cumprod)
#
#     sqrt_alpha_cumprod_t = get_value_at_t(sqrt_alpha_cumprod, t, x0.shape)
#     sqrt_one_minus_alpha_cumprod_t = get_value_at_t(sqrt_one_minus_alpha_cumprod, t, x0.shape)
#
#     z_t = None if return_z_t else z_t
#     if z_t is None:
#         z_t = tf.random_normal_initializer(mean=0., stddev=1.)(shape=x0.shape)
#     x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * z_t
#     if return_z_t:
#         return x_t, z_t
#     else:
#         return x_t


class DiffusionProcess:
    def __init__(self, time_steps, beta_min, beta_max, t, return_z_t):
        self.time_steps = time_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.t = t
        self.return_z_t = return_z_t

        self.beta = get_linear_beta(time_steps, beta_min=beta_min, beta_max=beta_max)
        self.alpha = get_alpha(beta=self.beta)
        self.alpha_cumprod = get_alpha_cumprod(alpha=self.alpha)
        self.sqrt_alpha_cumprod = get_sqrt_alpha_cumprod(alpha_cumprod=self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = get_sqrt_one_minus_alpha_cumprod(alpha_cumprod=self.alpha_cumprod)

    def q_sample(self, x0, z_t=None):
        sqrt_alpha_cumprod_t = get_value_at_t(self.sqrt_alpha_cumprod, self.t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = get_value_at_t(self.sqrt_one_minus_alpha_cumprod, self.t, x0.shape)
        if z_t is None:
            z_t = tf.random_normal_initializer(mean=0., stddev=1.)(shape=x0.shape)
        x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * z_t
        if self.return_z_t:
            return x_t, z_t
        else:
            return x_t


# def diffusion_process_seq(x0, t, time_steps, beta_min, beta_max):
#     """
#     x0(+z_0) -> x_1(+z_1) -> x_2(+z_2) ---->> x_t-1(+z_t-1) -> z_t
#     :param x0:
#     :param t:
#     :param time_steps:
#     :param beta_min:
#     :param beta_max:
#     :return: x_t_lise, z_t_list
#     """
#     assert 0 < t <= time_steps
#     x_t_lise = [x0]
#     z_t_list = list()
#     for i in range(t):
#         i = tf.ones(shape=(x0.shape[0]), dtype=tf.int64) * i
#         x_i, z_i = diffusion_process(x0=x_t_lise[-1], t=i, time_steps=time_steps, beta_min=beta_min, beta_max=beta_max)
#         x_t_lise.append(x_i)
#         z_t_list.append(z_i)
#     return x_t_lise[1:], z_t_list
#
#
# def inverse_diffusion_process(x_t, pred_z_t, t, time_steps, beta_min, beta_max):
#     """
#     x_t -> x0
#     :param x_t:
#     :param pred_z_t: model predict z_t under t time step
#     :param t: (batch, 1, 1, 1) tf.int64
#     :param time_steps: for generating beta
#     :param beta_min: for generating beta
#     :param beta_max: for generating beta
#     :return:
#     """
#     beta = get_linear_beta(time_steps, beta_min=beta_min, beta_max=beta_max)
#     alpha = get_alpha(beta=beta)
#     alpha_cumprod = get_alpha_cumprod(alpha=alpha)
#     sqrt_one_minus_alpha_cumprod = get_sqrt_one_minus_alpha_cumprod(alpha_cumprod=alpha_cumprod)
#     sqrt_one_minus_alpha_cumprod_t = get_value_at_t(value=sqrt_one_minus_alpha_cumprod, t=t, x_shape=tf.shape(x_t))
#     alpha_cumprod_t = get_value_at_t(value=alpha_cumprod, t=t, x_shape=tf.shape(x_t))
#
#     sample_result = 1. / tf.sqrt(alpha_cumprod_t) * (x_t - sqrt_one_minus_alpha_cumprod_t * pred_z_t)
#     return sample_result


class ReverseProcess:
    def __init__(self, time_steps, beta_min, beta_max, func='ddpm'):
        self.time_steps = time_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.func = func

        self.beta = get_linear_beta(time_steps, beta_min=beta_min, beta_max=beta_max)
        self.alpha = get_alpha(beta=self.beta)
        self.alpha_cumprod = get_alpha_cumprod(alpha=self.alpha)
        self.sqrt_one_minus_alpha_cumprod = get_sqrt_one_minus_alpha_cumprod(alpha_cumprod=self.alpha_cumprod)
        self.sqrt_recip_alpha = get_sqrt_recip_alpha(alpha=self.alpha)
        self.alpha_cumprod_prev = get_alpha_cumprod_prev(alpha_cumprod=self.alpha_cumprod)

    def p_sample(self, t, x_t, pred_z_t, t_index):
        beta_t = get_value_at_t(value=self.beta, t=t, x_shape=tf.shape(x_t))
        sqrt_one_minus_alpha_cumprod_t = get_value_at_t(
            value=self.sqrt_one_minus_alpha_cumprod, t=t, x_shape=tf.shape(x_t))
        sqrt_recip_alpha_t = get_value_at_t(value=self.sqrt_recip_alpha, t=t, x_shape=tf.shape(x_t))

        if t_index == 0:
            noise = tf.zeros_like(x_t)
        else:
            noise = tf.random_normal_initializer(mean=0., stddev=1.)(shape=tf.shape(x_t))

        if self.func == 'ddpm':
            reverse_mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * pred_z_t)
            if t_index == 0:
                sample_result = reverse_mean
            else:
                posterior_variance = self.beta * (1. - self.alpha_cumprod_prev) / (1. - self.alpha_cumprod)
                posterior_variance_t = get_value_at_t(value=posterior_variance, t=t, x_shape=tf.shape(x_t))
                sample_result = reverse_mean + posterior_variance_t * noise

        elif self.func == 'ddim':
            alpha_cumprod_t = get_value_at_t(value=self.alpha_cumprod, t=t, x_shape=tf.shape(x_t))
            alpha_cumprod_prev_t = get_value_at_t(value=self.alpha_cumprod_prev, t=t, x_shape=tf.shape(x_t))
            predict_x0 = alpha_cumprod_prev_t * (x_t - tf.sqrt(1 - alpha_cumprod_t) * pred_z_t) / tf.sqrt(
                alpha_cumprod_t)

            sigma = 0.0
            direction_point = tf.sqrt(1 - alpha_cumprod_prev_t - tf.square(sigma)) * pred_z_t
            random_noise = sigma * noise
            sample_result = predict_x0 + direction_point + random_noise
        else:
            raise ValueError

        return sample_result


# def reverse_process(x_t, pred_z_t, t, time_steps, beta_min, beta_max, t_index, func='ddpm'):
#     """
#     x_t -> x_t-1
#     p_sample denoise:
#     :param x_t:
#     :param pred_z_t: model predict z_t under t time step
#     :param t: (batch, 1, 1, 1) tf.int64
#     :param time_steps: for generating beta
#     :param beta_min: for generating beta
#     :param beta_max: for generating beta
#     :param t_index:
#     :param func:
#     :return:
#     """
#     beta = get_linear_beta(time_steps, beta_min=beta_min, beta_max=beta_max)
#     alpha = get_alpha(beta=beta)
#     alpha_cumprod = get_alpha_cumprod(alpha=alpha)
#     sqrt_one_minus_alpha_cumprod = get_sqrt_one_minus_alpha_cumprod(alpha_cumprod=alpha_cumprod)
#     sqrt_recip_alpha = get_sqrt_recip_alpha(alpha=alpha)
#     alpha_cumprod_prev = get_alpha_cumprod_prev(alpha_cumprod=alpha_cumprod)
#     posterior_variance = beta * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)
#
#     beta_t = get_value_at_t(value=beta, t=t, x_shape=tf.shape(x_t))
#     sqrt_one_minus_alpha_cumprod_t = get_value_at_t(value=sqrt_one_minus_alpha_cumprod, t=t, x_shape=tf.shape(x_t))
#     sqrt_recip_alpha_t = get_value_at_t(value=sqrt_recip_alpha, t=t, x_shape=tf.shape(x_t))
#     posterior_variance_t = get_value_at_t(value=posterior_variance, t=t, x_shape=tf.shape(x_t))
#
#     if t_index == 0:
#         noise = tf.zeros_like(x_t)
#     else:
#         noise = tf.random_normal_initializer(mean=0., stddev=1.)(shape=tf.shape(x_t))
#
#     if func == 'ddpm':
#         reverse_mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * pred_z_t)
#         if t_index == 0:
#             sample_result = reverse_mean
#         else:
#             sample_result = reverse_mean + posterior_variance_t * noise
#
#     elif func == 'ddim':
#         sigma = 0.0
#         alpha_cumprod_t = get_value_at_t(value=alpha_cumprod, t=t, x_shape=tf.shape(x_t))
#         alpha_cumprod_prev_t = get_value_at_t(value=alpha_cumprod_prev, t=t, x_shape=tf.shape(x_t))
#         predict_x0 = alpha_cumprod_prev_t * (x_t - tf.sqrt(1 - alpha_cumprod_t) * pred_z_t) / tf.sqrt(alpha_cumprod_t)
#         direction_point = tf.sqrt(1 - alpha_cumprod_prev_t - tf.square(sigma)) * pred_z_t
#         random_noise = sigma * noise
#         sample_result = predict_x0 + direction_point + random_noise
#     else:
#         raise ValueError
#
#     return sample_result


# def reverse_process_seq(x_t, pred_z_t, time_steps, beta_min, beta_max, model, func='ddpm', return_all=False):
#     """
#     xt -> x_t-1 -> x_t-2 -> x_t-3 ------>> x0
#     :param x_t:
#     :param pred_z_t: model predict z_t under t time step, list: [z_t, z_t-1, z_t-2,.... z_0]  todo:
#     :param time_steps: for generating beta
#     :param beta_min: for generating beta
#     :param beta_max: for generating beta
#     :param model:
#     :param func:
#     :param return_all: if True, return [xt, xt-1, xt-2, ... , x0] else only return x0
#     :return:
#     """
#     reverse_xt_list = [x_t]  # [xt, x_t-1, x_t-2, ..., x0]
#     for i in reversed(range(time_steps)):
#         reverse_t_xt = reverse_process(
#             x_t=reverse_xt_list[-1], pred_z_t=pred_z_t[i], t=tf.ones(shape=(tf.shape(x_t)[0]), dtype=tf.int64) * i,
#             time_steps=time_steps, beta_min=beta_min, beta_max=beta_max, t_index=i, func=func)
#         # reverse_t_xt = inverse_diffusion_process(
#         #     x_t=reverse_xt_list[-1], pred_z_t=pred_z_t[i], t=tf.ones(shape=(tf.shape(x_t)[0]), dtype=tf.int64) * i,
#         #     time_steps=time_steps, beta_min=beta_min, beta_max=beta_max)
#         reverse_xt_list.append(reverse_t_xt)
#
#     if return_all:
#         return reverse_xt_list
#     else:
#         return reverse_xt_list[-1]


class SVBRDFReverseProcessLoop:
    def __init__(self, model, filters, input_height, input_width, time_steps, beta_min, beta_max, func='ddpm'):
        self.model = model
        self.filters = filters
        self.input_height = input_height
        self.input_width = input_width
        self.time_steps = time_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.func = func

        self.reverse_process = ReverseProcess(
            time_steps=self.time_steps, beta_min=self.beta_min, beta_max=self.beta_max, func=self.func)

    def p_sample_loop(self, rendering_t, xt_n, xt_d, xt_r, xt_s):
        normal_x0 = xt_n
        diffuse_x0 = xt_d
        roughness_x0 = xt_r
        specular_x0 = xt_s
        batch_size = tf.shape(rendering_t)[0].numpy()
        assert batch_size == 1
        for t in reversed(range(self.time_steps)):
            t_emb = positional_embedding(t=t, filters=self.filters)  # (1, filters)
            t_emb = tf.tile(input=t_emb[:, None, None, :],  # (1, filters) -> (1, 1, 1, filters)
                            multiples=[1, self.input_height, self.input_width, 1])  # (1, h, w, filters)
            # (rendering(3), z_t_n(3), z_t_d(3), z_t_r(1), z_t_s(3), t_emb(filters))
            sample = tf.concat(
                [rendering_t, normal_x0, diffuse_x0, roughness_x0, specular_x0, t_emb], axis=channel_axis)
            sample_pred = self.model.predict(x=sample, steps=batch_size, verbose=0, use_multiprocessing=True)
            normal_pred_zt = tf.convert_to_tensor(sample_pred["normal"], dtype=tf.float32)
            diffuse_pred_zt = tf.convert_to_tensor(sample_pred["diffuse"], dtype=tf.float32)
            roughness_pred_zt = tf.convert_to_tensor(sample_pred["roughness"], dtype=tf.float32)
            specular_pred_zt = tf.convert_to_tensor(sample_pred["specular"], dtype=tf.float32)

            # t_tenor = tf.ones(shape=(batch_size, 1, 1, 1), dtype=tf.int64) * t
            normal_x0 = self.reverse_process.p_sample(t=t, x_t=normal_x0, pred_z_t=normal_pred_zt, t_index=t)
            diffuse_x0 = self.reverse_process.p_sample(t=t, x_t=diffuse_x0, pred_z_t=diffuse_pred_zt, t_index=t)
            roughness_x0 = self.reverse_process.p_sample(t=t, x_t=roughness_x0, pred_z_t=roughness_pred_zt, t_index=t)
            specular_x0 = self.reverse_process.p_sample(t=t, x_t=specular_x0, pred_z_t=specular_pred_zt, t_index=t)

        return normal_x0, diffuse_x0, roughness_x0, specular_x0


# def diffusion_reverse_process(x0, time_steps, beta_min, beta_max, t, z_t, pred_z_t):
#     """
#     p sample: denoise:
#     :param x0:
#     :param time_steps: for generating beta
#     :param beta_min: for generating beta
#     :param beta_max: for generating beta
#     :param t:
#     :param z_t: noise ~ N(0., 1.)
#     :param pred_z_t: model predict x under t time step, eps_z_t
#     :return: pred x0
#     """
#     beta = get_linear_beta(time_steps, beta_min=beta_min, beta_max=beta_max)
#     alpha = get_alpha(beta=beta)
#     alpha_cumprod = get_alpha_cumprod(alpha=alpha)
#     sqrt_alpha_cumprod = get_alpha_cumprod(alpha=alpha)
#     sqrt_one_minus_alpha_cumprod = get_sqrt_one_minus_alpha_cumprod(alpha_cumprod=alpha_cumprod)
#
#     beta_t = get_value_at_t(value=beta, t=t, x_shape=x0.shape)
#     sqrt_alpha_cumprod_t = get_value_at_t(sqrt_alpha_cumprod, t, x0.shape)
#     sqrt_one_minus_alpha_cumprod_t = get_value_at_t(value=sqrt_one_minus_alpha_cumprod, t=t, x_shape=x0.shape)
#
#     x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * z_t
#
#     sqrt_recip_alpha = get_sqrt_recip_alpha(alpha=alpha)
#     sqrt_recip_alpha_t = get_value_at_t(value=sqrt_recip_alpha, t=t, x_shape=x0.shape)
#
#     # reverse_mean = sqrt_recip_alpha_t * (x_t - (beta_t * pred_z_t / sqrt_one_minus_alpha_cumprod_t))
#     coeff = beta_t / sqrt_one_minus_alpha_cumprod_t
#     reverse_mean = (1 / 1 - tf.math.sqrt(1 - beta_t)) * (x_t - (coeff * pred_z_t))
#     alpha_cumprod_prev = get_alpha_cumprod_prev(alpha_cumprod=alpha_cumprod)
#     posterior_variance = beta * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)
#     posterior_variance_t = get_value_at_t(value=posterior_variance, t=t, x_shape=tf.shape(x_t))
#     # z_t = tf.random_normal_initializer(mean=0., stddev=1.)(shape=tf.shape(x_t))
#     pred_x0 = reverse_mean + tf.math.sqrt(posterior_variance_t) * z_t
#
#     return pred_x0


if __name__ == "__main__":
    # _dir = '/mnt/data1/jy/datasets/env/'
    # img_name = 'fixed_rendering_gt'
    # _format = 'png'
    # _img_path = os.path.join(_dir, '%s.%s' % (img_name, _format))
    # _x0 = image_util.read_image(
    #     image_path=tf.convert_to_tensor(_img_path, tf.string),
    #     image_height=256,
    #     image_width=256,
    #     image_format='png',
    #     image_channel=3,
    #     dtype=tf.float32,
    #     normalize=True
    # )
    # _x0_numpy = _x0.numpy()
    # _x0 = _x0[None, :, :, :]
    # rmd = np.random.RandomState()
    # # t = rmd.randint(low=0, high=100, size=1, dtype=np.int)
    #
    # _time_steps = 300
    # _beta_min = 1e-4
    # _beta_max = 2e-2
    # _t = 1
    # _x_t_lise, _z_t_list = diffusion_process_seq(
    #     x0=_x0, t=_t, time_steps=_time_steps, beta_min=_beta_min, beta_max=_beta_max)
    #
    # _reverse_x0 = reverse_process_seq(
    #     x_t=_x_t_lise[-1], pred_z_t=_z_t_list, t=_t, time_steps=_time_steps,
    #     beta_min=_beta_min, beta_max=_beta_max, func='ddpm')
    #
    # _reverse_x0 = tf.squeeze(_reverse_x0, axis=0).numpy()
    #
    # _reverse_x0 = _reverse_x0 * 255
    # # _reverse_x0 = np.clip(a=_reverse_x0, a_min=0., a_max=1.) * 255
    # _save_path = os.path.join(_dir, '%s.%s' % ('reverse_x0', _format))
    # cv2.imwrite(_save_path, _reverse_x0)
    #
    # _img = cv2.imread(_save_path, cv2.IMREAD_UNCHANGED)
    # os.remove(_save_path)
    # _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(_save_path, _img)
    # import os
    #
    # os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    # import random
    # import numpy as np
    #
    # a1 = tf.ones(shape=(1, 256, 256, 1), dtype=tf.int64) * 1
    # a2 = tf.ones(shape=(1, 256, 256, 1), dtype=tf.int64) * 2
    # a = tf.concat([a1, a2], axis=0)
    # c = list()
    # for b in range(tf.shape(a).numpy()[0]):
    #     d = a[b, 0, 0, 0]
    #     d = positional_embedding(d, 32)
    #     c.append(d)
    # print(tf.config.list_physical_devices('GPU'))
    # print(tf.test.is_gpu_available())
    # t = 200
    # t_emb = positional_embedding(t=t, filters=64)
    # c = tf.concat(c, axis=0)
    pass
