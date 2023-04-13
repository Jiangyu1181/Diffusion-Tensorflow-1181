# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Author: Jiangyu1181
# Github: https://github.com/Jiangyu1181
# ----------------------------------------------------------------------------------------------------------------------
# Description: ""
# Reference Paper: "Single-image SVBRDF capture with a rendering-aware deep network"
# Reference Code: ""
# ----------------------------------------------------------------------------------------------------------------------
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import backend

from Model.base import Base as BaseModel
from Util.diffusion import GaussianDiffusion as DDPM

channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1


class Model(BaseModel):
    def __init__(self, extra_info, config,
                 ema=0.999, clip_min=-1.0, clip_max=1.0,
                 inputs=None, outputs=None, name=None, trainable=True):
        super(Model, self).__init__(extra_info, config, inputs, outputs, name, trainable)

        self.ema = ema  # todo: add ema network for better performance
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.beta_start = self.dict_util.get_float(dict=self.config, key='beta_start')
        self.beta_end = self.dict_util.get_float(dict=self.config, key='beta_end')
        self.time_steps = self.dict_util.get_int(dict=self.config, key='time_steps')
        self.diffusion_util = DDPM(beta_start=self.beta_start, beta_end=self.beta_end,
                                   time_steps=self.time_steps, clip_min=clip_min, clip_max=clip_max)

    def call(self, inputs, training=None, mask=None):
        return super(Model, self).call(inputs, training, mask)

    def train_step(self, data):
        images, _ = data  # unpack data

        # image: (rendering, n, d, r, s)
        rendering = images[:, :, :, 0:3]
        n = images[:, :, :, 3:6]
        d = images[:, :, :, 6:9]
        r = images[:, :, :, 9:10]
        s = images[:, :, :, 10:13]

        # 1. Get the batch size
        batch_size = tf.shape(images)[0]

        # 2. Sample time_steps uniformly
        t = tf.random.uniform(minval=0, maxval=self.time_steps, shape=(batch_size,), dtype=tf.int64)

        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            n_zt = tf.random.normal(shape=tf.shape(n), dtype=n.dtype)
            d_zt = tf.random.normal(shape=tf.shape(d), dtype=d.dtype)
            r_zt = tf.random.normal(shape=tf.shape(r), dtype=r.dtype)
            s_zt = tf.random.normal(shape=tf.shape(s), dtype=s.dtype)

            # 4. Diffuse the images with noise
            n_xt = self.diffusion_util.q_sample(n, t, n_zt)
            d_xt = self.diffusion_util.q_sample(d, t, d_zt)
            r_xt = self.diffusion_util.q_sample(r, t, r_zt)
            s_xt = self.diffusion_util.q_sample(s, t, s_zt)
            rendering_with_svbrdf_xt = tf.concat([rendering, n_xt, d_xt, r_xt, s_xt], axis=self.channel_axis)

            # 5. Pass the diffused images and time steps to the network
            pred_zt = self([rendering_with_svbrdf_xt, t], training=True)

            # 6. Calculate the loss
            gt_zt = dict()
            gt_zt['normal'] = n_zt
            gt_zt['diffuse'] = d_zt
            gt_zt['roughness'] = r_zt
            gt_zt['specular'] = s_zt

            loss = self.compiled_loss(gt_zt, pred_zt, regularization_losses=self.losses)

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # 10. Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(gt_zt, pred_zt)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, _ = data  # unpack data

        # image: (rendering, n, d, r, s)
        rendering = images[:, :, :, 0:3]  # light_condition is 1 <-> 3 * 1
        n = images[:, :, :, 3:6]
        d = images[:, :, :, 6:9]
        r = images[:, :, :, 9:10]
        s = images[:, :, :, 10:13]

        # 1. Get the batch size
        batch_size = tf.shape(images)[0]

        # 2. Sample time_steps uniformly
        t = tf.random.uniform(minval=0, maxval=self.time_steps, shape=(batch_size,), dtype=tf.int64)

        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            n_zt = tf.random.normal(shape=tf.shape(n), dtype=n.dtype)
            d_zt = tf.random.normal(shape=tf.shape(d), dtype=d.dtype)
            r_zt = tf.random.normal(shape=tf.shape(r), dtype=r.dtype)
            s_zt = tf.random.normal(shape=tf.shape(s), dtype=s.dtype)

            # 4. Diffuse the images with noise
            n_xt = self.diffusion_util.q_sample(n, t, n_zt)
            d_xt = self.diffusion_util.q_sample(d, t, d_zt)
            r_xt = self.diffusion_util.q_sample(r, t, r_zt)
            s_xt = self.diffusion_util.q_sample(s, t, s_zt)
            rendering_with_svbrdf_xt = tf.concat([rendering, n_xt, d_xt, r_xt, s_xt], axis=self.channel_axis)

            # 5. Pass the diffused images and time steps to the network
            pred_zt = self([rendering_with_svbrdf_xt, t], training=False)

            # 6. Calculate the loss
            gt_zt = dict()
            gt_zt['normal'] = n_zt
            gt_zt['diffuse'] = d_zt
            gt_zt['roughness'] = r_zt
            gt_zt['specular'] = s_zt

            self.compiled_loss(gt_zt, pred_zt, regularization_losses=self.losses)

        # 10. Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(gt_zt, pred_zt)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def predict_step(self, data):
        images = data  # unpack data
        batch_size = tf.shape(images)[0]

        # image: (rendering, n, d, r, s)
        rendering = images[:, :, :, 0:3]  # light_condition is 1 -> 3 * 1
        n = rendering
        d = rendering
        r = images[:, :, :, 0:1]
        s = rendering

        n_xt = tf.random.normal(shape=tf.shape(n), dtype=n.dtype)
        d_xt = tf.random.normal(shape=tf.shape(d), dtype=d.dtype)
        r_xt = tf.random.normal(shape=tf.shape(r), dtype=r.dtype)
        s_xt = tf.random.normal(shape=tf.shape(s), dtype=s.dtype)

        xt = tf.concat([rendering, n_xt, d_xt, r_xt, s_xt], axis=self.channel_axis)
        for t in tqdm(reversed(range(0, self.time_steps)), desc="xt->x0"):
            tt = tf.ones(shape=(batch_size, 1), dtype=tf.int64) * t
            pred_zt = self([xt, tt], training=False)

            pred_n_xt = pred_zt['normal']
            pred_d_xt = pred_zt['diffuse']
            pred_r_xt = pred_zt['roughness']
            pred_s_xt = pred_zt['specular']

            n_xt = self.diffusion_util.p_sample(pred_n_xt, n_xt, tt, clip_denoised=True)  # xt -> x_t-1 ----> x0
            d_xt = self.diffusion_util.p_sample(pred_d_xt, d_xt, tt, clip_denoised=True)  # xt -> x_t-1 ----> x0
            r_xt = self.diffusion_util.p_sample(pred_r_xt, r_xt, tt, clip_denoised=True)  # xt -> x_t-1 ----> x0
            s_xt = self.diffusion_util.p_sample(pred_s_xt, s_xt, tt, clip_denoised=True)  # xt -> x_t-1 ----> x0
            xt = tf.concat([rendering, n_xt, d_xt, r_xt, s_xt], axis=self.channel_axis)

        n_x0 = tf.clip_by_value(n_xt, self.clip_min, self.clip_max)
        d_x0 = tf.clip_by_value(d_xt, self.clip_min, self.clip_max)
        r_x0 = tf.clip_by_value(r_xt, self.clip_min, self.clip_max)
        s_x0 = tf.clip_by_value(s_xt, self.clip_min, self.clip_max)

        x0_dict = dict()
        x0_dict['normal'] = n_x0
        x0_dict['diffuse'] = d_x0
        x0_dict['roughness'] = r_x0
        x0_dict['specular'] = s_x0

        return x0_dict

    def model_compile(self):
        """compiling model"""
        with self.strategy.scope():
            loss_dict = dict()
            loss_weights_dict = dict()
            loss_dict['normal'] = self.loss_util.normal_loss(config=self.config)
            loss_dict['diffuse'] = self.loss_util.diffuse_loss(config=self.config)
            loss_dict['roughness'] = self.loss_util.roughness_loss(config=self.config)
            loss_dict['specular'] = self.loss_util.specular_loss(config=self.config)

            loss_weights_dict['normal'] = self.dict_util.get_float(dict=self.config, key='normal_loss_weight')
            loss_weights_dict['diffuse'] = self.dict_util.get_float(dict=self.config, key='diffuse_loss_weight')
            loss_weights_dict['roughness'] = self.dict_util.get_float(dict=self.config, key='roughness_loss_weight')
            loss_weights_dict['specular'] = self.dict_util.get_float(dict=self.config, key='specular_loss_weight')
            self.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.dict_util.get_float(dict=self.config, key='lr'), beta_1=0.5, amsgrad=True),
                loss=loss_dict,
                loss_weights=loss_weights_dict,
                # run_eagerly=True,  # set it to True, for single gpu debugging
            )
            self.summary(line_length=200, positions=[0.3, 0.6, 0.9, 1.])

    def load_model(self, model, model_path):
        return super(Model, self).load_model(model, model_path)

    def log_image_callback(self, output_dir, train_data, val_data, *args):
        return super(Model, self).log_image_callback(output_dir, train_data, val_data, *args)

    def log_img_with_pattern_material(self, mode, data, epoch, render, config, *args):
        if epoch == 0 or epoch % self.dict_util.get_int(dict=config, key='log_frequency') != 0:
            return

        for data_example in data:
            # data_example: {data, gt}
            batch_rendered_data = data_example[0]  # data: shape (batch, height, width, (rendering+n+d+r+s=13))
            batch_i = self.batch_size - 1  # last batch, depend on batch_size and num of gpus
            # single data, shape (height, width, channel * light)
            assert self.light_condition == 1
            input_rendered_image = batch_rendered_data[batch_i]
            input_rendered_image = self.image_util.img_deprocess(img=input_rendered_image, is_list=False)
            input_rendered_image_list = list()
            input_rendered_image_list.append(input_rendered_image[:, :, 0:3])

            tf.summary.image(
                name="%s_1_input_rendered_image_batch_%d" % (mode, batch_i),
                data=input_rendered_image_list,
                max_outputs=len(input_rendered_image_list),
                step=epoch,
            )

            normal_gt = data_example[1]['normal'][batch_i]
            diffuse_gt = data_example[1]['diffuse'][batch_i]
            roughness_gt = data_example[1]['roughness'][batch_i]
            specular_gt = data_example[1]['specular'][batch_i]
            env_gt = data_example[1]['env'][batch_i] if 'env' in self.light_position_func else None

            materials_gt = tf.concat([
                normal_gt,
                diffuse_gt,
                roughness_gt,
                specular_gt
            ], axis=channel_axis)
            if env_gt is not None:
                materials_gt = tf.concat([materials_gt, env_gt], axis=channel_axis)

            # [-1, 1] -> [0, 1]
            normal_gt = self.image_util.img_deprocess(img=normal_gt) if normal_gt is not None else None
            diffuse_gt = self.image_util.img_deprocess(img=diffuse_gt) if diffuse_gt is not None else None
            roughness_gt = self.image_util.img_deprocess(img=roughness_gt) if roughness_gt is not None else None
            specular_gt = self.image_util.img_deprocess(img=specular_gt) if specular_gt is not None else None
            env_gt = self.image_util.img_deprocess(img=env_gt) if env_gt is not None else None

            gt_maps_list = [
                normal_gt,
                diffuse_gt,
                tf.tile(roughness_gt, multiples=[1, 1, 3 // roughness_gt.get_shape()[-1]]),
                specular_gt
            ]
            if env_gt is not None:
                gt_maps_list = gt_maps_list + [env_gt]

            if gt_maps_list is not None:
                tf.summary.image(
                    name="%s_2_gt_maps_batch_%d" % (mode, batch_i),
                    data=gt_maps_list,
                    max_outputs=len(gt_maps_list),
                    step=epoch
                )

            # log_normal_predict, log_diffuse_predict, log_roughness_predict, log_specular_predict
            result_predict = self.predict(x=tf.expand_dims(batch_rendered_data[batch_i], axis=0), verbose=0)
            normal_predict = result_predict['normal'][0]  # shape: (height, width, channel)
            diffuse_predict = result_predict['diffuse'][0]
            roughness_predict = result_predict['roughness'][0]
            specular_predict = result_predict['specular'][0]

            materials_pred = tf.concat([
                normal_predict,
                diffuse_predict,
                roughness_predict,
                specular_predict,
            ], axis=channel_axis)

            env_predict = None
            if 'env' in self.light_position_func:
                env_predict = result_predict['env'][0]

            normal_predict = self.image_util.img_deprocess(img=normal_predict)
            diffuse_predict = self.image_util.img_deprocess(img=diffuse_predict)
            roughness_predict = self.image_util.img_deprocess(img=roughness_predict)
            specular_predict = self.image_util.img_deprocess(img=specular_predict)
            env_predict = self.image_util.img_deprocess(img=env_predict) if env_predict is not None else None
            # due to after alpha_matting, can not use tile or get_shape
            roughness_predict = roughness_predict.repeat(3 // roughness_predict.shape[-1], axis=channel_axis)

            log_predict_maps_list = [
                normal_predict,
                diffuse_predict,
                roughness_predict,
                specular_predict
            ]

            if env_predict is not None:
                log_predict_maps_list += [env_predict]
                materials_pred = tf.concat([
                    materials_pred,
                    self.image_util.img_preprocess(img=env_predict),
                ], axis=channel_axis)

            tf.summary.image(
                name="%s_3_predict_maps_batch_%d" % (mode, batch_i),
                data=log_predict_maps_list,
                max_outputs=len(log_predict_maps_list),
                step=epoch
            )

            if render is not None:
                # get rendered results under different light condition
                materials_rendering_gt = render.rendering(material=materials_gt)
                materials_rendering_gt_list = tf.split(
                    materials_rendering_gt, num_or_size_splits=self.light_condition_for_train_loss,
                    axis=channel_axis)

                if materials_rendering_gt_list is not None:
                    tf.summary.image(
                        name="%s_4_gt_renders_batch_%d" % (mode, batch_i),
                        data=materials_rendering_gt_list,
                        max_outputs=len(materials_rendering_gt_list),
                        step=epoch
                    )

                log_render_predict = render.rendering(material=materials_pred)
                log_render_predict_list = tf.split(
                    log_render_predict, num_or_size_splits=self.light_condition_for_train_loss, axis=channel_axis)

                tf.summary.image(
                    name="%s_5_predict_renders_batch_%d" % (mode, batch_i),
                    data=log_render_predict_list,
                    max_outputs=len(log_render_predict_list),
                    step=epoch
                )
            break

    def log_img_with_pattern_capture(self, mode, data, epoch, render, config, *args):
        return super(Model, self).log_img_with_pattern_capture(mode, data, epoch, render, config, *args)

    def save_config(self, path):
        super(Model, self).save_config(path)

    def get_log_image_writer(self, path):
        return super(Model, self).get_log_image_writer(path)

    def model_checkpoint_callback(self, save_dir):
        return super(Model, self).model_checkpoint_callback(save_dir)

    def tensorboard_callback(self, path):
        return super(Model, self).tensorboard_callback(path)

    def log_img_during_training(self, mode, data, epoch, render, config, *args):
        return super(Model, self).log_img_during_training(mode, data, epoch, render, config, *args)

    def get_training_pbr_render(self):
        return super(Model, self).get_training_pbr_render()

    def get_config(self):
        return super(Model, self).get_config()


if __name__ == "__main__":
    pass
