import os

import anndata
import keras
import numpy as np
import tensorflow as tf

from keras.callbacks import EarlyStopping, History, ReduceLROnPlateau, LambdaCallback
from keras.engine.saving import model_from_json
from keras.layers import Dense, BatchNormalization, Dropout, Lambda, Input, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from scipy import sparse

from trvae.models._activations import ACTIVATIONS
from trvae.models._layers import LAYERS
from trvae.models._losses import LOSSES
from trvae.models._utils import print_progress, sample_z
from trvae.utils import label_encoder, train_test_split, remove_sparsity

"""
trVAE 类的定义
- 功能: trVAE 类是条件变分自编码器（C-VAE）的实现。
- 继承: 继承自 Network 基类。
- 作用: 用于处理多种实验条件（如control 和 stimulated），并生成潜在表示（latent space）和生成新的数据。
"""
class trVAE(object):
    """
        trVAE class. This class contains the implementation of trVAE network.

        Parameters
        ----------
        x_dimension: int
            number of gene expression space dimensions.
        conditions: list
            number of conditions used for one-hot encoding.
        z_dimension: int
            number of latent space dimensions.
        task_name: str
            name of the task.

        kwargs:
            `learning_rate`: float
                trVAE's optimizer's step size (learning rate).
            `alpha`: float
                KL divergence coefficient in the loss function.
            `beta`: float
                MMD loss coefficient in the loss function.
            `eta`: float
                Reconstruction coefficient in the loss function.
            `dropout_rate`: float
                dropout rate for Dropout layers in trVAE' architecture.
            `model_path`: str
                path to save model config and its weights.
            `clip_value`: float
                Optimizer's clip value used for clipping the computed gradients.
            `output_activation`: str
                Output activation of trVAE which Depends on the range of data. For positive value you can use "relu" or "linear" but if your data
                have negative value set to "linear" which is default.
            `use_batchnorm`: bool
                Whether use batch normalization in trVAE or not.
            `architecture`: list
                Architecture of trVAE. Must be a list of integers.
            `gene_names`: list
                names of genes fed as trVAE'sinput. Must be a list of strings.
    """

    """
    类的初始化

    - 功能: 初始化 trVAE 模型，定义模型的输入维度、潜在空间维度和其他超参数
    - 参数：
        - x_dimension: 输入数据的特征维度（如基因表达的维度）。
        - n_conditions: 条件的数量，即不同实验状态的数量。
        - z_dimension: 潜在空间维度的大小，默认为 10。
        - **kwargs: 其他可选参数，如学习率（learning_rate）、正则化系数（lambda_l1, lambda_l2）、MMD 计算方式等。
    """
    def __init__(self, x_dimension: int, conditions: list, z_dimension=10, **kwargs):
        self.x_dim = x_dimension    # 输入数据的特征维度（如基因表达的维度）
        self.z_dim = z_dimension    # 潜在空间维度的大小，默认为 10

        self.conditions = sorted(conditions)
        self.n_conditions = len(self.conditions)    # 条件的数量，即不同实验状态的数量

        self.lr = kwargs.get("learning_rate", 0.001)    # 学习率，默认值为 0.001
        # 正则化项相关的参数，控制生成器和 MMD 损失的权重
        self.alpha = kwargs.get("alpha", 0.0001)
        self.eta = kwargs.get("eta", 50.0)
        self.dr_rate = kwargs.get("dropout_rate", 0.1)  # Dropout 的比例，默认为 0.2，用于防止模型过拟合
        self.model_path = kwargs.get("model_path", "./models/trVAE/")
        self.loss_fn = kwargs.get("loss_fn", 'mse')
        self.ridge = kwargs.get('ridge', 0.1)
        self.scale_factor = kwargs.get("scale_factor", 1.0)
        self.clip_value = kwargs.get('clip_value', 3.0)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.output_activation = kwargs.get("output_activation", 'linear')
        self.use_batchnorm = kwargs.get("use_batchnorm", True)
        self.architecture = kwargs.get("architecture", [128, 128])
        self.size_factor_key = kwargs.get("size_factor_key", 'size_factors')
        self.device = kwargs.get("device", "gpu") if len(K.tensorflow_backend._get_available_gpus()) > 0 else 'cpu'

        self.gene_names = kwargs.get("gene_names", None)
        self.model_name = kwargs.get("model_name", "trvae")
        self.class_name = kwargs.get("class_name", 'trVAE')

        self.x = Input(shape=(self.x_dim,), name="data")
        self.size_factor = Input(shape=(1,), name='size_factor')
        self.encoder_labels = Input(shape=(self.n_conditions,), name="encoder_labels")
        self.decoder_labels = Input(shape=(self.n_conditions,), name="decoder_labels")
        self.z = Input(shape=(self.z_dim,), name="latent_data")

        self.condition_encoder = kwargs.get("condition_encoder", None)

        self.network_kwargs = {
            "x_dimension": self.x_dim,
            "z_dimension": self.z_dim,
            "conditions": self.conditions,
            "dropout_rate": self.dr_rate,
            "loss_fn": self.loss_fn,
            "output_activation": self.output_activation,
            "size_factor_key": self.size_factor_key,
            "architecture": self.architecture,
            "use_batchnorm": self.use_batchnorm,
            "gene_names": self.gene_names,
            "condition_encoder": self.condition_encoder,
            "train_device": self.device,
        }

        self.training_kwargs = {
            "learning_rate": self.lr,
            "alpha": self.alpha,
            "eta": self.eta,
            "ridge": self.ridge,
            "scale_factor": self.scale_factor,
            "clip_value": self.clip_value,
            "model_path": self.model_path,
        }

        self.beta = kwargs.get('beta', 50.0)
        self.mmd_computation_method = kwargs.pop("mmd_computation_method", "general")

        kwargs.update({"model_name": "cvae", "class_name": "trVAE"})

        self.network_kwargs.update({
            "mmd_computation_method": self.mmd_computation_method,
        })

        self.training_kwargs.update({
            "beta": self.beta,
        })

        self.init_w = keras.initializers.glorot_normal()

        if kwargs.get("construct_model", True):
            self.construct_network()

        if kwargs.get("construct_model", True) and kwargs.get("compile_model", True):
            self.compile_models()

        print_summary = kwargs.get("print_summary", False)
        if print_summary:
            self.encoder_model.summary()
            self.decoder_model.summary()
            self.cvae_model.summary()

    @classmethod
    def from_config(cls, config_path, new_params=None, compile=True, construct=True):
        """create class object from exsiting class' config file.
        从一个已有的配置文件创建一个 trVAE 类的实例，并允许根据需要修改某些配置

        Parameters
        ----------
        config_path: str
            Path to trVAE'sconfig json file.
        new_params: dict, optional
            Python dict of parameters which you wanted to assign new values to them.
        compile: bool
            ``True`` by default. if ``True``, will compile trVAE'smodel after creating an instance.
        construct: bool
            ``True`` by default. if ``True``, will construct trVAE'smodel after creating an instance.
        """
        # 导入 json 模块
        import json
        # 读取配置文件
        with open(config_path, 'rb') as f:
            class_config = json.load(f)

        # 设置 construct_model 和 compile_model
        # 将 construct_model 和 compile_model 这两个标志添加到 class_config 字典中，分别控制是否构建和编译模型。
        class_config['construct_model'] = construct
        class_config['compile_model'] = compile

        # 更新配置
        if new_params:
            class_config.update(new_params)

        return cls(**class_config)

    """ 构建解码器输出层 """
    def _output_decoder(self, h):
        h = Dense(self.x_dim, activation=None,
                  kernel_initializer=self.init_w,
                  use_bias=True)(h)
        h = ACTIVATIONS[self.output_activation](h)
        model_inputs = [self.z, self.decoder_labels]
        model_outputs = [h]

        return model_inputs, model_outputs

    """ 构建编码器子网络，将输入转换为潜在空间 """
    def _encoder(self, name="encoder"):
        """
           Constructs the decoder sub-network of CVAE. This function implements the
           decoder part of CVAE. It will transform primary space input to
           latent space to with n_dimensions = z_dimension.
           构建CVAE的解码器子网络。该函数实现CVAE的解码器部分。它将把原始空间输入转换为潜在空间,其中n维=z维。
       """
        # 连接输入数据和条件标签
        # 将输入数据 self.x 和条件标签 self.encoder_labels 连接起来，作为编码器的输入
        h = concatenate([self.x, self.encoder_labels], axis=1)
        # 遍历网络架构，构建隐藏层
        for idx, n_neuron in enumerate(self.architecture):
            h = Dense(n_neuron, kernel_initializer=self.init_w, use_bias=False)(h)
            if self.use_batchnorm:
                h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            if self.dr_rate > 0:
                h = Dropout(self.dr_rate)(h)

        # 计算潜在空间的均值和对数方差
        mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        # 重参数化采样,将潜在空间的均值和对数方差转换为潜在变量z
        z = Lambda(sample_z, output_shape=(self.z_dim,))([mean, log_var])
        # 返回编码器模型
        model = Model(inputs=[self.x, self.encoder_labels], outputs=[mean, log_var, z], name=name)
        return mean, log_var, model

    def _decoder(self, name="decoder"):
        # 连接潜在变量和条件标签
        h = concatenate([self.z, self.decoder_labels], axis=1)
        # 反向遍历网络架构，构建隐藏层
        # 这里使用了反向遍历（从大到小），因为解码器要从潜在空间逐渐解码回原始数据维度
        for idx, n_neuron in enumerate(self.architecture[::-1]):
            h = Dense(n_neuron, kernel_initializer=self.init_w, use_bias=False)(h)

            if self.use_batchnorm:
                h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            # 保存用于 MMD 正则化的第一层输出，用于使不同条件下的潜在表示分布更加相似
            if idx == 0:
                h_mmd = h
            if self.dr_rate > 0:
                h = Dropout(self.dr_rate)(h)

        # 生成解码器输出层
        model_inputs, model_outputs = self._output_decoder(h)
        # 构建并返回两个模型
        # model 是完整的解码器模型，它的输入是潜在变量和条件标签，输出是解码后的数据。
        model = Model(inputs=model_inputs, outputs=model_outputs, name=name)
        # mmd_model 是一个专门用于 MMD 正则化 的模型，它的输入是相同的潜在变量和条件标签，输出是第一层隐藏层的激活值 h_mmd，这个输出将用于计算 MMD 损失。
        mmd_model = Model(inputs=model_inputs, outputs=h_mmd, name='mmd_decoder')
        return model, mmd_model

    def construct_network(self):
        """
            Constructs the whole trVAE'snetwork. It is step-by-step constructing the trVAE network.
            First, It will construct the encoder part and get mu, log_var of
            latent space. Second, It will sample from the latent space to feed the
            decoder part in next step. Finally, It will reconstruct the data by
            constructing decoder part of trVAE.
            构建整个trVAE网络。
            它正在逐步构建trVAE网络。首先,它将构建编码器部分,并获取潜在空间的 mu和 log_var。
            其次,它将从潜在空间中采样,以在下一步中为解码器部分提供数据。最后,它将构建trVAE的解码器部分,从而重建数据。
        """
        # 构建编码器
        self.mu, self.log_var, self.encoder_model = self._encoder(name="encoder")
        # 构建解码器 和 MMD 解码器
        self.decoder_model, self.decoder_mmd_model = self._decoder(name="decoder")

        # 定义模型的输入
        inputs = [self.x, self.encoder_labels, self.decoder_labels]
        # 获取编码器的输出，潜在变量z
        encoder_outputs = self.encoder_model(inputs[:2])[2]
        # 把 z 传递给解码器，并合并上目标条件 decoder_labels
        decoder_inputs = [encoder_outputs, self.decoder_labels]

        # 获取解码器的输出
        decoder_outputs = self.decoder_model(decoder_inputs)
        decoder_mmd_outputs = self.decoder_mmd_model(decoder_inputs)

        # 生成最终输出
        reconstruction_output = Lambda(lambda x: x, name="reconstruction")(decoder_outputs) # 重构数据的输出，经过完整解码器生成
        mmd_output = Lambda(lambda x: x, name="mmd")(decoder_mmd_outputs)   # MMD 正则化的输出，用于计算 MMD 损失

        # 构建 trVAE 模型
        self.cvae_model = Model(inputs=inputs,
                                outputs=[reconstruction_output, mmd_output],
                                name="cvae")

        # 自定义层和激活函数
        self.custom_objects = {'mean_activation': ACTIVATIONS['mean_activation'],   # 自定义的激活函数
                               'disp_activation': ACTIVATIONS['disp_activation'],   # 自定义的激活函数
                               'SliceLayer': LAYERS['SliceLayer'],      # 自定义的层
                               'ColwiseMultLayer': LAYERS['ColWiseMultLayer'],  # 自定义的层
                               }
        # 将这些自定义对象添加到 Keras 的全局对象库中，以便在模型的编译和训练过程中使用这些自定义层和激活函数
        get_custom_objects().update(self.custom_objects)
        print(f"{self.class_name}' network has been successfully constructed!")

    def _calculate_loss(self):
        """
            Defines the loss function of trVAE'snetwork after constructing the wholenetwork.
            定义构建全局网络后 trVAE 网络的损失函数。

            - 功能：为 trVAE 模型计算总损失，包括：重构损失、MMD 损失、KL 散度损失等。这些损失会在训练过程中被最小化，以优化模型性能。
        """
        # 计算总损失：据模型选择的损失函数（self.loss_fn），计算总损失，这里应该是均方误差（MSE）
        loss = LOSSES[self.loss_fn](self.mu, self.log_var, self.alpha, self.eta)
        # 计算 MMD 损失：MMD 损失用于匹配潜在空间中不同条件（如控制组和刺激组）下的分布
        mmd_loss = LOSSES['mmd'](self.n_conditions, self.beta)
        # 计算 KL 散度损失
        kl_loss = LOSSES['kl'](self.mu, self.log_var)
        # 计算重构损失：表示模型生成的数据与输入数据之间的差异
        recon_loss = LOSSES[f'{self.loss_fn}_recon']

        return loss, mmd_loss, kl_loss, recon_loss

    def compile_models(self):
        """
            Compiles trVAE network with the defined loss functions and
            Adam optimizer with its pre-defined hyper-parameters.
            使用定义的损失函数 和 Adam 优化器及其预定义的超参数来编译trVAE网络。
        """
        # 定义 Adam 优化器
        optimizer = keras.optimizers.Adam(lr=self.lr, clipvalue=self.clip_value, epsilon=self.epsilon)
        # 计算损失函数
        loss, mmd_loss, kl_loss, recon_loss = self._calculate_loss()
        # 编译模型
        self.cvae_model.compile(optimizer=optimizer,
                                loss=[loss, mmd_loss],
                                metrics={self.cvae_model.outputs[0].name: loss,
                                         self.cvae_model.outputs[1].name: mmd_loss}
                                )

        print("trVAE'snetwork has been successfully compiled!")

    def to_mmd_layer(self, adata, batch_key):
        """
            Map ``adata`` in to the MMD space. This function will feed data
            in ``mmd_model`` of trVAE and compute the MMD space coordinates
            for each sample in data.
            将输入数据 adata 映射到 MMD 空间
            - 通过传递输入数据和条件标签给模型的 MMD 模型部分来完成的，目的是计算数据在潜在空间中的 MMD 编码。
            - MMD 编码的输出用于评估不同条件下的潜在空间分布，并计算 MMD 损失。
            - 模型在预测时通过该函数从编码器得到潜在表示，然后使用这些表示来计算不同条件下潜在空间的分布相似性。
            - 用于模型预测过程

            Parameters
            ----------
            adata: :class:`~anndata.AnnData`
                Annotated data matrix to be mapped to MMD latent space.
                Please note that ``adata.X`` has to be in shape [n_obs, x_dimension]
            encoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as trVAE'sencoder condition array.
            decoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as trVAE'sdecoder condition array.

            Returns
            -------
            adata_mmd: :class:`~anndata.AnnData`
                returns Annotated data containing MMD latent space encoding of ``adata``
                返回一个 AnnData 对象，其中包含输入数据在 MMD 空间中的表示
        """
        # 去除稀疏矩阵
        adata = remove_sparsity(adata)

        # 获取编码器和解码器的条件标签
        encoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)
        decoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)

        # 将条件标签转换为One-Hot编码
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        # 准备输入数据
        # 将输入数据 adata.X（样本特征矩阵）、编码器条件标签 encoder_labels 和解码器条件标签 decoder_labels 组合成一个列表，作为模型的输入
        cvae_inputs = [adata.X, encoder_labels, decoder_labels]

        # 获取 MMD 输出
        mmd = self.cvae_model.predict(cvae_inputs)[1]   # 使用 trVAE 模型的预测功能，将数据传递给模型，获取 MMD 空间的编码
        # 处理缺失值：将 MMD 输出中的 NaN 值和无穷大值替换为 0，确保所有值都是数值类型
        mmd = np.nan_to_num(mmd, nan=0.0, posinf=0.0, neginf=0.0)

        # 使用 MMD 输出创建一个新的 AnnData 对象，保存 MMD 空间的表示
        adata_mmd = anndata.AnnData(X=mmd)  # 将 MMD 编码作为 AnnData 对象的主数据矩阵
        adata_mmd.obs = adata.obs.copy(deep=True)

        return adata_mmd

    def to_z_latent(self, adata, batch_key):
        """
            Map `adata` in to the latent space. This function will feed data
            in encoder part of C-VAE and compute the latent space coordinates
            for each sample in data.
            将`adata`映射到潜在空间。此功能将数据输入C-VAE的编码器部分,并计算数据中每个样本的潜在空间坐标。

            # Parameters
                adata: `~anndata.AnnData`
                    Annotated data matrix to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].
                encoder_labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.
                return_adata: boolean
                    if `True`, will output as an `anndata` object or put the results in the `obsm` attribute of `adata`

            # Returns
                output: `~anndata.AnnData`
                    returns `anndata` object containing latent space encoding of 'adata'
        """
        # 检查并处理稀疏矩阵
        if sparse.issparse(adata.X):
            adata.X = adata.X.A

        # 获取编码器的条件标签
        encoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)
        # 将条件标签转换为一热编码
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        # 通过编码器计算潜在空间表示:将输入数据 adata.X 和条件标签 encoder_labels 传递给 trVAE 模型的编码器，计算潜在空间中的表示z
        latent = self.encoder_model.predict([adata.X, encoder_labels])[2]
        # 处理潜在空间中的缺失值:将潜在空间表示中的 NaN 值或无穷大值替换为 0，确保所有数值都是有效的
        latent = np.nan_to_num(latent)

        # 使用计算得到的潜在空间表示创建一个新的 AnnData 对象，并保留原始数据的元数据
        latent_adata = anndata.AnnData(X=latent)
        latent_adata.obs = adata.obs.copy(deep=True)

        return latent_adata

    def get_latent(self, adata, batch_key, return_z=True):
        """ Transforms `adata` in latent space of trVAE and returns the latent
        coordinates in the annotated (adata) format.

        该函数根据参数 return_z 决定是返回潜在空间中的表示 z，还是返回 MMD 层的表示。如果 beta == 0，则直接返回潜在空间的表示。
        用于删除批次效应

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset matrix in Primary space.
        batch_key: str
            Name of the column containing the study (batch) names for each sample.
        return_z: bool
            ``False`` by defaul. if ``True``, the output of bottleneck layer of network will be computed.

        Returns
        -------
        adata_pred: `~anndata.AnnData`
            Annotated data of transformed ``adata`` into latent space.
        """
        # 检查输入数据 adata 的基因名称（adata.var_names）是否与模型中使用的基因名称一致
        if set(self.gene_names).issubset(set(adata.var_names)):
            adata = adata[:, self.gene_names]
        else:
            raise Exception("set of gene names in train adata are inconsistent with trVAE'sgene_names")

        # 根据 beta 参数决定是否返回潜在变量 z
        if self.beta == 0:
            return_z = True

        # 根据 return_z 参数选择返回值
        if return_z or self.beta == 0:
            return self.to_z_latent(adata, batch_key)   # 调用 to_z_latent 函数，将输入数据映射到潜在空间，并返回潜在空间表示
        else:
            return self.to_mmd_layer(adata, batch_key)  # 调用 to_mmd_layer 函数，返回 MMD 层的表示

    def predict(self, adata, condition_key, target_condition=None):
        """
            Feeds ``adata`` to trVAE and produces the reconstructed data.
            使用训练好的 trVAE 模型对输入的单细胞 RNA 数据进行预测，生成重构后的数据，通常用于生成目标条件下的细胞数据。

            Parameters
            ----------
            adata: :class:`~anndata.AnnData`
                Annotated data matrix whether in primary space.
            condition_key: str  表示输入数据中的实验条件的列名
                :class:`~numpy.ndarray` of labels to be fed as trVAE'sencoder condition array.
            target_condition: str   指定要生成目标条件的标签
                :class:`~numpy.ndarray` of labels to be fed as trVAE'sdecoder condition array.

            Returns
            -------
            adata_pred: `~anndata.AnnData`
                Annotated data of predicted cells in primary space.
        """
        # 移除稀疏矩阵
        adata = remove_sparsity(adata)

        # 获取编码器的条件标签
        encoder_labels, _ = label_encoder(adata, self.condition_encoder, condition_key)
        # 获取解码器的条件标签
        if target_condition is not None:
            # 所有的解码器标签都会被设置为目标条件的标签，表示生成目标条件下的数据
            decoder_labels = np.zeros_like(encoder_labels) + self.condition_encoder[target_condition]
        else:
            # 解码器的标签将与编码器标签相同，表示重构输入数据的源条件
            decoder_labels, _ = label_encoder(adata, self.condition_encoder, condition_key)

        # 将条件标签转换为 one-hot 编码
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        # 通过模型生成预测数据
        x_hat = self.cvae_model.predict([adata.X, encoder_labels, decoder_labels])[0]

        # 生成新的 AnnData 对象
        adata_pred = anndata.AnnData(X=x_hat)
        adata_pred.obs = adata.obs
        adata_pred.var_names = adata.var_names

        return adata_pred

    def restore_model_weights(self, compile=True):
        """
            restores model weights from ``model_path``.
            从保存路径加载模型权重。

            Parameters
            ----------
            compile: bool
                if ``True`` will compile model after restoring its weights.

            Returns
            -------
            ``True`` if the model has been successfully restored.
            ``False`` if ``model_path`` is invalid or the model weights couldn't be found in the specified ``model_path``.
        """
        if os.path.exists(os.path.join(self.model_path, f"{self.model_name}.h5")):
            self.cvae_model.load_weights(os.path.join(self.model_path, f'{self.model_name}.h5'))

            self.encoder_model = self.cvae_model.get_layer("encoder")
            self.decoder_model = self.cvae_model.get_layer("decoder")

            if compile:
                self.compile_models()
            print(f"{self.model_name}'s weights has been successfully restored!")
            return True
        return False

    def restore_model_config(self, compile=True):
        """
            restores model config from ``model_path``.
            从保存路径加载模型配置

            Parameters
            ----------
            compile: bool
                if ``True`` will compile model after restoring its config.

            Returns
            -------
            ``True`` if the model config has been successfully restored.
            ``False`` if `model_path` is invalid or the model config couldn't be found in the specified ``model_path``.
        """
        if os.path.exists(os.path.join(self.model_path, f"{self.model_name}.json")):
            json_file = open(os.path.join(self.model_path, f"{self.model_name}.json"), 'rb')
            loaded_model_json = json_file.read()
            self.cvae_model = model_from_json(loaded_model_json)
            self.encoder_model = self.cvae_model.get_layer("encoder")
            self.decoder_model = self.cvae_model.get_layer("decoder")

            if compile:
                self.compile_models()

            print(f"{self.model_name}'s network's config has been successfully restored!")
            return True
        else:
            return False

    def restore_class_config(self, compile_and_consturct=True):
        """
            restores class' config from ``model_path``.
            从保存路径加载类配置

            Parameters
            ----------
            compile_and_consturct: bool
                if ``True`` will construct and compile model from scratch.

            Returns
            -------
            ``True`` if the scNet config has been successfully restored.
            ``False`` if `model_path` is invalid or the class' config couldn't be found in the specified ``model_path``.
        """
        import json
        if os.path.exists(os.path.join(self.model_path, f"{self.class_name}.json")):
            with open(os.path.join(self.model_path, f"{self.class_name}.json"), 'rb') as f:
                trVAE_config = json.load(f)

            # Update network_kwargs and training_kwargs dictionaries
            for key, value in trVAE_config.items():
                if key in self.network_kwargs.keys():
                    self.network_kwargs[key] = value
                elif key in self.training_kwargs.keys():
                    self.training_kwargs[key] = value

            # Update class attributes
            for key, value in trVAE_config.items():
                setattr(self, key, value)

            if compile_and_consturct:
                self.construct_network()
                self.compile_models()

            print(f"{self.class_name}'s config has been successfully restored!")
            return True
        else:
            return False

    def save(self, make_dir=True):
        """
            Saves all model weights, configs, and hyperparameters in the ``model_path``.
            保存模型的权重、配置和超参数

            Parameters
            ----------
            make_dir: bool
                Whether makes ``model_path`` directory if it does not exists.

            Returns
            -------
            ``True`` if the model has been successfully saved.
            ``False`` if ``model_path`` is an invalid path and ``make_dir`` is set to ``False``.
        """
        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            self.save_model_weights(make_dir)
            self.save_model_config(make_dir)
            self.save_class_config(make_dir)
            print(f"\n{self.class_name} has been successfully saved in {self.model_path}.")
            return True
        else:
            return False

    def save_model_weights(self, make_dir=True):
        """
            Saves model weights in the ``model_path``.
            保存模型权重

            Parameters
            ----------
            make_dir: bool
                Whether makes ``model_path`` directory if it does not exists.

            Returns
            -------
            ``True`` if the model has been successfully saved.
            ``False`` if ``model_path`` is an invalid path and ``make_dir`` is set to ``False``.
        """
        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            self.cvae_model.save_weights(os.path.join(self.model_path, f"{self.model_name}.h5"),
                                         overwrite=True)
            return True
        else:
            return False

    def save_model_config(self, make_dir=True):
        """
            Saves model's config in the ``model_path``.
            保存模型配置

            Parameters
            ----------
            make_dir: bool
                Whether makes ``model_path`` directory if it does not exists.

            Returns
            -------
            ``True`` if the model has been successfully saved.
            ``False`` if ``model_path`` is an invalid path and ``make_dir`` is set to ``False``.
        """
        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            model_json = self.cvae_model.to_json()
            with open(os.path.join(self.model_path, f"{self.model_name}.json"), 'w') as file:
                file.write(model_json)
            return True
        else:
            return False

    def save_class_config(self, make_dir=True):
        """
            Saves class' config in the ``model_path``.
            保存类的配置

            Parameters
            ----------
            make_dir: bool
                Whether makes ``model_path`` directory if it does not exists.

            Returns
            -------
                ``True`` if the model has been successfully saved.
                ``False`' if ``model_path`` is an invalid path and ``make_dir`` is set to ``False``.
        """
        import json

        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            config = {"x_dimension": self.x_dim,
                      "z_dimension": self.z_dim,
                      "n_conditions": self.n_conditions,
                      "condition_encoder": self.condition_encoder,
                      "gene_names": self.gene_names}
            all_configs = dict(list(self.network_kwargs.items()) +
                               list(self.training_kwargs.items()) +
                               list(config.items()))
            with open(os.path.join(self.model_path, f"{self.class_name}.json"), 'w') as f:
                json.dump(all_configs, f)

            return True
        else:
            return False

    """ 训练trVAE模型 """
    def _fit(self, adata,
             condition_key, train_size=0.8,
             n_epochs=300, batch_size=512,
             early_stop_limit=10, lr_reducer=7,
             save=True, retrain=True, verbose=3):
        # 数据集拆分:将输入数据 adata 按照比例拆分为训练集 train_adata 和验证集 valid_adata
        train_adata, valid_adata = train_test_split(adata, train_size)

        # 基因名称一致性检查
        # 确保输入数据集中的基因名称与模型中存储的基因名称一致。如果基因名称没有定义，则使用训练集的基因名称
        if self.gene_names is None:
            self.gene_names = train_adata.var_names.tolist()
        else:
            if set(self.gene_names).issubset(set(train_adata.var_names)):
                train_adata = train_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in train adata are inconsistent with class' gene_names")

            if set(self.gene_names).issubset(set(valid_adata.var_names)):
                valid_adata = valid_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in valid adata are inconsistent with class' gene_names")

        # 处理稀疏矩阵
        train_expr = train_adata.X.A if sparse.issparse(train_adata.X) else train_adata.X
        valid_expr = valid_adata.X.A if sparse.issparse(valid_adata.X) else valid_adata.X

        # 条件标签编码
        # 将训练集和验证集的条件标签（如实验条件、批次）进行编码，使用标签编码器将其转换为整数表示。
        train_conditions_encoded, self.condition_encoder = label_encoder(train_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        valid_conditions_encoded, self.condition_encoder = label_encoder(valid_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        # 检查是否需要重新训练
        # 如果指定不重新训练（retrain=False）且模型权重文件存在，则加载已有的模型权重并直接返回。
        if not retrain and os.path.exists(os.path.join(self.model_path, f"{self.model_name}.h5")):
            self.restore_model_weights()
            return

        # 设置训练回调函数，用于监控训练状态
        callbacks = [
            History(),
        ]
        # 如果 verbose 参数大于 2，则为每个 epoch 结束后添加一个回调函数，打印训练进度
        if verbose > 2:
            callbacks.append(
                LambdaCallback(on_epoch_end=lambda epoch, logs: print_progress(epoch, logs, n_epochs)))
            fit_verbose = 0
        else:
            fit_verbose = verbose
        # 如果设置了早停限制 early_stop_limit，则添加一个早停回调函数
        # EarlyStopping：如果验证集损失在指定的 early_stop_limit 个 epoch 中没有改善，训练将停止。
        if early_stop_limit > 0:
            callbacks.append(EarlyStopping(patience=early_stop_limit, monitor='val_loss'))
        # 如果设置了学习率调整器 lr_reducer，则添加一个学习率调整回调函数
        # ReduceLROnPlateau：当验证集损失在指定的 lr_reducer 个 epoch 内没有改善时，自动减小学习率。
        if lr_reducer > 0:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', patience=lr_reducer))

        # 将条件标签转换为一热编码
        train_conditions_onehot = to_categorical(train_conditions_encoded, num_classes=self.n_conditions)
        valid_conditions_onehot = to_categorical(valid_conditions_encoded, num_classes=self.n_conditions)

        # 准备训练和验证数据准备训练和验证数据
        # 将训练数据和验证数据的输入准备好。模型的输入包括表达矩阵（train_expr 和 valid_expr）以及条件标签的 一热编码
        x_train = [train_expr, train_conditions_onehot, train_conditions_onehot]
        x_valid = [valid_expr, valid_conditions_onehot, valid_conditions_onehot]

        y_train = [train_expr, train_conditions_encoded]
        y_valid = [valid_expr, valid_conditions_encoded]

        # 训练模型
        self.cvae_model.fit(x=x_train,
                            y=y_train,
                            validation_data=(x_valid, y_valid),
                            epochs=n_epochs,
                            batch_size=batch_size,
                            verbose=fit_verbose,
                            callbacks=callbacks,
                            )
        if save:
            self.save(make_dir=True)

    def _train_on_batch(self, adata,
                        condition_key, train_size=0.8,
                        n_epochs=300, batch_size=512,
                        early_stop_limit=10, lr_reducer=7,
                        save=True, retrain=True, verbose=3):
        # 数据集拆分
        train_adata, valid_adata = train_test_split(adata, train_size)

        # 基因名称检查
        if self.gene_names is None:
            self.gene_names = train_adata.var_names.tolist()
        else:
            if set(self.gene_names).issubset(set(train_adata.var_names)):
                train_adata = train_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in train adata are inconsistent with class' gene_names")

            if set(self.gene_names).issubset(set(valid_adata.var_names)):
                valid_adata = valid_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in valid adata are inconsistent with class' gene_names")

        # 条件标签编码
        train_conditions_encoded, self.condition_encoder = label_encoder(train_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        valid_conditions_encoded, self.condition_encoder = label_encoder(valid_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        # 检查是否需要重新训练
        if not retrain and os.path.exists(os.path.join(self.model_path, f"{self.model_name}.h5")):
            self.restore_model_weights()
            return

        # 条件标签的 One-hot 编码
        train_conditions_onehot = to_categorical(train_conditions_encoded, num_classes=self.n_conditions)
        valid_conditions_onehot = to_categorical(valid_conditions_encoded, num_classes=self.n_conditions)

        # 处理稀疏矩阵
        if sparse.issparse(train_adata.X):
            is_sparse = True
        else:
            is_sparse = False

        train_expr = train_adata.X
        valid_expr = valid_adata.X.A if is_sparse else valid_adata.X

        # 准备验证集数据：准备验证集的输入数据，包括特征矩阵和条件标签
        x_valid = [valid_expr, valid_conditions_onehot, valid_conditions_onehot]
        # 如果使用 nb（负二项式）或 zinb（零膨胀负二项式）作为损失函数，还需要将 size factor 作为额外的输入
        if self.loss_fn in ['nb', 'zinb']:
            x_valid.append(valid_adata.obs[self.size_factor_key].values)
            y_valid = [valid_adata.raw.X.A if sparse.issparse(valid_adata.raw.X) else valid_adata.raw.X,
                       valid_conditions_encoded]
        else:
            y_valid = [valid_expr, valid_conditions_encoded]

        # 批次训练循环
        es_patience, best_val_loss = 0, 1e10
        for i in range(n_epochs):
            train_loss = train_recon_loss = train_mmd_loss = 0.0
            for j in range(min(200, train_adata.shape[0] // batch_size)):
                # 随机采样批次样本:确保每次迭代时模型看到不同的样本子集。
                batch_indices = np.random.choice(train_adata.shape[0], batch_size)

                # 获取批次输入数据
                batch_expr = train_expr[batch_indices, :].A if is_sparse else train_expr[batch_indices, :]

                x_train = [batch_expr, train_conditions_onehot[batch_indices], train_conditions_onehot[batch_indices]]

                # 处理负二项式（NB）和零膨胀负二项式（ZINB）损失函数
                if self.loss_fn in ['nb', 'zinb']:
                    x_train.append(train_adata.obs[self.size_factor_key].values[batch_indices])
                    y_train = [train_adata.raw.X[batch_indices].A if sparse.issparse(
                        train_adata.raw.X[batch_indices]) else train_adata.raw.X[batch_indices],
                               train_conditions_encoded[batch_indices]]
                else:
                    y_train = [batch_expr, train_conditions_encoded[batch_indices]]

                # 批次训练:执行一个批次的前向传播和反向传播。模型会根据输入和目标计算损失，并更新权重。
                batch_loss, batch_recon_loss, batch_kl_loss = self.cvae_model.train_on_batch(x_train, y_train)

                train_loss += batch_loss / batch_size
                train_recon_loss += batch_recon_loss / batch_size
                train_mmd_loss += batch_kl_loss / batch_size

            # 验证集评估
            valid_loss, valid_recon_loss, valid_mmd_loss = self.cvae_model.evaluate(x_valid, y_valid, verbose=0)
            # 如果验证集的损失改善，则更新最佳损失 best_val_loss 并重置早停计数器 es_patience
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                es_patience = 0
            # 如果损失没有改善，则增加早停计数器，如果连续 early_stop_limit 个 epoch 没有改善，则触发早停，停止训练
            else:
                es_patience += 1
                if es_patience == early_stop_limit:
                    print("Training stopped with Early Stopping")
                    break

            # 训练日志输出
            logs = {"loss": train_loss, "recon_loss": train_recon_loss, "mmd_loss": train_mmd_loss,
                    "val_loss": valid_loss, "val_recon_loss": valid_recon_loss, "val_mmd_loss": valid_mmd_loss}
            print_progress(i, logs, n_epochs)

        # 模型保存
        if save:
            self.save(make_dir=True)

    def train(self, adata,
              condition_key, train_size=0.8,
              n_epochs=200, batch_size=128,
              early_stop_limit=10, lr_reducer=8,
              save=True, retrain=True, verbose=3):

        """
            Trains the network with ``n_epochs`` times given ``adata``.
            This function is using ``early stopping`` and ``learning rate reduce on plateau``
            techniques to prevent over-fitting.

            用给定的``adata``对网络进行``n_epochs``次训练。该函数使用``早期停止``和``学习率降低``技术来防止过度拟合。
            GPU 训练：如果设备为 GPU，则调用 _fit 函数，通常通过整个数据集训练。
            CPU 训练：如果设备为 CPU，则调用 _train_on_batch 函数，逐批次地进行训练，适合大数据集或内存受限的情况。

            Parameters
            ----------
            adata: :class:`~anndata.AnnData`
                Annotated dataset used to train & evaluate scNet.
            condition_key: str
                column name for conditions in the `obs` matrix of `train_adata` and `valid_adata`.
            train_size: float
                fraction of samples in `adata` used to train scNet.
            n_epochs: int
                number of epochs.
            batch_size: int
                number of samples in the mini-batches used to optimize scNet.
            early_stop_limit: int
                patience of EarlyStopping
            lr_reducer: int
                patience of LearningRateReduceOnPlateau.
            save: bool
                Whether to save scNet after the training or not.
            verbose: int
                Verbose level
            retrain: bool
                ``True`` by default. if ``True`` scNet will be trained regardless of existance of pre-trained scNet in ``model_path``. if ``False`` scNet will not be trained if pre-trained scNet exists in ``model_path``.

        """

        if self.device == 'gpu':
            return self._fit(adata, condition_key, train_size, n_epochs, batch_size, early_stop_limit,
                             lr_reducer, save, retrain, verbose)
        else:
            return self._train_on_batch(adata, condition_key, train_size, n_epochs, batch_size,
                                        early_stop_limit, lr_reducer, save, retrain,
                                        verbose)




class CLDRCVAE(object):
    """
        trVAE class. This class contains the implementation of trVAE network.

        Parameters
        ----------
        x_dimension: int
            number of gene expression space dimensions.
        conditions: list
            number of conditions used for one-hot encoding.
        z_dimension: int
            number of latent space dimensions.
        task_name: str
            name of the task.

        kwargs:
            `learning_rate`: float
                trVAE's optimizer's step size (learning rate).
            `alpha`: float
                KL divergence coefficient in the loss function.
            `beta`: float
                MMD loss coefficient in the loss function.
            `eta`: float
                Reconstruction coefficient in the loss function.
            `dropout_rate`: float
                dropout rate for Dropout layers in trVAE' architecture.
            `model_path`: str
                path to save model config and its weights.
            `clip_value`: float
                Optimizer's clip value used for clipping the computed gradients.
            `output_activation`: str
                Output activation of trVAE which Depends on the range of data. For positive value you can use "relu" or "linear" but if your data
                have negative value set to "linear" which is default.
            `use_batchnorm`: bool
                Whether use batch normalization in trVAE or not.
            `architecture`: list
                Architecture of trVAE. Must be a list of integers.
            `gene_names`: list
                names of genes fed as trVAE'sinput. Must be a list of strings.
    """

    """
    类的初始化

    - 功能: 初始化 trVAE 模型，定义模型的输入维度、潜在空间维度和其他超参数
    """
    def __init__(self, gene_size: int, conditions: list, cell_types: list, n_topic=10, **kwargs):
        self.gene_size = gene_size    # 输入数据的特征维度（基因的大小），等价于x_dim
        self.n_topic = n_topic  # 主题数量，等价于z_dim

        self.conditions = sorted(conditions)
        self.n_conditions = len(self.conditions)    # 条件的数量，即不同实验状态的数量
        self.cell_types = sorted(cell_types)  # 细胞类型的唯一值列表
        self.n_cell_types = len(self.cell_types)  # 细胞类型的数量
        self.cell_type_key = kwargs.get('cell_type_key', 'cell_type')

        # 超参数设置
        self.lr = kwargs.get("learning_rate", 0.001)  # 学习率，默认值为 0.001
        self.alpha = kwargs.get("alpha", 0.0001)  # 正则化项的权重参数
        self.eta = kwargs.get("eta", 50.0)  # eta 参数，用于损失函数或生成过程
        self.dr_rate = kwargs.get("dropout_rate", 0.1)  # Dropout 比例，默认 0.1，用于防止过拟合
        self.model_path = kwargs.get("model_path", "./models/CLDRCVAE/")  # 模型保存路径
        self.loss_fn = kwargs.get("loss_fn", 'mse')  # 损失函数类型
        self.ridge = kwargs.get('ridge', 0.1)  # Ridge 正则化系数
        self.scale_factor = kwargs.get("scale_factor", 1.0)  # 缩放因子
        self.clip_value = kwargs.get('clip_value', 3.0)  # 梯度裁剪值
        self.epsilon = kwargs.get('epsilon', 0.01)  # Epsilon，用于数值稳定性
        self.output_activation = kwargs.get("output_activation", 'linear')  # 输出激活函数类型
        self.use_batchnorm = kwargs.get("use_batchnorm", True)  # 是否使用批量归一化

        # 架构相关参数
        self.architecture = kwargs.get("architecture", [128, 128])  # 仅适用于编码器的结构，architecture 代表编码器的每一层的神经元数量
        self.size_factor_key = kwargs.get("size_factor_key", 'size_factors')   # Size Factor 键值
        self.device = kwargs.get("device", "gpu") if len(K.tensorflow_backend._get_available_gpus()) > 0 else 'cpu'

        # 可选参数
        self.gene_names = kwargs.get("gene_names", None)
        self.model_name = kwargs.get("model_name", "CLDRCVAE")
        self.class_name = kwargs.get("class_name", 'CLDRCVAE')

        # 输入的特征和标签
        self.x = Input(shape=(self.gene_size,), name="data")
        self.size_factor = Input(shape=(1,), name='size_factor')
        self.encoder_labels = Input(shape=(self.n_conditions,), name="encoder_labels")
        self.decoder_labels = Input(shape=(self.n_conditions,), name="decoder_labels")
        self.cell_type_labels = Input(shape=(self.n_cell_types,), name="cell_type_labels")
        self.x_hat = tf.random.normal([1, gene_size])  # 随机张量，后续在 train 过程中，可以生成实际的 x_hat 替换这个占位张量
        self.z = Input(shape=(self.n_topic,), name="latent_data")

        # 对比学习相关参数
        self.topk = kwargs.pop("topk", 5)  # 用于选择正样本的 top-k 个样本
        self.contrastive_lambda = kwargs.pop("contrastive_lambda", 10.0)  # 对比损失的权重
        self.contrastive_x = Input(shape=(self.gene_size,), name="contrastive_data")  # 对比样本的输入
        self.contrastive_labels = Input(shape=(self.n_conditions,), name="contrastive_labels")  # 对比样本的条件标签
        self.contrastive_z = Input(shape=(self.n_topic,), name="contrastive_z") # 对比潜在空间表示
        # 用于第二对比损失
        self.gamma_cov = kwargs.get('gamma_cov', 0.1)  # 控制协方差约束的权重
        self.second_contrastive_lambda= kwargs.pop("second_contrastive_lambda", 10.0)  # 第二对比损失的权重
        self.margin = kwargs.get('margin', 1.0)  # 控制协方差约束的权重
        self.common_dim = kwargs.get('common_dim', 25)  # 控制协方差约束的权重

        self.condition_encoder = kwargs.get("condition_encoder", None)

        # nb 使用的参数
        self.disp = tf.Variable(
            initial_value=np.ones(self.gene_size),
            dtype=tf.float32,
            name="disp"
        )

        self.network_kwargs = {
            "gene_size": self.gene_size,
            "n_topic": self.n_topic,
            "conditions": self.conditions,
            "dropout_rate": self.dr_rate,
            "loss_fn": self.loss_fn,
            "output_activation": self.output_activation,
            "size_factor_key": self.size_factor_key,
            "architecture": self.architecture,
            "use_batchnorm": self.use_batchnorm,
            "gene_names": self.gene_names,
            "condition_encoder": self.condition_encoder,
            "train_device": self.device,
        }

        self.training_kwargs = {
            "learning_rate": self.lr,
            "alpha": self.alpha,
            "eta": self.eta,
            "ridge": self.ridge,
            "scale_factor": self.scale_factor,
            "clip_value": self.clip_value,
            "model_path": self.model_path,
        }

        # MMD 损失计算的相关参数
        self.beta = kwargs.get('beta', 50.0)
        self.mmd_computation_method = kwargs.pop("mmd_computation_method", "general")

        kwargs.update({"model_name": "cvae", "class_name": "CLDRCVAE"})

        # 更新 network_kwargs 以支持对比学习相关参数
        self.network_kwargs.update({
            "mmd_computation_method": self.mmd_computation_method,
            "contrastive_lambda": self.contrastive_lambda,
        })

        self.training_kwargs.update({
            "beta": self.beta,
            "contrastive_lambda": self.contrastive_lambda,
        })

        self.init_w = keras.initializers.glorot_normal()

        if kwargs.get("construct_model", True):
            self.construct_network()

        if kwargs.get("construct_model", True) and kwargs.get("compile_model", True):
            self.compile_models()

        # 打印模型结构摘要
        print_summary = kwargs.get("print_summary", False)
        if print_summary:
            self.encoder_model.summary()
            self.decoder_model.summary()
            self.cvae_model.summary()


    @classmethod
    def from_config(cls, config_path, new_params=None, compile=True, construct=True):
        """create class object from exsiting class' config file.
        从一个已有的配置文件创建一个 trVAE 类的实例，并允许根据需要修改某些配置

        Parameters
        ----------
        config_path: str
            Path to trVAE'sconfig json file.
        new_params: dict, optional
            Python dict of parameters which you wanted to assign new values to them.
        compile: bool
            ``True`` by default. if ``True``, will compile trVAE'smodel after creating an instance.
        construct: bool
            ``True`` by default. if ``True``, will construct trVAE'smodel after creating an instance.
        """
        # 导入 json 模块
        import json
        # 读取配置文件
        with open(config_path, 'rb') as f:
            class_config = json.load(f)

        # 设置 construct_model 和 compile_model
        # 将 construct_model 和 compile_model 这两个标志添加到 class_config 字典中，分别控制是否构建和编译模型。
        class_config['construct_model'] = construct
        class_config['compile_model'] = compile

        # 更新配置
        if new_params:
            class_config.update(new_params)

        return cls(**class_config)


    """ 构建编码器子网络，将输入转换为潜在空间 """
    def _encoder(self, name="encoder"):
        """
           Constructs the decoder sub-network of CVAE. This function implements the
           decoder part of CVAE. It will transform primary space input to
           latent space to with n_dimensions = z_dimension.
           构建CVAE的解码器子网络。该函数实现CVAE的解码器部分。它将把原始空间输入转换为潜在空间,其中n维=z维。
       """
        # 连接输入数据和条件标签
        # 将输入数据 self.x 和条件标签 self.encoder_labels 连接起来，作为编码器的输入
        h = concatenate([self.x, self.encoder_labels], axis=1)

        # 遍历网络架构，构建隐藏层
        for idx, n_neuron in enumerate(self.architecture):
            h = Dense(n_neuron, kernel_initializer=self.init_w, use_bias=False)(h)
            if self.use_batchnorm:
                h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            if self.dr_rate > 0:
                h = Dropout(self.dr_rate)(h)    # 在隐藏层上应用 Dropout

        # 计算潜在空间的均值和对数方差
        mean = Dense(self.n_topic, kernel_initializer=self.init_w)(h)
        log_var = Dense(self.n_topic, kernel_initializer=self.init_w)(h)

        # 重参数化采样,将潜在空间的均值和对数方差转换为潜在变量z
        z = Lambda(sample_z, output_shape=(self.n_topic,))([mean, log_var])

        model = Model(inputs=[self.x, self.encoder_labels], outputs=[mean, log_var, z], name=name)

        return mean, log_var, model


    def _output_decoder(self, h):
        """
        构建解码器的输出层，将隐藏层 h 生成最终的输出矩阵。输出将结合两个 Softmax 分支 ，最终生成 X_recon。
        """
        h = Dense(self.gene_size, activation=None,
                  kernel_initializer=self.init_w,
                  use_bias=True)(h)
        h = ACTIVATIONS[self.output_activation](h)

        model_inputs = [self.z, self.decoder_labels]
        model_outputs = [h]

        return model_inputs, model_outputs


    def _decoder(self, name="decoder"):
        """
            解码器结构：
            输入为 theta 和 decoder_labels
            第一层：MMD 层
            第二层：theta_layer，相当于 CLNTM 中的 beta_layer
            第三层：eta_bn_layer
            最后一层为 softmax 层 (包含在_output_decoder()函数中)
            输出为 X_recon
        """
        # 连接潜在变量和条件标签
        h = concatenate([self.z, self.decoder_labels], axis=1)

        # 反向遍历网络架构，构建隐藏层
        # 这里使用了反向遍历（从大到小），因为解码器要从潜在空间逐渐解码回原始数据维度
        for idx, n_neuron in enumerate(self.architecture[::-1]):
            h = Dense(n_neuron, kernel_initializer=self.init_w, use_bias=False)(h)
            if self.use_batchnorm:
                h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            # 保存用于 MMD 正则化的第一层输出，用于使不同条件下的潜在表示分布更加相似
            if idx == 0:
                h_mmd = h
            if self.dr_rate > 0:
                h = Dropout(self.dr_rate)(h)

        # 生成解码器输出层
        model_inputs, model_outputs = self._output_decoder(h)

        # 构建并返回两个模型
        # model 是完整的解码器模型，它的输入是潜在变量和条件标签，输出是解码后的数据。
        model = Model(inputs=model_inputs, outputs=model_outputs, name=name)
        # mmd_model 是一个专门用于 MMD 正则化 的模型，它的输入是相同的潜在变量和条件标签，输出是第一层隐藏层的激活值 h_mmd，这个输出将用于计算 MMD 损失。
        mmd_model = Model(inputs=model_inputs, outputs=h_mmd, name='mmd_decoder')
        return model, mmd_model

    def construct_network(self):
        """
            Constructs the whole trVAE'snetwork. It is step-by-step constructing the trVAE network.
            First, It will construct the encoder part and get mu, log_var of
            latent space. Second, It will sample from the latent space to feed the
            decoder part in next step. Finally, It will reconstruct the data by
            constructing decoder part of trVAE.
            构建整个trVAE网络。
            它正在逐步构建trVAE网络。首先,它将构建编码器部分,并获取潜在空间的 mu和 log_var。
            其次,它将从潜在空间中采样,以在下一步中为解码器部分提供数据。最后,它将构建trVAE的解码器部分,从而重建数据。
        """
        # 构建编码器
        self.mu, self.log_var, self.encoder_model = self._encoder(name="encoder")
        # 构建解码器 和 MMD 解码器
        self.decoder_model, self.decoder_mmd_model = self._decoder(name="decoder")

        # 编码器输入
        inputs = [self.x, self.encoder_labels, self.decoder_labels, self.cell_type_labels]
        # 获取编码器的输出，细胞主题theta
        self.z = self.encoder_model(inputs[:2])[2]
        # 对潜在空间 z 进行分割
        self.z_common = self.z[:, :self.common_dim]  # 共性部分 A (前 common_dim 维)
        self.z_specific = self.z[:, self.common_dim:self.n_topic]  # 特异性部分 B (剩余维度)

        # 把 theta 传递给解码器，并合并上目标条件 decoder_labels
        decoder_inputs = [self.z, self.decoder_labels]

        # 获取解码器的输出
        self.decoder_outputs = self.decoder_model(decoder_inputs)
        decoder_mmd_outputs = self.decoder_mmd_model(decoder_inputs)

        # 生成最终输出:将 decoder_outputs 封装为 reconstruction_output
        reconstruction_output = Lambda(lambda x: x, name="reconstruction")(self.decoder_outputs)   # 重构数据的输出
        mmd_output = Lambda(lambda x: x, name="mmd")(decoder_mmd_outputs)   # MMD 正则化的输出，用于计算 MMD 损失
        contrastive_output = Lambda(lambda x: x, name="contrastive")(self.z)
        second_contrastive_output = Lambda(lambda x: x, name="second_contrastive")(self.z)

        # 构建 trVAE 模型
        self.cvae_model = Model(inputs=inputs,
                                outputs=[reconstruction_output, mmd_output, contrastive_output,
                                         second_contrastive_output],
                                name="cvae")

        # 自定义层和激活函数
        self.custom_objects = {'mean_activation': ACTIVATIONS['mean_activation'],
                               'disp_activation': ACTIVATIONS['disp_activation'],
                               'SliceLayer': LAYERS['SliceLayer'],
                               'ColwiseMultLayer': LAYERS['ColWiseMultLayer'],
                               }
        # 将这些自定义对象添加到 Keras 的全局对象库中，以便在模型的编译和训练过程中使用这些自定义层和激活函数
        get_custom_objects().update(self.custom_objects)
        print(f"{self.class_name}' network has been successfully constructed!")


    def _calculate_loss(self):
        """
            Defines the loss function of trVAE's network after constructing the whole network.
            定义构建全局网络后 trVAE 网络的损失函数。

            - 功能：为 trVAE 模型计算总损失，包括：重构损失、MMD 损失、KL 散度损失、对比损失等。
        """

        # 计算 MMD 损失：用于匹配不同条件下潜在空间分布的差异
        mmd_loss = LOSSES['mmd'](self.n_conditions, self.beta) # 计算 MMD 损失：用于匹配不同条件下潜在空间分布的差异
        # 将 MMD 损失固定为 0
        # mmd_loss = tf.constant(0.0, dtype=tf.float32)

        # 计算 KL 散度损失：使潜在表示分布接近标准正态分布
        kl_loss = LOSSES['kl'](self.mu, self.log_var)

        # 计算重构损失：衡量模型生成的数据与输入数据之间的差异（例如均方误差）
        if self.loss_fn == 'nb':
            # 使用与 gene_size 匹配的 disp 参数
            recon_loss = LOSSES['nb_wo_kl'](self.disp)  # 正确传递 disp
        elif self.loss_fn == 'zinb':
            recon_loss = LOSSES['zinb_wo_kl']
        else:
            recon_loss = LOSSES[f'{self.loss_fn}_recon']

        # 计算第一对比损失：通过 z 和 contrastive_z 的相似性来计算
        # 生成对比样本（正负样本）和对应标签
        contrastive_pos_x, contrastive_pos_labels, contrastive_neg_x, contrastive_neg_labels = \
            self.generate_contrastive_samples(self.x, self.encoder_labels, self.cell_type_labels)
        # 对比学习生成潜在表示
        self._contrastive_learning(contrastive_pos_x, contrastive_pos_labels, contrastive_neg_x, contrastive_neg_labels)
        # 计算对比损失
        contrastive_loss = LOSSES['contrastive'](self.contrastive_lambda, self.z, self.contrastive_pos_z,
                                                 self.contrastive_neg_z)

        # 计算第二对比损失
        # 生成用于第二对比学习的正样本和标签
        contrastive_pos_x, contrastive_pos_labels = self.select_samples_for_second_contrastive(self.x, self.encoder_labels, self.cell_type_labels)
        # 获取第二对比学习的潜在表示
        pos_z_common, pos_z_specific = self._contrastive_learning_second(contrastive_pos_x, contrastive_pos_labels)
        # 计算第二对比学习的综合损失（包括 L_A, L_B 和 L_cov）
        second_contrastive_loss = LOSSES['second_contrastive'](
            self.z_common,
            self.z_specific,
            pos_z_common,
            pos_z_specific,
            self.second_contrastive_lambda,
            margin=self.margin
        )

        # 计算组合损失：根据模型选择的损失函数（self.loss_fn）
        loss = LOSSES[self.loss_fn](self.mu, self.log_var, self.alpha, self.eta)

        return loss, recon_loss, mmd_loss, kl_loss, contrastive_loss, second_contrastive_loss


    def compile_models(self):
        """
        Compiles CLDRCVAE network with the defined loss functions, contrastive loss, and
        Adam optimizer with its pre-defined hyper-parameters.
        使用定义的损失函数、对比损失和 Adam 优化器及其预定义的超参数来编译 CLDRCVAE 网络。
        """
        # 定义 Adam 优化器
        optimizer = keras.optimizers.Adam(lr=self.lr, clipvalue=self.clip_value, epsilon=self.epsilon)

        # 计算损失函数
        loss, recon_loss, mmd_loss, kl_loss, contrastive_loss, second_contrastive_loss = self._calculate_loss()

        # 固定 mmd_loss 为 0
        # mmd_loss = lambda y_true, y_pred: tf.constant(0.0)

        # 编译模型：传递了 loss 和 mmd_loss
        self.cvae_model.compile(optimizer=optimizer,
                                loss=[loss, mmd_loss, contrastive_loss, second_contrastive_loss],
                                metrics={self.cvae_model.outputs[0].name: loss,
                                         self.cvae_model.outputs[1].name: mmd_loss,
                                         self.cvae_model.outputs[2].name: contrastive_loss,
                                         self.cvae_model.outputs[3].name: second_contrastive_loss}
                                )
        print("CLDRCVAE's network has been successfully compiled!")

    @tf.function
    def generate_contrastive_samples(self, X, encoder_labels, cell_type_labels):
        """
            生成正样本 contrastive_pos_X 和负样本 contrastive_neg_X。
            正样本通过随机选择与样本 X 的细胞类型和条件标签相同的一个样本生成。
            负样本通过将当前样本中最具差异的 top-k 个基因替换为重构数据中的值生成。

            参数：
            - X: 原始输入数据 (Tensor)
            - x_hat: 重构数据 (Tensor)，通常在 predict 函数中得到
            - encoder_labels: 条件标签（与 X 对应）
            - cell_type_labels: 细胞类型标签（与 X 对应）

            返回：
            - contrastive_pos_X: 与 X 具有相同标签的随机正样本
            - contrastive_pos_labels: 与 contrastive_pos_X 对应的条件标签
            - contrastive_neg_X: 基于差异基因替换生成的负样本
            - contrastive_neg_labels: 与 contrastive_neg_X 对应的条件标签
        """
        """
        计算每种细胞类型在不同实验条件下的基因差异
        - 在相同细胞类型下、不同实验条件之间变化幅度最大的基因。
        """

        def calculate_top_k_genes(cell_type):
            # 确保 cell_type 与 cell_type_labels 的类型一致
            cell_type = tf.cast(cell_type, dtype=cell_type_labels.dtype)
            # 筛选出该细胞类型的数据
            indices = tf.where(tf.equal(cell_type_labels, cell_type))[:, 0]
            cell_type_data = tf.gather(X, indices)

            # 计算每个基因的方差，并取 top-k 个差异最大的基因索引
            variance_per_gene = tf.math.reduce_variance(cell_type_data, axis=0)
            top_k_genes = tf.argsort(variance_per_gene, direction='DESCENDING')[:self.topk]
            return top_k_genes

        # 获取 unique 的细胞类型，标记存在的细胞类型
        unique_cell_types = tf.cast(
            tf.reduce_any(tf.equal(tf.reshape(cell_type_labels, [-1, 1]),
                                   tf.range(self.n_cell_types, dtype=cell_type_labels.dtype)), axis=0),
            dtype=tf.float32
        )
        unique_cell_types = tf.reshape(unique_cell_types, [1, -1])

        # 定义每种细胞类型的 top-k 基因列表
        top_k_genes_list = tf.map_fn(
            lambda ct: calculate_top_k_genes(ct) if unique_cell_types[0, ct] == 1 else tf.zeros(self.topk,
                                                                                                dtype=tf.int32),
            tf.range(self.n_cell_types),
            dtype=tf.int32
        )

        """定义处理每个样本的函数"""

        def process_sample(i):
            # 获取样本的条件标签和细胞类型标签
            condition_label_i = tf.expand_dims(tf.gather(encoder_labels, i), axis=0)
            cell_type_label_i = tf.expand_dims(tf.gather(cell_type_labels, i), axis=0)

            # 找到与当前样本细胞类型和条件标签相同的样本索引（正样本）
            same_condition_indices = tf.where(
                (tf.reduce_all(tf.equal(encoder_labels, condition_label_i), axis=1)) &
                (tf.reduce_all(tf.equal(cell_type_labels, cell_type_label_i), axis=1))
            )[:, 0]

            # 生成正样本
            if tf.size(same_condition_indices) > 0:
                chosen_index = tf.random.shuffle(same_condition_indices)[0]
                contrastive_pos_sample = X[chosen_index]
            else:
                contrastive_pos_sample = X[i]

            # 检查当前 cell_type_label_i 是否在 unique_cell_types 中
            exists_in_unique = tf.reduce_any(tf.reduce_all(tf.equal(unique_cell_types, cell_type_label_i), axis=1))
            if not exists_in_unique:
                return contrastive_pos_sample, encoder_labels[i], X[i], encoder_labels[i]

            # 获取与 cell_type_label_i 匹配的 top-k 基因
            cell_type_index = tf.where(tf.reduce_all(tf.equal(unique_cell_types, cell_type_label_i), axis=1))[0][0]
            top_k_genes = top_k_genes_list[cell_type_index]

            # 生成负样本
            contrastive_neg_sample = tf.identity(X[i])
            contrastive_neg_sample = tf.tensor_scatter_nd_update(
                contrastive_neg_sample,
                indices=tf.reshape(top_k_genes, [-1, 1]),
                updates=tf.gather(self.x_hat[i], top_k_genes)
            )

            return contrastive_pos_sample, encoder_labels[i], contrastive_neg_sample, encoder_labels[i]

        # 使用 tf.map_fn 对每个样本执行 process_sample
        contrastive_pos_X, contrastive_pos_labels, contrastive_neg_X, contrastive_neg_labels = tf.map_fn(
            lambda i: process_sample(i),
            elems=tf.range(tf.shape(X)[0]),
            dtype=(tf.float32, tf.float32, tf.float32, tf.float32)
        )

        return contrastive_pos_X, contrastive_pos_labels, contrastive_neg_X, contrastive_neg_labels

    def _contrastive_learning(self, contrastive_pos_x, contrastive_pos_labels, contrastive_neg_x,
                              contrastive_neg_labels):
        """
        处理正样本 contrastive_pos_X 和负样本 contrastive_neg_X，生成潜在表示 contrastive_pos_z 和 contrastive_neg_z

        参数：
        - contrastive_pos_X: 正样本数据，用于对比学习
        - contrastive_pos_labels: 与 contrastive_pos_X 对应的条件标签
        - contrastive_neg_X: 负样本数据，用于对比学习
        - contrastive_neg_labels: 与 contrastive_neg_X 对应的条件标签

        返回：
        - contrastive_pos_z: 正样本的潜在表示
        - contrastive_neg_z: 负样本的潜在表示
        """
        # 正样本的编码器输入：contrastive_pos_X 和对应的条件标签 contrastive_pos_labels
        contrastive_pos_encoder_inputs = [contrastive_pos_x, contrastive_pos_labels]
        # 负样本的编码器输入：contrastive_neg_X 和对应的条件标签 contrastive_neg_labels
        contrastive_neg_encoder_inputs = [contrastive_neg_x, contrastive_neg_labels]

        # 通过编码器生成 contrastive_pos_z 和 contrastive_neg_z
        self.contrastive_pos_z = self.encoder_model(contrastive_pos_encoder_inputs)
        self.contrastive_neg_z = self.encoder_model(contrastive_neg_encoder_inputs)


    def select_samples_for_second_contrastive(self, X, encoder_labels, cell_type_labels):
        """
        选择用于第二对比学习的样本，只需要满足相同细胞类型的条件。

        参数：
        - X: 输入数据 (Tensor)
        - cell_type_labels: 细胞类型标签

        返回：
        - pos_samples: 相同细胞类型的正样本
        - pos_labels: 与正样本对应的细胞类型标签
        """
        # 定义处理每个样本的函数
        def process_sample(i):
            # 找到与当前样本细胞类型和条件标签相同的样本索引
            same_condition_indices = \
            np.where((encoder_labels != encoder_labels[i]) & (cell_type_labels == cell_type_labels[i]))[0]

            # 排除当前样本自身的索引
            same_type_indices = same_condition_indices[same_condition_indices != i]

            if len(same_condition_indices) > 0:
                # 随机选择一个相同细胞类型的样本
                chosen_index = np.random.choice(same_type_indices, 1)[0]
                contrastive_sample = X[chosen_index]
                return contrastive_sample, encoder_labels[i]
            else:
                # 如果没有足够的样本，则返回当前样本及其条件标签
                return X[i], encoder_labels[i]

        # 使用 tf.map_fn 对每个样本执行 process_sample
        pos_samples, pos_labels = tf.map_fn(
            lambda i: process_sample(i),
            elems=tf.range(tf.shape(X)[0]),
            dtype=(X.dtype, encoder_labels.dtype)
        )

        return pos_samples, pos_labels


    def _contrastive_learning_second(self, contrastive_x, contrastive_labels):
        """
        处理用于第二对比学习的正样本和负样本，生成潜在表示 z_common 和 z_specific。

        参数：
        - contrastive_x: 用于第二对比学习的样本数据
        - contrastive_labels: 与 contrastive_x 对应的条件标签

        返回：
        - contrastive_z_common: 样本的共性潜在表示
        - contrastive_z_specific: 样本的特异性潜在表示
        """
        # 编码器输入：样本和对应的条件标签
        contrastive_encoder_inputs = [contrastive_x, contrastive_labels]

        # 使用编码器生成 z
        contrastive_z = self.encoder_model(contrastive_encoder_inputs)[2]  # 获取完整的 z 表示

        # 分割 z 为共性部分和特异性部分
        contrastive_z_common = contrastive_z[:, :self.common_dim]  # 共性部分
        contrastive_z_specific = contrastive_z[:, self.common_dim:self.n_topic]  # 特异性部分

        return contrastive_z_common, contrastive_z_specific


    def to_mmd_layer(self, adata, batch_key, cell_type_labels):
        """
            Map ``adata`` in to the MMD space. This function will feed data
            in ``mmd_model`` of trVAE and compute the MMD space coordinates
            for each sample in data.

            Parameters
            ----------
            adata: :class:`~anndata.AnnData`
                Annotated data matrix to be mapped to MMD latent space.
                Please note that ``adata.X`` has to be in shape [n_obs, x_dimension]
            encoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as trVAE'sencoder condition array.
            decoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as trVAE'sdecoder condition array.

            Returns
            -------
            adata_mmd: :class:`~anndata.AnnData`
                returns Annotated data containing MMD latent space encoding of ``adata``
        """
        adata = remove_sparsity(adata)

        encoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)
        decoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)

        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        cvae_inputs = [adata.X, encoder_labels, decoder_labels, cell_type_labels]

        mmd = self.cvae_model.predict(cvae_inputs)[1]
        mmd = np.nan_to_num(mmd, nan=0.0, posinf=0.0, neginf=0.0)

        adata_mmd = anndata.AnnData(X=mmd)
        adata_mmd.obs = adata.obs.copy(deep=True)

        return adata_mmd

    def to_z_latent(self, adata, batch_key, cell_type_labels):
        """
            Map `adata` in to the latent space. This function will feed data
            in encoder part of C-VAE and compute the latent space coordinates
            for each sample in data.
            # Parameters
                adata: `~anndata.AnnData`
                    Annotated data matrix to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].
                encoder_labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.
                return_adata: boolean
                    if `True`, will output as an `anndata` object or put the results in the `obsm` attribute of `adata`
            # Returns
                output: `~anndata.AnnData`
                    returns `anndata` object containing latent space encoding of 'adata'
        """
        if sparse.issparse(adata.X):
            adata.X = adata.X.A

        encoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        # 检查 encoder_model 的输入数量
        if cell_type_labels is not None and len(self.encoder_model.inputs) == 3:
            latent = self.encoder_model.predict([adata.X, encoder_labels, cell_type_labels])[2]
        else:
            latent = self.encoder_model.predict([adata.X, encoder_labels])[2]  # 仅使用两个输入

        latent = np.nan_to_num(latent)

        latent_adata = anndata.AnnData(X=latent)
        latent_adata.obs = adata.obs.copy(deep=True)

        return latent_adata

    def get_latent(self, adata, batch_key, return_z=True):
        """ Transforms `adata` in latent space of trVAE and returns the latent
        coordinates in the annotated (adata) format.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset matrix in Primary space.
        batch_key: str
            Name of the column containing the study (batch) names for each sample.
        return_z: bool
            ``False`` by defaul. if ``True``, the output of bottleneck layer of network will be computed.

        Returns
        -------
        adata_pred: `~anndata.AnnData`
            Annotated data of transformed ``adata`` into latent space.
        """
        if set(self.gene_names).issubset(set(adata.var_names)):
            adata = adata[:, self.gene_names]
        else:
            raise Exception("set of gene names in train adata are inconsistent with trVAE'sgene_names")

        if self.beta == 0:
            return_z = True

        # 生成 cell_type_labels 确保输入完整
        # 初始化 LabelEncoder
        le = LabelEncoder()
        # 将字符串标签转换为整数编码
        integer_encoded_labels = le.fit_transform(adata.obs[self.cell_type_key])
        # 转换为 one-hot 编码
        cell_type_labels = to_categorical(integer_encoded_labels, num_classes=self.n_cell_types)

        if return_z or self.beta == 0:
            return self.to_z_latent(adata, batch_key, cell_type_labels)
        else:
            return self.to_mmd_layer(adata, batch_key, cell_type_labels)


    def predict(self, adata, condition_key, target_condition=None):
        """Feeds ``adata`` to trVAE and produces the reconstructed data.

            Parameters
            ----------
            adata: :class:`~anndata.AnnData`
                Annotated data matrix whether in primary space.
            condition_key: str
                :class:`~numpy.ndarray` of labels to be fed as trVAE'sencoder condition array.
            target_condition: str
                :class:`~numpy.ndarray` of labels to be fed as trVAE'sdecoder condition array.

            Returns
            -------
            adata_pred: `~anndata.AnnData`
                Annotated data of predicted cells in primary space.
        """
        adata = remove_sparsity(adata)

        # 编码标签
        encoder_labels, _ = label_encoder(adata, self.condition_encoder, condition_key)
        if target_condition is not None:
            decoder_labels = np.zeros_like(encoder_labels) + self.condition_encoder[target_condition]
        else:
            decoder_labels, _ = label_encoder(adata, self.condition_encoder, condition_key)

        # 将标签转换为分类格式
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        # 将细胞类型标签映射为整数索引并进行 one-hot 编码
        if hasattr(self, 'cell_type_encoder'):
            # 检查是否存在未见过的细胞类型标签
            new_labels = set(adata.obs[self.cell_type_key]) - set(self.cell_type_encoder.classes_)

            if new_labels:
                # 如果存在新的标签，添加它们到编码器的 classes_
                self.cell_type_encoder.classes_ = np.append(self.cell_type_encoder.classes_, list(new_labels))

            # 转换为整数编码
            integer_encoded_labels = self.cell_type_encoder.transform(adata.obs[self.cell_type_key])
        else:
            # 初始化并拟合新的 LabelEncoder
            self.cell_type_encoder = LabelEncoder()
            integer_encoded_labels = self.cell_type_encoder.fit_transform(adata.obs[self.cell_type_key])

        cell_type_labels = to_categorical(integer_encoded_labels, num_classes=self.n_cell_types)

        x_hat = self.cvae_model.predict([adata.X, encoder_labels, decoder_labels, cell_type_labels])[0]

        # 构造结果
        adata_pred = anndata.AnnData(X=x_hat)
        adata_pred.obs = adata.obs
        adata_pred.var_names = adata.var_names

        return adata_pred

    def restore_model_weights(self, compile=True):
        """
            restores model weights from ``model_path``.

            Parameters
            ----------
            compile: bool
                if ``True`` will compile model after restoring its weights.

            Returns
            -------
            ``True`` if the model has been successfully restored.
            ``False`` if ``model_path`` is invalid or the model weights couldn't be found in the specified ``model_path``.
        """
        if os.path.exists(os.path.join(self.model_path, f"{self.model_name}.h5")):
            self.cvae_model.load_weights(os.path.join(self.model_path, f'{self.model_name}.h5'))

            self.encoder_model = self.cvae_model.get_layer("encoder")
            self.decoder_model = self.cvae_model.get_layer("decoder")

            if compile:
                self.compile_models()
            print(f"{self.model_name}'s weights has been successfully restored!")
            return True
        return False

    def restore_model_config(self, compile=True):
        """
            restores model config from ``model_path``.

            Parameters
            ----------
            compile: bool
                if ``True`` will compile model after restoring its config.

            Returns
            -------
            ``True`` if the model config has been successfully restored.
            ``False`` if `model_path` is invalid or the model config couldn't be found in the specified ``model_path``.
        """
        if os.path.exists(os.path.join(self.model_path, f"{self.model_name}.json")):
            json_file = open(os.path.join(self.model_path, f"{self.model_name}.json"), 'rb')
            loaded_model_json = json_file.read()
            self.cvae_model = model_from_json(loaded_model_json)
            self.encoder_model = self.cvae_model.get_layer("encoder")
            self.decoder_model = self.cvae_model.get_layer("decoder")

            if compile:
                self.compile_models()

            print(f"{self.model_name}'s network's config has been successfully restored!")
            return True
        else:
            return False

    def restore_class_config(self, compile_and_consturct=True):
        """
            restores class' config from ``model_path``.

            Parameters
            ----------
            compile_and_consturct: bool
                if ``True`` will construct and compile model from scratch.

            Returns
            -------
            ``True`` if the scNet config has been successfully restored.
            ``False`` if `model_path` is invalid or the class' config couldn't be found in the specified ``model_path``.
        """
        import json
        if os.path.exists(os.path.join(self.model_path, f"{self.class_name}.json")):
            with open(os.path.join(self.model_path, f"{self.class_name}.json"), 'rb') as f:
                trVAE_config = json.load(f)

            # Update network_kwargs and training_kwargs dictionaries
            for key, value in trVAE_config.items():
                if key in self.network_kwargs.keys():
                    self.network_kwargs[key] = value
                elif key in self.training_kwargs.keys():
                    self.training_kwargs[key] = value

            # Update class attributes
            for key, value in trVAE_config.items():
                setattr(self, key, value)

            if compile_and_consturct:
                self.construct_network()
                self.compile_models()

            print(f"{self.class_name}'s config has been successfully restored!")
            return True
        else:
            return False

    def save(self, make_dir=True):
        """
            Saves all model weights, configs, and hyperparameters in the ``model_path``.

            Parameters
            ----------
            make_dir: bool
                Whether makes ``model_path`` directory if it does not exists.

            Returns
            -------
            ``True`` if the model has been successfully saved.
            ``False`` if ``model_path`` is an invalid path and ``make_dir`` is set to ``False``.
        """
        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            self.save_model_weights(make_dir)
            self.save_model_config(make_dir)
            self.save_class_config(make_dir)
            print(f"\n{self.class_name} has been successfully saved in {self.model_path}.")
            return True
        else:
            return False

    def save_model_weights(self, make_dir=True):
        """
            Saves model weights in the ``model_path``.

            Parameters
            ----------
            make_dir: bool
                Whether makes ``model_path`` directory if it does not exists.

            Returns
            -------
            ``True`` if the model has been successfully saved.
            ``False`` if ``model_path`` is an invalid path and ``make_dir`` is set to ``False``.
        """
        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            self.cvae_model.save_weights(os.path.join(self.model_path, f"{self.model_name}.h5"),
                                         overwrite=True)
            return True
        else:
            return False

    def save_model_config(self, make_dir=True):
        """
            Saves model's config in the ``model_path``.

            Parameters
            ----------
            make_dir: bool
                Whether makes ``model_path`` directory if it does not exists.

            Returns
            -------
            ``True`` if the model has been successfully saved.
            ``False`` if ``model_path`` is an invalid path and ``make_dir`` is set to ``False``.
        """
        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            model_json = self.cvae_model.to_json()
            with open(os.path.join(self.model_path, f"{self.model_name}.json"), 'w') as file:
                file.write(model_json)
            return True
        else:
            return False

    def save_class_config(self, make_dir=True):
        """
            Saves class' config in the ``model_path``.

            Parameters
            ----------
            make_dir: bool
                Whether makes ``model_path`` directory if it does not exists.

            Returns
            -------
                ``True`` if the model has been successfully saved.
                ``False`' if ``model_path`` is an invalid path and ``make_dir`` is set to ``False``.
        """
        import json

        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            config = {"gene_size": self.gene_size,
                      "n_topic": self.n_topic,
                      "n_conditions": self.n_conditions,
                      "condition_encoder": self.condition_encoder,
                      "gene_names": self.gene_names}
            all_configs = dict(list(self.network_kwargs.items()) +
                               list(self.training_kwargs.items()) +
                               list(config.items()))
            with open(os.path.join(self.model_path, f"{self.class_name}.json"), 'w') as f:
                json.dump(all_configs, f)

            return True
        else:
            return False

    def _fit(self, adata,
             condition_key, train_size=0.8,
             n_epochs=300, batch_size=512,
             early_stop_limit=10, lr_reducer=7,
             save=True, retrain=True, verbose=3):
        train_adata, valid_adata = train_test_split(adata, train_size)

        if self.gene_names is None:
            self.gene_names = train_adata.var_names.tolist()
        else:
            if set(self.gene_names).issubset(set(train_adata.var_names)):
                train_adata = train_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in train adata are inconsistent with class' gene_names")

            if set(self.gene_names).issubset(set(valid_adata.var_names)):
                valid_adata = valid_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in valid adata are inconsistent with class' gene_names")

        train_expr = train_adata.X.A if sparse.issparse(train_adata.X) else train_adata.X
        valid_expr = valid_adata.X.A if sparse.issparse(valid_adata.X) else valid_adata.X

        train_conditions_encoded, self.condition_encoder = label_encoder(train_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        valid_conditions_encoded, self.condition_encoder = label_encoder(valid_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        if not retrain and os.path.exists(os.path.join(self.model_path, f"{self.model_name}.h5")):
            self.restore_model_weights()
            return

        callbacks = [
            History(),
        ]

        if verbose > 2:
            callbacks.append(
                LambdaCallback(on_epoch_end=lambda epoch, logs: print_progress(epoch, logs, n_epochs)))
            fit_verbose = 0
        else:
            fit_verbose = verbose

        if early_stop_limit > 0:
            callbacks.append(EarlyStopping(patience=early_stop_limit, monitor='val_loss'))

        if lr_reducer > 0:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', patience=lr_reducer))

        train_conditions_onehot = to_categorical(train_conditions_encoded, num_classes=self.n_conditions)
        valid_conditions_onehot = to_categorical(valid_conditions_encoded, num_classes=self.n_conditions)

        x_train = [train_expr, train_conditions_onehot, train_conditions_onehot]
        x_valid = [valid_expr, valid_conditions_onehot, valid_conditions_onehot]

        y_train = [train_expr, train_conditions_encoded]
        y_valid = [valid_expr, valid_conditions_encoded]

        self.cvae_model.fit(x=x_train,
                            y=y_train,
                            validation_data=(x_valid, y_valid),
                            epochs=n_epochs,
                            batch_size=batch_size,
                            verbose=fit_verbose,
                            callbacks=callbacks,
                            )
        if save:
            self.save(make_dir=True)


    def _train_on_batch(self, adata,
                        condition_key, train_size=0.8,
                        n_epochs=300, batch_size=512,
                        early_stop_limit=10, lr_reducer=7,
                        save=True, retrain=True, verbose=3):
        train_adata, valid_adata = train_test_split(adata, train_size)

        # 确保基因名的一致性
        if self.gene_names is None:
            self.gene_names = train_adata.var_names.tolist()
        else:
            if set(self.gene_names).issubset(set(train_adata.var_names)):
                train_adata = train_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in train adata are inconsistent with class' gene_names")
            if set(self.gene_names).issubset(set(valid_adata.var_names)):
                valid_adata = valid_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in valid adata are inconsistent with class' gene_names")

        # 编码 condition 标签
        train_conditions_encoded, self.condition_encoder = label_encoder(train_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)
        valid_conditions_encoded, self.condition_encoder = label_encoder(valid_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        # 检查是否需要重新训练
        if not retrain and os.path.exists(os.path.join(self.model_path, f"{self.model_name}.h5")):
            self.restore_model_weights()
            return

        # One-hot 编码条件标签
        train_conditions_onehot = to_categorical(train_conditions_encoded, num_classes=self.n_conditions)
        valid_conditions_onehot = to_categorical(valid_conditions_encoded, num_classes=self.n_conditions)

        # 检查稀疏性
        if sparse.issparse(train_adata.X):
            is_sparse = True
        else:
            is_sparse = False

        train_expr = train_adata.X
        valid_expr = valid_adata.X.A if is_sparse else valid_adata.X

        # 为验证集生成 cell_type 标签
        valid_cell_type_encoded, _ = label_encoder(valid_adata, condition_key=self.cell_type_key)
        valid_cell_type_onehot = to_categorical(valid_cell_type_encoded, num_classes=self.n_cell_types)

        # 添加 cell_type 标签到 x_valid
        x_valid = [valid_expr, valid_conditions_onehot, valid_conditions_onehot, valid_cell_type_onehot]

        # 设置 y_valid
        if self.loss_fn in ['nb', 'zinb']:
            x_valid.append(valid_adata.obs[self.size_factor_key].values)
            contrastive_placeholder = np.zeros((valid_expr.shape[0], self.n_topic))
            y_valid = [valid_adata.raw.X.A if sparse.issparse(valid_adata.raw.X) else valid_adata.raw.X,
                       valid_conditions_encoded, contrastive_placeholder, contrastive_placeholder]
        else:
            contrastive_placeholder = np.zeros((valid_expr.shape[0], self.n_topic))
            y_valid = [valid_expr, valid_conditions_encoded, contrastive_placeholder, contrastive_placeholder]

        # Early stopping 设置
        es_patience, best_val_loss = 0, 1e10
        for i in range(n_epochs):
            train_loss = train_recon_loss = train_mmd_loss = train_contrastive_loss = train_second_contrastive_loss = 0.0
            for j in range(min(200, train_adata.shape[0] // batch_size)):
                batch_indices = np.random.choice(train_adata.shape[0], batch_size)
                batch_expr = train_expr[batch_indices, :].A if is_sparse else train_expr[batch_indices, :]

                # 动态生成当前批次的细胞类型标签，用于对比学习
                batch_cell_type_encoded = label_encoder(train_adata, le=None, condition_key=self.cell_type_key)[0][
                    batch_indices]
                batch_cell_type_onehot = to_categorical(batch_cell_type_encoded, num_classes=self.n_cell_types)

                # 添加 cell_type 标签到 x_train
                x_train = [batch_expr, train_conditions_onehot[batch_indices], train_conditions_onehot[batch_indices],
                           batch_cell_type_onehot]

                # 设置 y_train
                if self.loss_fn in ['nb', 'zinb']:
                    x_train.append(train_adata.obs[self.size_factor_key].values[batch_indices])
                    # 添加一个与对比输出匹配的占位符
                    contrastive_placeholder = np.zeros((batch_size, self.n_topic))
                    y_train = [train_adata.raw.X[batch_indices].A if sparse.issparse(
                        train_adata.raw.X[batch_indices]) else train_adata.raw.X[batch_indices],
                               train_conditions_encoded[batch_indices], contrastive_placeholder, contrastive_placeholder]
                else:
                    # 创建与 self.z 形状匹配的占位符
                    contrastive_placeholder = np.random.uniform(low=0.01, high=0.1, size=(batch_size, self.n_topic))
                    y_train = [batch_expr, train_conditions_encoded[batch_indices], contrastive_placeholder, contrastive_placeholder]

                # 训练批次
                batch_loss, batch_recon_loss, batch_kl_loss, batch_contrastive_loss, batch_second_contrastive_loss = self.cvae_model.train_on_batch(x_train, y_train)

                train_loss += batch_loss / batch_size
                train_recon_loss += batch_recon_loss / batch_size
                train_mmd_loss += batch_kl_loss / batch_size
                train_contrastive_loss += batch_contrastive_loss / batch_size
                train_second_contrastive_loss += batch_second_contrastive_loss / batch_size

            # 计算验证损失
            valid_loss, valid_recon_loss, valid_mmd_loss, valid_contrastive_loss, valid_second_contrastive_loss = self.cvae_model.evaluate(x_valid, y_valid, verbose=0)

            # Early Stopping 机制
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                es_patience = 0
            else:
                es_patience += 1
                if es_patience == early_stop_limit:
                    print("Training stopped with Early Stopping")
                    break

            # 打印日志
            logs = {"loss": train_loss, "recon_loss": train_recon_loss, "mmd_loss": train_mmd_loss,
                    "contrastive_loss": train_contrastive_loss,
                    "second_contrastive_loss": train_second_contrastive_loss,
                    "val_loss": valid_loss, "val_recon_loss": valid_recon_loss, "val_mmd_loss": valid_mmd_loss,
                    "val_contrastive_loss": valid_contrastive_loss,
                    "val_second_contrastive_loss": valid_second_contrastive_loss}
            print_progress(i, logs, n_epochs)

        # 保存模型
        if save:
            self.save(make_dir=True)


    def train(self, adata,
              condition_key, train_size=0.8,
              n_epochs=200, batch_size=128,
              early_stop_limit=10, lr_reducer=8,
              save=True, retrain=True, verbose=3):

        """
            Trains the network with ``n_epochs`` times given ``adata``.
            This function is using ``early stopping`` and ``learning rate reduce on plateau``
            techniques to prevent over-fitting.
            Parameters
            ----------
            adata: :class:`~anndata.AnnData`
                Annotated dataset used to train & evaluate scNet.
            condition_key: str
                column name for conditions in the `obs` matrix of `train_adata` and `valid_adata`.
            train_size: float
                fraction of samples in `adata` used to train scNet.
            n_epochs: int
                number of epochs.
            batch_size: int
                number of samples in the mini-batches used to optimize scNet.
            early_stop_limit: int
                patience of EarlyStopping
            lr_reducer: int
                patience of LearningRateReduceOnPlateau.
            save: bool
                Whether to save scNet after the training or not.
            verbose: int
                Verbose level
            retrain: bool
                ``True`` by default. if ``True`` scNet will be trained regardless of existance of pre-trained scNet in ``model_path``. if ``False`` scNet will not be trained if pre-trained scNet exists in ``model_path``.

        """

        if self.device == 'gpu':
            return self._fit(adata, condition_key, train_size, n_epochs, batch_size, early_stop_limit,
                             lr_reducer, save, retrain, verbose)
        else:
            return self._train_on_batch(adata, condition_key, train_size, n_epochs, batch_size,
                                        early_stop_limit, lr_reducer, save, retrain,
                                        verbose)