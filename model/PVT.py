

import tensorflow as tf
from model.droppath import droppath
import numpy as np
import tensorflow_addons as tfa
import logging
def get_patch_embed_model(img_size,first_level_patch_size=4,embed_dims=[64, 128, 320, 512]):
    num_level = len(embed_dims)
    patch_size = [first_level_patch_size] + [2 for _ in range(num_level-1)]
    embed_dims = [3] + embed_dims
    outputs = []
    feat_size_list = []
    feat_size = np.array(img_size)

    for i in range(1,len(embed_dims)):
        input = tf.keras.layers.Input([feat_size[0]*feat_size[1],embed_dims[i-1]])
        x = tf.reshape(input,(tf.shape(input)[0],feat_size[0],feat_size[1],embed_dims[i-1]))
        x = tf.keras.layers.Conv2D(embed_dims[i],patch_size[i-1],patch_size[i-1])(x)
        x = tf.reshape(x,(tf.shape(x)[0],-1,embed_dims[i]))
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        outputs.append(tf.keras.Model(inputs=input,outputs=x))
        feat_size = feat_size // patch_size[i - 1]
        feat_size_list.append(feat_size.tolist())
    return outputs,feat_size_list


def get_attention_model(feat_size,num_patch,embed_dims,num_heads,sr_ratio,attention_drop_rate,drop_rate,name=None):
    input = tf.keras.layers.Input((num_patch, embed_dims))
    q_x = tf.keras.layers.Dense(embed_dims, use_bias=True)(input)
    q_x = tf.reshape(q_x, (tf.shape(input)[0], -1, num_heads, embed_dims // num_heads))
    q = tf.transpose(q_x, perm=(0, 2, 1, 3))
    if sr_ratio > 1:
        x = tf.reshape(input, (tf.shape(input)[0], feat_size[0], feat_size[1], input.shape[-1]))
        x = tf.keras.layers.Conv2D(embed_dims, sr_ratio, sr_ratio)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Dense(embed_dims * 2, use_bias=True)(x)
        x = tf.reshape(x, (tf.shape(x)[0], -1, 2, num_heads, embed_dims // num_heads))
        kv = tf.transpose(x, perm=(2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]
    else:
        k = tf.keras.layers.Dense(embed_dims, use_bias=True)(input)
        k = tf.reshape(k, (tf.shape(k)[0], -1, num_heads, embed_dims // num_heads))
        k = tf.transpose(k, perm=(0, 2, 1, 3))
        v = tf.keras.layers.Dense(embed_dims, use_bias=True)(input)
        v = tf.reshape(v, (tf.shape(v)[0], -1, num_heads, embed_dims // num_heads))
        v = tf.transpose(v, perm=(0, 2, 1, 3))

    head_dim = embed_dims // num_heads
    x = tf.matmul(q, tf.transpose(k, perm=(0, 1, 3, 2))) / tf.math.sqrt(tf.cast(head_dim, tf.dtypes.float32))
    x = tf.math.softmax(x, axis=-1)
    score = tf.keras.layers.Dropout(attention_drop_rate)(x)
    x = tf.matmul(score, v)
    x = tf.transpose(x, perm=(0, 2, 1, 3))
    x = tf.reshape(x, (tf.shape(x)[0], -1, embed_dims))
    x = tf.keras.layers.Dense(embed_dims, use_bias=True)(x)
    output = tf.keras.layers.Dropout(drop_rate)(x)
    return tf.keras.Model(input, output, name=name)


def get_block_attention_model(depth,block_drop_path_rate,mlp_ratio,feat_size,num_patch,embed_dims,num_heads,sr_ratio,attention_drop_rate,drop_rate,name=None):
    block_index = 0
    block_input = input = tf.keras.layers.Input([num_patch, embed_dims])
    for _ in range(depth):
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(input)
        x = get_attention_model(feat_size,num_patch,embed_dims,num_heads,sr_ratio,attention_drop_rate,drop_rate)(x)
        attention_output = input + droppath(x, drop_prob=block_drop_path_rate[block_index])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output)
        num_mlp_hidden_layers = int(input.shape[-1] * mlp_ratio)
        x = tf.keras.layers.Dense(num_mlp_hidden_layers, activation=tfa.activations.gelu)(x)
        x = tf.keras.layers.Dropout(drop_rate)(x)
        x = tf.keras.layers.Dense(input.shape[-1])(x)
        x = tf.keras.layers.Dropout(drop_rate)(x)
        mlp_output = attention_output + droppath(x, drop_prob=block_drop_path_rate[block_index])
        input = mlp_output
        block_index += 1
    return tf.keras.Model(block_input, mlp_output, name=name)



class AddClsToken(tf.keras.layers.Layer):
    def __init__(self):
        super(AddClsToken, self).__init__()

    def build(self, input_shape):
        self.cls_token = self.add_weight(shape=(1,1,input_shape[-1]),
                                         initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),trainable=True,dtype=tf.dtypes.float32)

    def call(self, x):
        cls_token = tf.broadcast_to(self.cls_token, [tf.shape(x)[0], 1, self.cls_token.shape[-1]])
        return tf.concat([cls_token,x],axis=1)

class AddPosEmbed(tf.keras.layers.Layer):
    def __init__(self,img_len):
        super(AddPosEmbed, self).__init__()
        self.img_len = img_len

    def build(self, input_shape):
        self.pos_embed = self.add_weight(shape=[1,self.img_len,input_shape[-1]],
                                         initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),trainable=True,dtype=tf.dtypes.float32)
    def call(self, x):
        return x+self.pos_embed

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'img_len': self.img_len,
        })
        return config

def get_pvt(img_size,num_classes,block_depth,mlp_ratio,drop_path_rate,first_level_patch_size,embed_dims,num_heads,sr_ratio,attention_drop_rate,drop_rate):
    block_drop_path_rate = np.linspace(0, drop_path_rate, sum(block_depth))
    block_depth_index = 0

    input = tf.keras.layers.Input((img_size[0], img_size[1], 3))
    patch_embed_model_list,feat_size_list = get_patch_embed_model(img_size,first_level_patch_size,embed_dims)
    x = input
    x = tf.reshape(x,[tf.shape(input)[0],-1,tf.shape(input)[-1]])
    for i in range(len(patch_embed_model_list)):
        x = patch_embed_model_list[i](x)
        if i == len(patch_embed_model_list)-1:
            num_patch = feat_size_list[i][0] * feat_size_list[i][1] + 1
            x = AddClsToken()(x)
            x = AddPosEmbed(num_patch)(x)
        else:
            num_patch = feat_size_list[i][0] * feat_size_list[i][1]
            x = AddPosEmbed(num_patch)(x)
        x = tf.keras.layers.Dropout(drop_rate)(x)
        x = get_block_attention_model(block_depth[i],block_drop_path_rate[block_depth_index:block_depth_index+block_depth[i]],mlp_ratio[i],
                                      feat_size_list[i],num_patch,embed_dims[i],num_heads[i],sr_ratio[i],attention_drop_rate,drop_rate)(x)
        block_depth_index+=block_depth[i]
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x[:,0])
    return tf.keras.Model(input,output,name='PVT')


class PVTNet():
    def __init__(self, img_size=224, classes=2,type='tiny',pretrain=None):
        self.img_size = img_size
        self.classes = classes
        self.pretrain = pretrain
        self.type = type
    def get_model(self):
        if self.type=='tiny':
            model = get_pvt(img_size=(self.img_size,self.img_size),num_classes=self.classes,
                            first_level_patch_size=4,
                            block_depth=[2, 2, 2, 2], mlp_ratio=[8, 8, 4, 4],
                            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],sr_ratio=[8, 4, 2, 1],
                            drop_path_rate=0.1,attention_drop_rate=0.0,drop_rate=0.02)
        elif self.type=='small':
            model = get_pvt(img_size=(self.img_size,self.img_size),num_classes=self.classes,
                            first_level_patch_size=4,
                            block_depth=[3, 4, 6, 3], mlp_ratio=[8, 8, 4, 4],
                            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],sr_ratio=[8, 4, 2, 1],
                            drop_path_rate=0.1,attention_drop_rate=0.0,drop_rate=0.0)

        elif self.type=='medium':
            model = get_pvt(img_size=(self.img_size,self.img_size),num_classes=self.classes,
                            first_level_patch_size=4,
                            block_depth=[3, 4, 18, 3], mlp_ratio=[8, 8, 4, 4],
                            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],sr_ratio=[8, 4, 2, 1],
                            drop_path_rate=0.1,attention_drop_rate=0.0,drop_rate=0.0)
        elif self.type=='large':
            model = get_pvt(img_size=(self.img_size,self.img_size),num_classes=self.classes,
                            first_level_patch_size=4,
                            block_depth=[3, 8, 27, 3], mlp_ratio=[8, 8, 4, 4],
                            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],sr_ratio=[8, 4, 2, 1],
                            drop_path_rate=0.1,attention_drop_rate=0.0,drop_rate=0.02)
        else:
            raise ValueError('Unsupported PVT type:{}'.format(self.type))
        try:
            model.load_weights(self.pretrain)
        except:
            logging.warning('Failed to load weights file:{}'.format(self.pretrain))

        return model
