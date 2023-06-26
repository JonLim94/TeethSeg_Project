import tensorflow as tf # Tensorflow 2.12.0
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, BatchNormalization, Activation, LeakyReLU, Reshape, ReLU
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import OneHotMeanIoU, MeanIoU


class EdgeFeatureLayer(tf.keras.layers.Layer):
    def __init__(self, k=16,dtype=tf.float32,**kwargs):
        super(EdgeFeatureLayer, self).__init__(dtype=dtype,**kwargs)
        self.k = k
    
    def call(self, X_inputs, nn_idx):
        og_batch_size = X_inputs.get_shape().as_list()[0]
        X_inputs = tf.squeeze(X_inputs)
        if og_batch_size == 1:
            X_inputs = tf.expand_dims(X_inputs, 0)
        
        mesh_central = X_inputs
        mesh_shape = X_inputs.shape
        batch_size = mesh_shape[0]
        num_points = mesh_shape[1]
        num_dims = mesh_shape[2]
        
        idx_ = tf.range(batch_size) * num_points
        idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 
        
        mesh_flat = tf.reshape(X_inputs, [-1, num_dims])
        mesh_neighbors = tf.gather(mesh_flat, nn_idx+idx_)
        mesh_central = tf.expand_dims(mesh_central, axis=-2)
    
        mesh_central = tf.tile(mesh_central, [1, 1, self.k, 1])
    
        edge_feature = tf.concat([mesh_central, mesh_neighbors-mesh_central], axis=-1)
        return edge_feature
    
    def get_config(self):
        config = super(EdgeFeatureLayer, self).get_config()
        config.update({"k": self.k})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class KNNLayer(tf.keras.layers.Layer):
    def __init__(self, k=16, dtype=tf.float32, **kwargs):
        super(KNNLayer, self).__init__(dtype=dtype,**kwargs)
        self.k = k
    
    def call(self, inputs):
        P_inputs = inputs

        mesh_transpose = tf.transpose(P_inputs, perm=[0, 2, 1])
        mesh_inner = tf.matmul(P_inputs, mesh_transpose)
        mesh_inner = -2*mesh_inner
        mesh_square = tf.reduce_sum(tf.square(P_inputs), axis=-1, keepdims=True)
        mesh_square_tranpose = tf.transpose(mesh_square, perm=[0, 2, 1])
        adj_matrix = mesh_square + mesh_inner + mesh_square_tranpose
        
        neg_adj = -adj_matrix
        _, nn_idx = tf.nn.top_k(neg_adj, k=self.k)
        return nn_idx
    
    def get_config(self):
        config = super(KNNLayer, self).get_config()
        config.update({"k": self.k})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Offset_Loss(tf.keras.losses.Loss):
    def __init__(self, N=10000):
        super().__init__()
        self.N = N
  
    def call(self, y_true, y_pred):
        # y_true here is (c^i - ci)/delta
        L_offset = (tf.norm(y_pred - y_true,ord=2))/self.N
        return L_offset
    
    def get_config(self):
        return {"N": self.N}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# # Building the Neural Network
def createModel(face_count = 10000,
                BATCH_SIZE=2,
                k=16):
    ipt_X_inputs = Input(shape = (face_count, 15),batch_size=BATCH_SIZE,dtype=tf.float32)
    ipt_P_inputs = Input(shape = (face_count, 3),batch_size=BATCH_SIZE,dtype=tf.float32)

	
    nn_idx = KNNLayer(k=k)(ipt_P_inputs)

    net = EdgeFeatureLayer(k=k)(ipt_X_inputs, nn_idx)
    
    net = Conv2D(64,kernel_size=1,use_bias=False,name="Edge_ConV1a")(net)
    net = BatchNormalization(momentum=0.9)(net)
    net = LeakyReLU(0.2)(net)
    net = Conv2D(64,kernel_size=1,use_bias=False,name="Edge_ConV1b")(net)
    net = BatchNormalization(momentum=0.9)(net)
    net = LeakyReLU(0.2)(net)
    net = tf.reduce_max(net, axis=-2, keepdims=True)
    net1 = net
    
    net = EdgeFeatureLayer(k=k)(net,nn_idx)
    
    net = Conv2D(64,kernel_size=1,use_bias=False,name="Edge_ConV2a")(net)
    net = BatchNormalization(momentum=0.9)(net)
    net = LeakyReLU(0.2)(net)
    net = Conv2D(64,kernel_size=1,use_bias=False,name="Edge_ConV2b")(net)
    net = BatchNormalization(momentum=0.9)(net)
    net = LeakyReLU(0.2)(net)
    net = tf.reduce_max(net, axis=-2, keepdims=True)
    net2 = net
    
    net = EdgeFeatureLayer(k=k)(net,nn_idx)
    
    net = Conv2D(64,kernel_size=1,use_bias=False,name="Edge_ConV3")(net)
    net = BatchNormalization(momentum=0.9)(net)
    net = LeakyReLU(0.2)(net)
    net = tf.reduce_max(net, axis=-2, keepdims=True)
    net3 = net
    
    net_192 = tf.concat([net1, net2, net3], axis=-1)

    net = Conv2D(1024,kernel_size=1,use_bias=False,name="Flattening_ConV")(net_192)
    net = BatchNormalization(momentum=0.9)(net)
    net = LeakyReLU(0.2)(net)
    
    net = tf.reduce_mean(net, axis=1, keepdims=True)
    net_1024 = net

    repeating = tf.tile(net_1024, [1, face_count, 1,1])
    
    feature_output = tf.concat([net_192, repeating],axis=-1)
    
    # Semantic Branch
    net_s = Conv2D(256,kernel_size=1,use_bias=False,name="semantic_branch")(feature_output)    
    net_s = BatchNormalization(momentum=0.9)(net_s)
    net_s = LeakyReLU(0.2)(net_s)
    semantic_branch = net_s
    
    # THP Input Generator
    net = Conv2D(256,kernel_size=1,use_bias=False,name="offset_256")(feature_output)    
    net = BatchNormalization(momentum=0.9)(net)
    net = LeakyReLU(0.2)(net)
    net = Dropout((0.5))(net)
    net = Conv2D(128,kernel_size=1,use_bias=False,name="offset_128")(net)
    net = BatchNormalization(momentum=0.9)(net)
    net = LeakyReLU(0.2)(net)
    net = Dropout((0.5))(net)
    net = Conv2D(3,kernel_size=1,activation = None,name="offset_3")(net)
    
    net = tf.reshape(net,[BATCH_SIZE,face_count,3])
    net = tf.cast(net,dtype=tf.float32,name="offset_output")
    offset_branch = net
    
    P_offsets_comb = ipt_P_inputs + 6*net
    
    # THP Network
    nn_idx2 = KNNLayer(k=k)(P_offsets_comb)
    
    net = EdgeFeatureLayer(k=k)(semantic_branch, nn_idx2)
    net = Conv2D(256,kernel_size=1,use_bias=False,name="Dyn_Edge_ConV")(net)
    net = BatchNormalization(momentum=0.9)(net)
    net = LeakyReLU(0.2)(net)
    net = tf.reduce_max(net, axis=-2, keepdims=True)
    net = Conv2D(256,kernel_size=1,use_bias=False,name="Dense_256")(net)
    net = BatchNormalization(momentum=0.9)(net)
    net = LeakyReLU(0.2)(net)
    net = Dropout((0.5))(net)
    net = Conv2D(128,kernel_size= 1,use_bias=False,name="Dense_128_2")(net)
    net = BatchNormalization(momentum=0.9)(net)
    net = LeakyReLU(0.2)(net)
    net = Dropout((0.5))(net)
    net = Conv2D(17, kernel_size=1,activation ='softmax',name="pre_semantic_output")(net)
    semantic_probabilities = Reshape((face_count,17),name="semantic_output")(net)
    
    model = Model(inputs=[ipt_X_inputs, ipt_P_inputs], outputs=[semantic_probabilities,offset_branch])
    
    model.compile(loss=['categorical_crossentropy',Offset_Loss(N=100)], 
                  optimizer='adam', 
                  metrics=[OneHotMeanIoU(num_classes=17,name='MeanIoU'),'accuracy'])
    return model
    

if __name__ == "main":
    pass