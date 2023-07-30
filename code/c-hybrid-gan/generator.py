import tensorflow as tf
from relational_memory import *
from utils import *

"""## Generator"""

class Generator(tf.keras.models.Model):
    def __init__(self, emb_units, proj_units, emb_dropout_rate, proj_dropout_rate, 
               mem_slots, head_size, num_heads, num_blocks, num_tokens, num_meta_features):
        super(Generator, self).__init__()

        self.p_subg = SubGenerator(emb_units[0], proj_units[0], emb_dropout_rate[0], 
                                   proj_dropout_rate[0], mem_slots[0], head_size[0], 
                                   num_heads[0], num_blocks[0], num_tokens[0], 
                                   num_meta_features)

        self.d_subg = SubGenerator(emb_units[1], proj_units[1], emb_dropout_rate[1], 
                                   proj_dropout_rate[1], mem_slots[1], head_size[1], 
                                   num_heads[1], num_blocks[1], num_tokens[1], 
                                   num_meta_features)

        self.r_subg = SubGenerator(emb_units[2], proj_units[2], emb_dropout_rate[2], 
                                   proj_dropout_rate[2], mem_slots[2], head_size[2], 
                                   num_heads[2], num_blocks[2], num_tokens[2], 
                                   num_meta_features)

    def call(self, inputs, memory, training=False):
        """
        :param inputs: meta and note data
        :type inputs : tuple
        :shape inputs: ([None, NUM_META_FEATURES], 
                        ([None, NUM_P_TOKENS], [None, NUM_D_TOKENS], [None, NUM_R_TOKENS]))

        :param memory: memory for rmc core for each sub generator
        :type  memory: tuple
        """

        m, (p, d, r) = inputs
        memory_p, memory_d, memory_r = memory

        p, memory_p = self.p_subg((m, p), memory_p, training=training)
        d, memory_d = self.d_subg((m, d), memory_d, training=training)
        r, memory_r = self.r_subg((m, r), memory_r, training=training)

        return (p, d, r), (memory_p, memory_d, memory_r)

"""### SubGenerator"""

class SubGenerator(tf.keras.models.Model):
    def __init__(self, emb_units, proj_units, emb_dropout_rate, proj_dropout_rate, 
               mem_slots, head_size, num_heads, num_blocks, num_tokens, num_meta_features):
        super(SubGenerator, self).__init__()
        self.proj_units = proj_units
        self.embedding = tf.keras.layers.Dense(
            emb_units, use_bias=False, kernel_initializer=create_linear_initializer(num_tokens))

        self.embedding_dropout = tf.keras.layers.Dropout(emb_dropout_rate)

        self.projection = tf.keras.layers.Dense(
            proj_units, activation='relu', kernel_initializer=create_linear_initializer(emb_units+num_meta_features))

        self.projection_dropout = tf.keras.layers.Dropout(proj_dropout_rate)

        self.lstm = tf.keras.layers.RNN(
        tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(proj_units) for _ in range(2)]),
        return_sequences=True,
        return_state=True)
        # self.rmc = RelationalMemory(mem_slots, head_size, num_heads, num_blocks)

        self.outputs = tf.keras.layers.Dense(
            num_tokens, kernel_initializer=create_linear_initializer(mem_slots*head_size*num_heads))
    
    def call(self, inputs, memory, training=False):
        """
        :param inputs: meta (i.e. syllable) and note attribute (pitch, duration or rest)
        :type. inputs: tuple
        :shape inputs: ([None, NUM_META_FEATURES], [None, NUM_[.]_TOKENS])

        :param memory: rmc memory
        """
        m, n = inputs

        n = self.embedding(n)
        n = self.embedding_dropout(n, training=training)

        x = tf.concat([n, m], axis=-1) # [p+d+r+m]

        x = self.projection(x)
        x = self.projection_dropout(x, training=training)

        x, *memory = self.lstm(tf.expand_dims(x, 1), initial_state=memory)

        x = self.outputs(tf.squeeze(x, 1))

        return x, memory

    def initial_state(self, batch_size):
        """
        This method returns initial state of lstm layers.
        """
        return [[tf.zeros((batch_size, self.proj_units)), tf.zeros((batch_size, self.proj_units))] for _ in range(2)]



class Generator_attention(tf.keras.models.Model):
    def __init__(self, emb_units, proj_units, emb_dropout_rate, proj_dropout_rate,
                 mem_slots, head_size, num_heads, num_blocks, num_tokens, num_meta_features):
        super(Generator_attention, self).__init__()

        self.subg = SubGenerator_attention(emb_units, proj_units, emb_dropout_rate, proj_dropout_rate,
                                           mem_slots, head_size, num_heads, num_blocks, num_tokens)

    def call(self, inputs, training=False):
        """
        :param inputs: meta and note data
        :type inputs : tuple
        :shape inputs: ([None, NUM_META_FEATURES],
                        ([None, NUM_P_TOKENS], [None, NUM_D_TOKENS], [None, NUM_R_TOKENS]))

        :param memory: memory for rmc core for each sub generator
        :type  memory: tuple
        """

        lyric = inputs

        out, attr_out = self.subg(lyric, training=training)

        return out, attr_out


class SubGenerator_attention(tf.keras.models.Model):
    def __init__(self, emb_units, proj_units, emb_dropout_rate, proj_dropout_rate,
                 mem_slots, head_size, num_heads, num_blocks, num_tokens):
        super(SubGenerator_attention, self).__init__()
        self.num_block = num_blocks
        self.attention = []
        self.num_token = num_tokens

        self.embedding_attr = SubGenerator_embedding(emb_units, emb_dropout_rate, proj_units, proj_dropout_rate)

        for i in range(num_blocks):
            self.attention.append(attention(proj_units, proj_dropout_rate, num_heads, num_blocks))

        self.out = tf.keras.layers.Dense(num_tokens, kernel_initializer='he_uniform',)

    def call(self, inputs, training=False):
        lyric = inputs
        batch_size = lyric.shape[0]
        attr_out = self.embedding_attr(lyric, training=training)

        for i in range(self.num_block):
            # attr_out, batch_m = self.attention[i]((attr_out, key), training=training)
            attr_out = self.attention[i](attr_out, training=training)
            # attr_out= self.attention[i](attr_out, training=training)

        out = self.out(attr_out)
        return out, attr_out

import numpy as np

"""### SubGenerator"""
class SubGenerator_embedding(tf.keras.models.Model):
    def __init__(self, emb_units, emb_dropout_rate, proj_units, proj_dropout_rate):
        super(SubGenerator_embedding, self).__init__()
        self.embedding = tf.keras.layers.Dense(emb_units, kernel_initializer='he_uniform', activation=tf.nn.relu)
        self.dropout_embedding = tf.keras.layers.Dropout(emb_dropout_rate)


        encoded_vec = np.array([[pos/10000 ** (2*i/emb_units) for pos in range(20)] for i in range(emb_units)],
                                  dtype=np.float32)
        encoded_vec[::2] = tf.sin(encoded_vec[::2])
        encoded_vec[1::2] = tf.cos(encoded_vec[1::2])


        self.position_embedding = tf.Variable(encoded_vec, trainable=False, dtype=tf.float32)
        self.projection = tf.keras.layers.Dense(proj_units, kernel_initializer='he_uniform', activation=tf.nn.relu)
        self.proj_dropout = tf.keras.layers.Dropout(proj_dropout_rate)
        self.BN_feedward = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        # attr, batch_m = inputs
        batch_m = inputs

        batch_size = batch_m.shape[0]

        pos_emb = tf.tile(self.position_embedding[tf.newaxis, ...], [batch_size, 1, 1])
        combin_attr = tf.concat([batch_m, pos_emb], axis=-1)
        attr_out = self.proj_dropout(self.projection(combin_attr), training=training)

        return attr_out#, key

class attention(tf.keras.models.Model):
    def __init__(self, emb_units, proj_dropout_rate, num_heads, num_blocks):
        super(attention, self).__init__()
        self.block = num_blocks
        self.heads = num_heads
        self.emb_units = emb_units

        self.W_k = []
        for _ in range(num_blocks):
            self.W_k.append(tf.keras.layers.Dense(emb_units/num_heads, kernel_initializer='he_uniform', use_bias=False))

        self.W_q = []
        for _ in range(num_blocks):
            self.W_q.append(tf.keras.layers.Dense(emb_units/num_heads, kernel_initializer='he_uniform', use_bias=False))

        self.W_v = []
        for _ in range(num_blocks):
            self.W_v.append(tf.keras.layers.Dense(emb_units/num_heads, kernel_initializer='he_uniform', use_bias=False))
        self.W_z = tf.keras.layers.Dense(emb_units, kernel_initializer='he_uniform', use_bias=False)

        self.feedward = tf.keras.Sequential([])
        self.feedward.add(tf.keras.layers.Dense(emb_units, kernel_initializer='he_uniform', use_bias=True, activation=tf.nn.relu))
        self.feedward.add(tf.keras.layers.Dense(emb_units, kernel_initializer='he_uniform', use_bias=True))
        self.Ly_attention = tf.keras.layers.LayerNormalization(axis=1)
        self.Ly_feedward = tf.keras.layers.LayerNormalization(axis=-1)
        self.projection_dropout = tf.keras.layers.Dropout(proj_dropout_rate)

    def call(self, inputs, training=False):
        """
        :param inputs: meta (i.e. syllable) and note attribute (pitch, duration or rest)
        :type. inputs: tuple
        :shape inputs: ([None, NUM_META_FEATURES], [None, NUM_[.]_TOKENS])

        :param memory: rmc memory
        """
        # inputs, batch_m = inputs

        input_batch = inputs.shape[0]
        input_length = inputs.shape[1]

        z = []
        for i in range(self.heads):

            Q_i = self.W_q[i](inputs)
            K_i = self.W_k[i](inputs)
            V_i = self.W_v[i](inputs)

            self_attention_i = tf.nn.softmax(tf.divide(tf.matmul(Q_i, K_i, transpose_b=True), tf.sqrt(tf.constant([self.emb_units], dtype=tf.float32))))

            z_i = tf.matmul(self_attention_i, V_i)
            z.append(z_i)
        z = tf.concat(z, axis=-1)
        Z = self.W_z(z)

        output_attention = self.Ly_attention(tf.add(inputs, Z))
        out_FFC = self.projection_dropout(self.feedward(output_attention), training=training)

        output_feedward = self.Ly_feedward(tf.add(output_attention, out_FFC))
        # return (output_feedward, batch_m)

        return output_feedward


class Generator_mapping(tf.keras.models.Model):
    def __init__(self, embedding_size, act, Attention_EMB_UNITS, Attention_PROJ_UNITS, Attention_EMB_DROPOUT_RATE, Attention_PROJ_DROPOUT_RATE,
        Attention_MEM_SLOTS, Attention_HEAD_SIZE, Attention_NUM_HEADS, Attention_NUM_BLOCKS, NUM_TOKENS,
        NUM_META_FEATURES):
        super(Generator_mapping, self).__init__()

        self.attention = Generator_attention(Attention_EMB_UNITS[0], Attention_PROJ_UNITS[0], Attention_EMB_DROPOUT_RATE[0],
                                                 Attention_PROJ_DROPOUT_RATE[0], Attention_MEM_SLOTS[0], Attention_HEAD_SIZE[0],
                                                 Attention_NUM_HEADS[0], Attention_NUM_BLOCKS[0], 20, NUM_META_FEATURES)
        self.lyrics_embedding_7 = tf.keras.layers.Dense(128, activation=act)
        self.mutual_information_G = tf.keras.layers.Dense(400)

        self.projection_bn = tf.keras.layers.BatchNormalization(axis=-1)
        self.mi_bn = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, batch_m, training=False):
        lyrics, attr_out = self.attention(batch_m)
        lyrics_6 = lyrics

        lyrics_7 = self.projection_bn(self.lyrics_embedding_7(lyrics_6), training=training)

        mi_fc2 = self.mi_bn(self.mutual_information_G(lyrics_7), training=training)

        return lyrics, mi_fc2, attr_out
