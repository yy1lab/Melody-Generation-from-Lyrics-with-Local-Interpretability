import tensorflow as tf
import numpy as np
## PreTrain Driver

"""Encapsulates pre-training logic.
"""

class PreTrainDriver():

    def __init__(self, g_model, g_opt, max_grad_norm, num_tokens):
        self.g_model = g_model
        self.g_opt = g_opt
        self.max_grad_norm = max_grad_norm
        self.num_tokens = num_tokens

    def _g_recurrence(self, inp, training=False):
        """
        Generator forward pass with MLE objective.
        """
        batch_m, (batch_p, batch_d, batch_r) = inp

        batch_size = batch_m.shape[0]
        song_length= batch_m.shape[1]

        batch_p_prob = []
        batch_d_prob = []
        batch_r_prob = []

        g_memory = (
            self.g_model.p_subg.initial_state(batch_size),
            self.g_model.d_subg.initial_state(batch_size),
            self.g_model.r_subg.initial_state(batch_size))

        # initial input, equivalent to [START] token in LM Task.
        batch_p_t = tf.random.uniform((batch_size, self.num_tokens[0]), 0.0, 1.0)
        batch_d_t = tf.random.uniform((batch_size, self.num_tokens[1]), 0.0, 1.0)
        batch_r_t = tf.random.uniform((batch_size, self.num_tokens[2]), 0.0, 1.0)

        # generator forward pass
        for t in range(song_length): 
            batch_m_t = batch_m[:, t, :]
            batch_n_t = batch_p_t, batch_d_t, batch_r_t
            (batch_p_logits_t, batch_d_logits_t, batch_r_logits_t), g_memory = self.g_model(
              (batch_m_t, batch_n_t), g_memory, training=training)

            batch_p_prob_t = tf.nn.softmax(batch_p_logits_t, axis=-1) # [None, NUM_P_TOKENS]
            batch_d_prob_t = tf.nn.softmax(batch_d_logits_t, axis=-1) # [None, NUM_D_TOKENS]
            batch_r_prob_t = tf.nn.softmax(batch_r_logits_t, axis=-1) # [None, NUM_R_TOKENS]

            batch_p_prob.append(tf.expand_dims(batch_p_prob_t, 1)) # [None, 1, NUM_P_TOKENS]
            batch_d_prob.append(tf.expand_dims(batch_d_prob_t, 1)) # [None, 1, NUM_D_TOKENS]
            batch_r_prob.append(tf.expand_dims(batch_r_prob_t, 1)) # [None, 1, NUM_R_TOKENS]

            # teacher forcing
            batch_p_t = batch_p[:, t, :] # [None, NUM_P_TOKENS]
            batch_d_t = batch_d[:, t, :] # [None, NUM_D_TOKENS]
            batch_r_t = batch_r[:, t, :] # [None, NUM_R_TOKENS]

        batch_p_prob = tf.concat(batch_p_prob, axis=1) # [None, song_length, NUM_P_TOKENS]
        batch_d_prob = tf.concat(batch_d_prob, axis=1) # [None, song_length, NUM_D_TOKENS]
        batch_r_prob = tf.concat(batch_r_prob, axis=1) # [None, song_length, NUM_R_TOKENS]

        return batch_p_prob, batch_d_prob, batch_r_prob

    def _loss_fn(self, true, pred):
        """Categorical CrossEntropy loss.
        :shape true: [None, SONG_LENGTH, NUM_P/D/R_TOKENS]
        :shape pred: [None, SONG_LENGTH, NUM_P/D/R_TOKENS]
        """
        batch_size = true.shape[0]
        song_length= true.shape[1]

        true = tf.reshape(true, (batch_size * song_length, -1))
        pred = tf.reshape(pred, (batch_size * song_length, -1))
        pred = tf.math.log(tf.clip_by_value(pred, 1e-20, 1.0))
        loss= -tf.reduce_sum(true * pred) / (batch_size * song_length)

        return loss

    def _compute_loss(self, inp, out):
        batch_m, (batch_p, batch_d, batch_r)     = inp
        batch_p_prob, batch_d_prob, batch_r_prob = out

        # loss computation
        p_loss = self._loss_fn(batch_p, batch_p_prob)      
        d_loss = self._loss_fn(batch_d, batch_d_prob)      
        r_loss = self._loss_fn(batch_r, batch_r_prob)      

        return p_loss, d_loss, r_loss

    def _step(self, inp, training=False):
        # forward pass
        g_out = self._g_recurrence(inp, training=training) 
        # loss computation
        g_loss = self._compute_loss(inp, g_out) 

        return g_loss
    
    @tf.function
    def train_step(self, inp):
        """Pretrain generator.
        """
        with tf.GradientTape(persistent=True) as g_tape:
            p_loss, d_loss, r_loss = self._step(inp, training=True)

        p_vars = self.g_model.p_subg.trainable_variables
        d_vars = self.g_model.d_subg.trainable_variables
        r_vars = self.g_model.r_subg.trainable_variables

        p_grads, _ = tf.clip_by_global_norm(g_tape.gradient(p_loss, p_vars), self.max_grad_norm)
        d_grads, _ = tf.clip_by_global_norm(g_tape.gradient(d_loss, d_vars), self.max_grad_norm)
        r_grads, _ = tf.clip_by_global_norm(g_tape.gradient(r_loss, r_vars), self.max_grad_norm)

        g_vars  = p_vars  + d_vars  + r_vars 
        g_grads = p_grads + d_grads + r_grads 

        self.g_opt.apply_gradients(zip(g_grads, g_vars))

        return p_loss, d_loss, r_loss

    @tf.function
    def test_step(self, inp):
        """
        Compute pretrain loss a.k.a negative log likelihood or nll.
        Teacher forcing is done during loss computation.
        """
        p_loss, d_loss, r_loss = self._step(inp, training=False)
        return p_loss, d_loss, r_loss

    @tf.function
    def generate(self, inp):
        """
        Perform generator forward pass.
        """
        g_p_out, g_d_out, g_r_out = self._g_recurrence(inp, training=False)
        g_out = (g_p_out, g_d_out, g_r_out)

        return g_out

"""## Adversarial Driver"""

"""Encapsulates adversarial training and inference logic.
"""
class AdversarialDriver():

    def __init__(self, g_model, g_map_model, d_model, d_map_model, g_opt, d_opt, temp_max, steps_per_epoch,
               adv_train_epochs, max_grad_norm, num_tokens):
        """
        :param temp_max: maximum temperature a.k.a beta_max
        :param n_adv_steps : total number of adversarial steps
        """
        self.g_model = g_model
        self.g_map_model = g_map_model
        self.d_model = d_model
        self.d_map_model = d_map_model

        self.g_opt = g_opt
        self.d_opt = d_opt

        self.temp = tf.Variable(1., trainable=False)
        self.temp_max = temp_max

        self.steps_per_epoch = steps_per_epoch
        self.n_adv_steps = adv_train_epochs * steps_per_epoch

        self.max_grad_norm = max_grad_norm
        self.num_tokens = num_tokens

    def add_gumbel(self, o, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        u = tf.random.uniform(tf.shape(o), minval=0, maxval=1, dtype=tf.float32)
        g = -tf.math.log(-tf.math.log(u + eps) + eps)
        gumbel = tf.add(o, g)
        return gumbel

    def _g_recurrence(self, batch_m, training=False):
        """Generator forward pass.
        """
        lyrics, mi_fc2, attr_out = self.g_map_model(batch_m)

        batch_p_out = []
        batch_d_out = []
        batch_r_out = []

        batch_p_out_oha = []
        batch_d_out_oha = []
        batch_r_out_oha = []

        batch_size = batch_m.shape[0]
        song_length= batch_m.shape[1]

        g_memory = (
            self.g_model.p_subg.initial_state(batch_size),
            self.g_model.d_subg.initial_state(batch_size),
            self.g_model.r_subg.initial_state(batch_size))

        # initial input, equivalent to [START] token in LM Task
        batch_p_t = tf.random.uniform((batch_size, self.num_tokens[0]), 0.0, 1.0)
        batch_d_t = tf.random.uniform((batch_size, self.num_tokens[1]), 0.0, 1.0)
        batch_r_t = tf.random.uniform((batch_size, self.num_tokens[2]), 0.0, 1.0)

        # generator forward pass
        for t in range(song_length): 
            batch_m_t = lyrics[:, t, :]
            batch_n_t = batch_p_t, batch_d_t, batch_r_t
            (batch_p_logits_t, batch_d_logits_t, batch_r_logits_t), g_memory = self.g_model(
              (batch_m_t, batch_n_t), g_memory, training=training) 

            batch_p_gumbel_t = self.add_gumbel(batch_p_logits_t) # [None, NUM_P_TOKENS]
            batch_d_gumbel_t = self.add_gumbel(batch_d_logits_t) # [None, NUM_D_TOKENS]
            batch_r_gumbel_t = self.add_gumbel(batch_r_logits_t) # [None, NUM_R_TOKENS]

            batch_p_out_t = tf.stop_gradient(tf.argmax(batch_p_gumbel_t, axis=-1)) # [None]
            batch_d_out_t = tf.stop_gradient(tf.argmax(batch_d_gumbel_t, axis=-1)) # [None]
            batch_r_out_t = tf.stop_gradient(tf.argmax(batch_r_gumbel_t, axis=-1)) # [None]

            batch_p_out_oha_t = tf.nn.softmax(tf.multiply(batch_p_gumbel_t, self.temp)) # [None, NUM_P_TOKENS]
            batch_d_out_oha_t = tf.nn.softmax(tf.multiply(batch_d_gumbel_t, self.temp)) # [None, NUM_D_TOKENS]
            batch_r_out_oha_t = tf.nn.softmax(tf.multiply(batch_r_gumbel_t, self.temp)) # [None, NUM_R_TOKENS]

            batch_p_out.append(tf.expand_dims(batch_p_out_t, 1))  # [None, 1]
            batch_d_out.append(tf.expand_dims(batch_d_out_t, 1))  # [None, 1]
            batch_r_out.append(tf.expand_dims(batch_r_out_t, 1))  # [None, 1]

            batch_p_out_oha.append(tf.expand_dims(batch_p_out_oha_t, 1)) # [None, 1, NUM_P_TOKENS]
            batch_d_out_oha.append(tf.expand_dims(batch_d_out_oha_t, 1)) # [None, 1, NUM_D_TOKENS]
            batch_r_out_oha.append(tf.expand_dims(batch_r_out_oha_t, 1)) # [None, 1, NUM_R_TOKENS]

            # No teacher forcing so avoid so-called exposure bias
            batch_p_t = batch_p_out_oha_t
            batch_d_t = batch_d_out_oha_t
            batch_r_t = batch_r_out_oha_t
  
        batch_p_out = tf.concat(batch_p_out, axis=1) # [None, song_length]
        batch_d_out = tf.concat(batch_d_out, axis=1) # [None, song_length]
        batch_r_out = tf.concat(batch_r_out, axis=1) # [None, song_length]

        batch_p_out_oha = tf.concat(batch_p_out_oha, axis=1) # [None, song_length, NUM_P_TOKENS]
        batch_d_out_oha = tf.concat(batch_d_out_oha, axis=1) # [None, song_length, NUM_D_TOKENS]
        batch_r_out_oha = tf.concat(batch_r_out_oha, axis=1) # [None, song_length, NUM_R_TOKENS]
    
        return ((batch_p_out, batch_d_out, batch_r_out),
                (batch_p_out_oha, batch_d_out_oha, batch_r_out_oha), tf.keras.layers.Flatten()(mi_fc2),
                lyrics, attr_out)

    def _d_recurrence(self, batch_m, batch_n, training=False):
        """Discriminator forward pass.
        """
        d_out = []

        batch_size = batch_m.shape[0]
        song_length= batch_m.shape[1]

        d_memory = self.d_model.initial_state(batch_size)

        batch_p, batch_d, batch_r = batch_n

        for t in range(song_length):
            batch_m_t = batch_m[:, t, :]
            batch_n_t = batch_p[:, t, :], batch_d[:, t, :], batch_r[:, t, :]
            d_out_t, d_memory = self.d_model((batch_m_t, batch_n_t), d_memory, training=training)
            d_out.append(tf.expand_dims(d_out_t, 1))

        d_out = tf.concat(d_out, axis=1) # [None, song_length, 1]
        mi_fc2 = self.d_map_model(d_out)
        d_out = tf.reduce_mean(d_out, axis=[1, 2]) # [None]

        return d_out, tf.keras.layers.Flatten()(mi_fc2)

    def _loss_fn(self, real_logits, fake_logits):
        """Relativistic GAN loss 
        """
        g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits-real_logits, labels=tf.ones_like(fake_logits))

        d_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_logits-fake_logits, labels=tf.ones_like(real_logits))

        return tf.reduce_mean(g_loss), tf.reduce_mean(d_loss)
    
    def _compute_loss(self, inp, out, training=False):
        """Relativistic Standard GAN Loss.
        :param inp: syllable meta data & note 
        :param out: output from the generator
        """
        batch_m, (batch_p, batch_d, batch_r)    = inp
        (batch_p_oha, batch_d_oha, batch_r_oha) = out

        # discriminator forward pass with generated melodies
        batch_fake_logits, batch_fake_D_mi = self._d_recurrence(
            batch_m, (batch_p_oha, batch_d_oha, batch_r_oha), training=training)

        # discriminator forward pass with real melodies
        batch_real_logits, batch_real_D_mi = self._d_recurrence(
            batch_m, (batch_p, batch_d, batch_r), training=training)

        # loss computation
        g_loss, d_loss = self._loss_fn(batch_real_logits, batch_fake_logits)

        return g_loss, d_loss, batch_fake_D_mi

    def _step(self, inp, training=False):
        batch_m, _ = inp

        g_out, g_out_oha, batch_G_mi, _, _ = self._g_recurrence(batch_m, training=training)
        G_mi = -tf.math.reduce_sum(self._compute_mi_loss(batch_G_mi, tf.compat.v1.layers.flatten(batch_m), 1e-6))

        g_loss, d_loss, batch_fake_D_mi = self._compute_loss(inp, g_out_oha, training=training)
        D_mi = -tf.math.reduce_sum(self._compute_mi_loss(batch_fake_D_mi, tf.compat.v1.layers.flatten(batch_m), 1e-6))

        G_loss = g_loss + G_mi + D_mi
        D_loss = d_loss + D_mi

        return G_loss, D_loss, g_out, G_mi, D_mi
    
    @tf.function
    def train_step(self, inp):
        """Adversarial Step
        Each adversarial step is composed of 1 generator step + 1 discriminator step
        """
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            g_loss, d_loss, g_out, G_mi, D_mi = self._step(inp, training=True)

        g_vars = self.g_model.trainable_variables
        d_vars = self.d_model.trainable_variables


        g_map_vars = self.g_map_model.trainable_variables
        d_map_vars = self.d_map_model.trainable_variables


        g_vars.extend(g_map_vars)
        d_vars.extend(d_map_vars)

        g_grads, _ = tf.clip_by_global_norm(g_tape.gradient(g_loss, g_vars), self.max_grad_norm)
        d_grads, _ = tf.clip_by_global_norm(d_tape.gradient(d_loss, d_vars), self.max_grad_norm)

        self.g_opt.apply_gradients(zip(g_grads, g_vars))
        self.d_opt.apply_gradients(zip(d_grads, d_vars))

        return g_loss, d_loss, g_out, G_mi, D_mi
    
    @tf.function
    def generate(self, batch_m):
        """
        Perform generator forward pass.
        """
        g_out, g_out_oha, _, lyrics, attr_out = self._g_recurrence(batch_m, training=False)
        return g_out, g_out_oha, lyrics, attr_out

    @tf.function
    def test_step(self, inp):
        """
        Compute generator & discriminator loss, generate pitch, duration & rest 
        conditioned on syllable using test/validation data.
        """
        g_loss, d_loss, g_out, G_mi, D_mi = self._step(inp, training=False)
        return g_loss, d_loss, g_out, G_mi, D_mi

    def reset_temp(self):
        tf.keras.backend.set_value(self.temp, 1.0)

    def update_temp(self, epoch, step):
        """
        Update temperautre for the current step in current epoch.
        """
        step = (epoch * self.steps_per_epoch) + step
        temp = self.temp_max ** (step / self.n_adv_steps)
        tf.keras.backend.set_value(self.temp, temp)

    # Mutual information loss function
    def _compute_mi_loss(self, mi_fc2, labels, mi_lambda=1.0, fix_std=True):
        label_size = labels.get_shape()[1]
        mean_contig = mi_fc2[:, 0:label_size]
        # mean_contig = mi_fc2
        std_contig = tf.ones_like(mean_contig)
        epsilon = (labels - mean_contig) / (std_contig + 1e-6)
        mi_l1 = tf.math.reduce_sum(- 0.5 * np.log(2 * np.pi) - tf.math.log(std_contig + 1e-6) - 0.5 * tf.math.square(epsilon),
                              axis=1)
        # mi_l1 = tf.math.reduce_sum(- 0.5 * tf.math.square(epsilon), axis=1)
        mi_loss = mi_lambda * mi_l1
        return mi_loss
