import tensorflow as tf
from utils import *

"""Relational Memory architecture.
An implementation of the architecture described in "Relational Recurrent
Neural Networks", Santoro et al., 2018.
"""
class RelationalMemory(tf.keras.layers.Layer):
    """Relational Memory Core."""

    def __init__(self, mem_slots, head_size, num_heads, num_blocks,
                 forget_bias=1.0, input_bias=0.0, gate_style='unit',
                 attention_mlp_layers=2, key_size=None, name='relational_memory'):
        """Constructs a `RelationalMemory` object.
        Args:
          mem_slots: The total number of memory slots to use.
          head_size: The size of an attention head.
          num_heads: The number of attention heads to use. Defaults to 1.
          num_blocks: Number of times to compute attention per time step. Defaults
            to 1.
          forget_bias: Bias to use for the forget gate, assuming we are using
            some form of gating. Defaults to 1.
          input_bias: Bias to use for the input gate, assuming we are using
            some form of gating. Defaults to 0.
          gate_style: Whether to use per-element gating ('unit'),
            per-memory slot gating ('memory'), or no gating at all (None).
            Defaults to `unit`.
          attention_mlp_layers: Number of layers to use in the post-attention
            MLP. Defaults to 2.
          key_size: Size of vector to use for key & query vectors in the attention
            computation. Defaults to None, in which case we use `head_size`.
          name: Name of the module.
        Raises:
          ValueError: gate_style not one of [None, 'memory', 'unit'].
          ValueError: num_blocks is < 1.
          ValueError: attention_mlp_layers is < 1.
        """
        super(RelationalMemory, self).__init__()
        
        self._mem_slots = mem_slots
        self._head_size = head_size
        self._num_heads = num_heads
        self._mem_size = self._head_size * self._num_heads
        self._name = name

        if num_blocks < 1:
            raise ValueError('num_blocks must be >= 1. Got: {}.'.format(num_blocks))
        self._num_blocks = num_blocks

        self._forget_bias = forget_bias
        self._input_bias = input_bias

        if gate_style not in ['unit', 'memory', None]:
            raise ValueError(
                'gate_style must be one of [\'unit\', \'memory\', None]. Got: '
                '{}.'.format(gate_style))
        self._gate_style = gate_style

        if attention_mlp_layers < 1:
            raise ValueError('attention_mlp_layers must be >= 1. Got: {}.'.format(
                attention_mlp_layers))
        self._attention_mlp_layers = attention_mlp_layers

        self._key_size = key_size if key_size else self._head_size

        if self._gate_style == 'unit' or self._gate_style == 'memory':
            self._num_gates = 2 * self._calculate_gate_size()          
          
            self.fc2 = tf.keras.layers.Dense(
                self._num_gates, use_bias=False, name='gate_in', 
                kernel_initializer=create_linear_initializer(self._mem_size))
          
            self.fc3 = tf.keras.layers.Dense(
                self._num_gates, use_bias=False, name='gate_mem', 
                kernel_initializer=create_linear_initializer(self._mem_size))

        self._qkv_size = 2 * self._key_size + self._head_size
        self._total_size = self._qkv_size * self._num_heads  # Denote as F.
        
        self.linear = [tf.keras.layers.Dense(
            self._total_size, kernel_initializer=create_linear_initializer(
                self._mem_size)) for _ in range(self._num_blocks)]

        self.mlp = [[tf.keras.layers.Dense(
            self._mem_size, kernel_initializer=create_linear_initializer(
                self._mem_size)) for _ in range(self._attention_mlp_layers)] 
                for _ in range(self._num_blocks)]

        self.layer_norm = [[tf.keras.layers.LayerNormalization(trainable=True) 
                            for _ in range(3)] for _ in range(self._num_blocks)]
            
    def build(self, input_shape):
        self.fc1 = tf.keras.layers.Dense(
            self._mem_size, name='input_for_concat', kernel_initializer=create_linear_initializer(input_shape[-1]))

    def initial_state(self, batch_size):
        """Creates the initial memory.
        We should ensure each row of the memory is initialized to be unique,
        so initialize the matrix to be the identity. We then pad or truncate
        as necessary so that init_state is of size
        (batch_size, self._mem_slots, self._mem_size).
        Args:
          batch_size: The size of the batch.
        Returns:
          init_state: A truncated or padded matrix of size
            (batch_size, self._mem_slots, self._mem_size).
        """
        init_state = tf.eye(self._mem_slots, batch_shape=[batch_size])

        # Pad the matrix with zeros.
        if self._mem_size > self._mem_slots:
            difference = self._mem_size - self._mem_slots
            pad = tf.zeros((batch_size, self._mem_slots, difference))
            init_state = tf.concat([init_state, pad], -1)
        # Truncation. Take the first `self._mem_size` components.
        elif self._mem_size < self._mem_slots:
            init_state = init_state[:, :, :self._mem_size]
        return init_state

    def _multihead_attention(self, memory, block_num):
        """Perform multi-head attention from 'Attention is All You Need'.
        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.
        Args:
          memory: Memory tensor to perform attention on, with size [B, N, H*V].
        Returns:
          new_memory: New memory tensor.
        """

        batch_size = memory.get_shape().as_list()[0]  # Denote as B
        memory_flattened = tf.reshape(memory, [-1, self._mem_size])  # [B * N, H * V]
        
        qkv = self.linear[block_num](memory_flattened)
        qkv = tf.reshape(qkv, [batch_size, -1, self._total_size])  # [B, N, F]
        qkv = self.layer_norm[block_num][-1](qkv)  # [B, N, F]
        
        # [B, N, F] -> [B, N, H, F/H]
        qkv_reshape = tf.reshape(qkv, [batch_size, -1, self._num_heads, self._qkv_size])
        
        # [B, N, H, F/H] -> [B, H, N, F/H]
        qkv_transpose = tf.transpose(qkv_reshape, [0, 2, 1, 3])
        
        q, k, v = tf.split(qkv_transpose, [self._key_size, self._key_size, self._head_size], -1)

        q *= self._qkv_size ** -0.5
        dot_product = tf.matmul(q, k, transpose_b=True)  # [B, H, N, N]
        weights = tf.nn.softmax(dot_product)

        output = tf.matmul(weights, v)  # [B, H, N, V]
        
        # [B, H, N, V] -> [B, N, H, V]
        output_transpose = tf.transpose(output, [0, 2, 1, 3])
        
        # [B, N, H, V] -> [B, N, H * V]
        new_memory = tf.reshape(output_transpose, [batch_size, -1, self._mem_size])
        
        return new_memory

    @property
    def state_size(self):
        return tf.TensorShape([self._mem_slots, self._mem_size])

    @property
    def output_size(self):
        return tf.TensorShape(self._mem_slots * self._mem_size)

    def _calculate_gate_size(self):
        """Calculate the gate size from the gate_style.
        Returns:
          The per sample, per head parameter size of each gate.
        """
        if self._gate_style == 'unit':
            return self._mem_size
        elif self._gate_style == 'memory':
            return 1
        else:  # self._gate_style == None
            return 0

    def _attend_over_memory(self, memory):
        """Perform multiheaded attention over `memory`.
        Args:
          memory: Current relational memory.
        Returns:
          The attended-over memory.
        """

        for block_num in range(self._num_blocks):
            attended_memory = self._multihead_attention(memory, block_num)  # [B, N, H * V]

            # Add a skip connection to the multiheaded attention's input.
            memory = self.layer_norm[block_num][0](memory + attended_memory)  # [B, N, H * V]

            # Add a mlp map
            batch_size = memory.get_shape().as_list()[0]

            memory_mlp = tf.reshape(memory, [-1, self._mem_size])  # [B * N, H * V]

            for idx in range(self._attention_mlp_layers):
                memory_mlp = self.mlp[block_num][idx](memory_mlp)
                if idx != self._attention_mlp_layers - 1:
                    memory_mlp = tf.nn.relu(memory_mlp)
              
            memory_mlp = tf.reshape(memory_mlp, [batch_size, -1, self._mem_size])

            # Add a skip connection to the memory_mlp's input.
            memory = self.layer_norm[block_num][1](memory + memory_mlp)  # [B, N, H * V]
            
        return memory

    def _build(self, inputs, memory):
        """Adds relational memory to the TensorFlow graph.
        Args:
          inputs: Tensor input.
          memory: Memory output from the previous time step.
        Returns:
          output: This time step's output.
          next_memory: The next version of memory to use.
        """
        batch_size = memory.get_shape().as_list()[0]
        inputs = tf.reshape(inputs, [batch_size, -1])  # [B, In_size]
        
        inputs = self.fc1(inputs)
        inputs_reshape = tf.expand_dims(inputs, 1)  # [B, 1, V * H]
        
        memory_plus_input = tf.concat([memory, inputs_reshape], axis=1)  # [B, N + 1, V * H]
        next_memory = self._attend_over_memory(memory_plus_input)  # [B, N + 1, V * H]
        
        n = inputs_reshape.get_shape().as_list()[1]
        next_memory = next_memory[:, :-n, :]  # [B, N, V * H]
        
        if self._gate_style == 'unit' or self._gate_style == 'memory':
            batch_size = memory.get_shape().as_list()[0]

            memory = tf.tanh(memory)  # B x N x H * V
            inputs = tf.reshape(inputs_reshape, [batch_size, -1])  # B x In_size

            gate_inputs = self.fc2(inputs) # B x num_gates
            gate_inputs = tf.expand_dims(gate_inputs, axis=1)  # B x 1 x num_gates

            memory_flattened = tf.reshape(memory, [-1, self._mem_size])  # [B * N, H * V]

            gate_memory = self.fc3(memory_flattened) # [B * N, num_gates]
            gate_memory = tf.reshape(gate_memory, [batch_size, self._mem_slots, self._num_gates])  # [B, N, num_gates]

            gates = tf.split(gate_memory + gate_inputs, num_or_size_splits=2, axis=2)
            input_gate, forget_gate = gates  # B x N x num_gates/2, B x N x num_gates/2

            self._input_gate = tf.sigmoid(input_gate + self._input_bias)
            self._forget_gate = tf.sigmoid(forget_gate + self._forget_bias)

            next_memory = self._input_gate * tf.tanh(next_memory)
            next_memory += self._forget_gate * memory

        output = tf.reshape(next_memory, [batch_size, -1])  # [B, V * H]
        
        return output, next_memory


    def call(self, *args):
        """Operator overload for calling.
        This is the entry point when users connect a Module into the Graph. The
        underlying _build method will have been wrapped in a Template by the
        constructor, and we call this template with the provided inputs here.
        Args:
          *args: Arguments for underlying _build method.
          **kwargs: Keyword arguments for underlying _build method.
        Returns:
          The result of the underlying _build method.
        """
        outputs = self._build(*args)

        return outputs

    @property
    def input_gate(self):
        """Returns the input gate Tensor."""
        return self._input_gate

    @property
    def forget_gate(self):
        """Returns the forget gate Tensor."""
        return self._forget_gate
