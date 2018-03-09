class Layer:
    """
    A building block. Each layer is capable of performing two things:

    - Process input to get output:           output = layer.forward(input)

    - Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)

    Some layers also have learnable parameters which they update during layer.backward.
    """

    def __init__(self):
        """
        Here you can initialize layer parameters (if any) and auxiliary stuff.
        """

        raise NotImplementedError("Not implemented in interface")

    def forward(self, input):
        """
        Takes input data of shape [batch, ...], returns output data [batch, ...]
        """

        raise NotImplementedError("Not implemented in interface")

    def backward(self, input, grad_output):
        """
        Performs a backpropagation step through the layer, with respect to the given input.

        To compute loss gradients w.r.t input, you need to apply chain rule (backprop):

        d loss / d x  = (d loss / d layer) * (d layer / d x)

        Luckily, you already receive d loss / d layer as input, so you only need to multiply it by d layer / d x.

        If your layer has parameters (e.g. dense layer), you also need to update them here using d loss / d layer
        """

        raise NotImplementedError("Not implemented in interface")


class ReLU(Layer):
    def __init__(self):
        """
        ReLU layer simply applies elementwise rectified linear unit to all inputs
        This layer does not have any parameters.
        """

        raise NotImplementedError("Implement me plz ;(")

    def forward(self, input):
        """
        Perform ReLU transformation
        input shape: [batch, input_units]
        output shape: [batch, input_units]
        """

        raise NotImplementedError("Implement me plz ;(")

    def backward(self, input, grad_output):
        """
        Compute gradient of loss w.r.t. ReLU input
        """

        raise NotImplementedError("Implement me plz ;(")


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        A dense layer is a layer which performs a learned affine transformation:
        f(x) = Wx + b

        W: matrix of shape [num_inputs, num_outputs]
        b: vector of shape [num_outputs]
        """

        self.learning_rate = learning_rate

        # initialize weights with small random numbers from normal distribution

        # self.weights =
        # self.biases =
        raise NotImplementedError("Implement me plz ;(")

    def forward(self, input):
        """
        Perform an affine transformation:
        f(x) = <W*x> + b

        input shape: [batch, input_units]
        output shape: [batch, output units]
        """

        raise NotImplementedError("Implement me plz ;(")

    def backward(self, input, grad_output):
        """
        Computes d f / d x = d f / d dense * d dense / d x,
        where d dense/ d x = weights transposed, and performs
        one step of gradient descent on W and b.

        input shape: [batch, input_units]
        grad_output: [batch, output units]

        Returns: grad_input, gradient of output w.r.t input
        """

        raise NotImplementedError("Implement me plz ;(")


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, learning_rate=0.1):
        """
        A convolutional layer with out_channels kernels of kernel_size.

        in_channels — number of input channels
        out_channels — number of convolutional filters
        kernel_size — tuple of two numbers: k_1 and k_2

        Initialize required weights.
        """

        raise NotImplementedError("Implement me plz ;(")

    def forward(self, input):
        """
        Perform convolutional transformation:

        input shape: [batch, in_channels, h, w]
        output shape: [batch, out_channels, h_out, w_out]
        """

        raise NotImplementedError("Implement me plz ;(")

    def backward(self, input, grad_output):
        """
        Compute gradients w.r.t input and weights and update weights
        """

        raise NotImplementedError("Implement me plz ;(")


class Maxpool2d(Layer):
    def __init__(self, kernel_size):
        """
        A maxpooling layer with kernel of kernel_size.
        This layer donwsamples [kernel_size, kernel_size] to
        1 number which represents maximum.

        Stride description is identical to the convolution
        layer. But default value we use is kernel_size to
        reduce dim by kernel_size times.

        This layer does not have any learnable parameters.
        """

        self.stride = kernel_size
        self.kernel_size = kernel_size

    def forward(self, input):
        """
        Perform maxpooling transformation:

        input shape: [batch, in_channels, h, w]
        output shape: [batch, out_channels, h_out, w_out]
        """

        raise NotImplementedError("Implement me plz ;(")

    def backward(self, input, grad_output):
        """
        This layer just propagates gradients into
        maximum values, more preciesly grad_output
        into corresponting argmaxes and 0 otherwise.
        """

        raise NotImplementedError("Implement me plz ;(")


class Flatten(Layer):
    def __init__(self):
        """
        This layer does not have any parameters
        """

        raise NotImplementedError("Implement me plz ;(")

    def forward(self, input):
        """
        input shape: [batch_size, channels, feature_nums_h, feature_nums_w]
        output shape: [batch_size, channels * feature_nums_h * feature_nums_w]
        """

        raise NotImplementedError("Implement me plz ;(")

    def backward(self, input, grad_output):
        """
        Compute gradient of loss w.r.t. Flatten input
        """

        raise NotImplementedError("Implement me plz ;(")


def softmax_crossentropy_with_logits(logits, y_true):
    """
    Compute crossentropy from logits and ids of correct answers
    logits shape: [batch_size, num_classes]
    reference_answers: [batch_size]
    output is a number
    """

    raise NotImplementedError("Implement me plz ;(")


def grad_softmax_crossentropy_with_logits(logits, y_true):
    """
    Compute crossentropy gradient from logits and ids of correct answers
    logits shape: [batch_size, num_classes]
    reference_answers: [batch_size]
    """

    raise NotImplementedError("Implement me plz ;(")