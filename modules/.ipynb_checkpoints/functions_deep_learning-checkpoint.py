# import packages

# general tools
import numpy as np
from scipy import stats

# tensorflow
import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, Input, Dense, Add, InputSpec, Dropout, BatchNormalization
from tensorflow.keras.models import Model

# Spektral
from spektral.layers import GCNConv, GlobalSumPool






# auxiliary functions for plotting and data processing

def extract_learning_curves(classifier_model_training_histories):

    n_models = len(classifier_model_training_histories)
    n_epochs = len(classifier_model_training_histories[0].history["loss"])
    metric_names = list(classifier_model_training_histories[0].history.keys())

    #assert metric_names[0] == "loss"
    #assert metric_names[1][0:3] == "auc"
    #assert metric_names[2] == "binary_accuracy"
    #assert metric_names[3] == "val_loss"
    #assert metric_names[4][0:7] == "val_auc"
    #assert metric_names[5] == "val_binary_accuracy"


    train_loss_array = np.zeros((n_models, n_epochs))
    train_roc_auc_array = np.zeros((n_models, n_epochs))
    train_acc_array = np.zeros((n_models, n_epochs))
    test_loss_array = np.zeros((n_models, n_epochs))
    test_roc_auc_array = np.zeros((n_models, n_epochs))
    test_acc_array = np.zeros((n_models, n_epochs))

    for (k, history) in enumerate(classifier_model_training_histories):

        train_loss_array[k,:] = history.history[metric_names[0]]
        train_roc_auc_array[k,:] = history.history[metric_names[1]]
        train_acc_array[k,:] = history.history[metric_names[2]]

        test_loss_array[k,:] = history.history[metric_names[3]]
        test_roc_auc_array[k,:] = history.history[metric_names[4]]
        test_acc_array[k,:] = history.history[metric_names[5]]

    train_loss_avg = np.mean(train_loss_array, axis = 0)
    train_roc_auc_avg = np.mean(train_roc_auc_array, axis = 0)
    train_acc_avg = np.mean(train_acc_array, axis = 0)
    test_loss_avg = np.mean(test_loss_array, axis = 0)
    test_roc_auc_avg = np.mean(test_roc_auc_array, axis = 0)
    test_acc_avg = np.mean(test_acc_array, axis = 0)

    return (train_loss_avg, train_roc_auc_avg, train_acc_avg, test_loss_avg, test_roc_auc_avg, test_acc_avg)


def extract_learning_curves_from_dual_task_model(classifier_model_training_histories):

    n_models = len(classifier_model_training_histories)
    n_epochs = len(classifier_model_training_histories[0].history["loss"])
    metric_names = list(classifier_model_training_histories[0].history.keys())

    #assert metric_names[0] == "loss"
    #assert metric_names[1] == "ac_mlp_loss"
    #assert metric_names[2] == "pd_mlp_loss"
    #assert metric_names[3][0:10] == "ac_mlp_auc"
    #assert metric_names[4] == "ac_mlp_binary_accuracy"
    #assert metric_names[5][0:11] == "pd_mlp_auc"
    #assert metric_names[6] == "pd_mlp_binary_accuracy"

    #assert metric_names[7] == "val_loss"
    #assert metric_names[8] == "val_ac_mlp_loss"
    #assert metric_names[9] == "val_pd_mlp_loss"
    #assert metric_names[10][0:14] == "val_ac_mlp_auc"
    #assert metric_names[11] == "val_ac_mlp_binary_accuracy"
    #assert metric_names[12][0:15] == "val_pd_mlp_auc"
    #assert metric_names[13] == "val_pd_mlp_binary_accuracy"

    train_loss_array = np.zeros((n_models, n_epochs))
    train_ac_mlp_loss_array = np.zeros((n_models, n_epochs))
    train_pd_mlp_loss_array = np.zeros((n_models, n_epochs))
    train_ac_mlp_auc_array = np.zeros((n_models, n_epochs))
    train_ac_mlp_binary_accuracy_array = np.zeros((n_models, n_epochs))
    train_pd_mlp_auc_array = np.zeros((n_models, n_epochs))
    train_pd_mlp_binary_accuracy_array = np.zeros((n_models, n_epochs))

    test_loss_array = np.zeros((n_models, n_epochs))
    test_ac_mlp_loss_array = np.zeros((n_models, n_epochs))
    test_pd_mlp_loss_array = np.zeros((n_models, n_epochs))
    test_ac_mlp_auc_array = np.zeros((n_models, n_epochs))
    test_ac_mlp_binary_accuracy_array = np.zeros((n_models, n_epochs))
    test_pd_mlp_auc_array = np.zeros((n_models, n_epochs))
    test_pd_mlp_binary_accuracy_array = np.zeros((n_models, n_epochs))

    for (k, history) in enumerate(classifier_model_training_histories):

        train_loss_array[k,:] = history.history[metric_names[0]]
        train_ac_mlp_loss_array[k,:] = history.history[metric_names[1]]
        train_pd_mlp_loss_array[k,:] = history.history[metric_names[2]]
        train_ac_mlp_auc_array[k,:] = history.history[metric_names[3]]
        train_ac_mlp_binary_accuracy_array[k,:] = history.history[metric_names[4]]
        train_pd_mlp_auc_array[k,:] = history.history[metric_names[5]]
        train_pd_mlp_binary_accuracy_array[k,:] = history.history[metric_names[6]]

        test_loss_array[k,:] = history.history[metric_names[7]]
        test_ac_mlp_loss_array[k,:] = history.history[metric_names[8]]
        test_pd_mlp_loss_array[k,:] = history.history[metric_names[9]]
        test_ac_mlp_auc_array[k,:] = history.history[metric_names[10]]
        test_ac_mlp_binary_accuracy_array[k,:] = history.history[metric_names[11]]
        test_pd_mlp_auc_array[k,:] = history.history[metric_names[12]]
        test_pd_mlp_binary_accuracy_array[k,:] = history.history[metric_names[13]]

    train_loss_avg = np.mean(train_loss_array, axis = 0)
    train_ac_mlp_loss_avg = np.mean(train_ac_mlp_loss_array, axis = 0)
    train_pd_mlp_loss_avg = np.mean(train_pd_mlp_loss_array, axis = 0)
    train_ac_mlp_auc_avg = np.mean(train_ac_mlp_auc_array, axis = 0)
    train_ac_mlp_binary_accuracy_avg = np.mean(train_ac_mlp_binary_accuracy_array, axis = 0)
    train_pd_mlp_auc_avg = np.mean(train_pd_mlp_auc_array, axis = 0)
    train_pd_mlp_binary_accuracy_avg = np.mean(train_pd_mlp_binary_accuracy_array, axis = 0)

    test_loss_avg = np.mean(test_loss_array, axis = 0)
    test_ac_mlp_loss_avg = np.mean(test_ac_mlp_loss_array, axis = 0)
    test_pd_mlp_loss_avg = np.mean(test_pd_mlp_loss_array, axis = 0)
    test_ac_mlp_auc_avg = np.mean(test_ac_mlp_auc_array, axis = 0)
    test_ac_mlp_binary_accuracy_avg = np.mean(test_ac_mlp_binary_accuracy_array, axis = 0)
    test_pd_mlp_auc_avg = np.mean(test_pd_mlp_auc_array, axis = 0)
    test_pd_mlp_binary_accuracy_avg = np.mean(test_pd_mlp_binary_accuracy_array, axis = 0)

    return (train_loss_avg,
            train_ac_mlp_loss_avg,
            train_pd_mlp_loss_avg,
            train_ac_mlp_auc_avg,
            train_ac_mlp_binary_accuracy_avg,
            train_pd_mlp_auc_avg,
            train_pd_mlp_binary_accuracy_avg,
            test_loss_avg,
            test_ac_mlp_loss_avg,
            test_pd_mlp_loss_avg,
            test_ac_mlp_auc_avg,
            test_ac_mlp_binary_accuracy_avg,
            test_pd_mlp_auc_avg,
            test_pd_mlp_binary_accuracy_avg)



## deep learning functions for model creation

# layer for trainable rational activation functions

class RationalLayer(Layer):
    """ This class was taken from Nicolas Boulle at
    https://github.com/NBoulle/RationalNets/blob/master/src/RationalLayer.py

    Rational Activation function.
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are learned array with the same shape as x.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha_initializer: initializer function for the weights of the numerator P.
        beta_initializer: initializer function for the weights of the denominator Q.
        alpha_regularizer: regularizer for the weights of the numerator P.
        beta_regularizer: regularizer for the weights of the denominator Q.
        alpha_constraint: constraint for the weights of the numerator P.
        beta_constraint: constraint for the weights of the denominator Q.
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """

    def __init__(self, alpha_initializer=[1.1915, 1.5957, 0.5, 0.0218], beta_initializer=[2.383, 0.0, 1.0],
                 alpha_regularizer=None, beta_regularizer=None, alpha_constraint=None, beta_constraint=None,
                 shared_axes=None, **kwargs):
        super(RationalLayer, self).__init__(**kwargs)
        self.supports_masking = True

        # degree of rationals
        self.degreeP = len(alpha_initializer) - 1
        self.degreeQ = len(beta_initializer) - 1

        # initializers for P
        self.alpha_initializer = [initializers.Constant(value=alpha_initializer[i]) for i in range(len(alpha_initializer))]
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)

        # initializers for Q
        self.beta_initializer = [initializers.Constant(value=beta_initializer[i]) for i in range(len(beta_initializer))]
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)

        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1

        self.coeffsP = []
        for i in range(self.degreeP+1):
            # add weight
            alpha_i = self.add_weight(shape=param_shape,
                                     name='alpha_%s'%i,
                                     initializer=self.alpha_initializer[i],
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
            self.coeffsP.append(alpha_i)

        # create coefficients of Q
        self.coeffsQ = []
        for i in range(self.degreeQ+1):
            # add weight
            beta_i = self.add_weight(shape=param_shape,
                                     name='beta_%s'%i,
                                     initializer=self.beta_initializer[i],
                                     regularizer=self.beta_regularizer,
                                     constraint=self.beta_constraint)
            self.coeffsQ.append(beta_i)

        # set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
                    self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
                    self.built = True

    def call(self, inputs, mask=None):
        # evaluation of P
        outP = tf.math.polyval(self.coeffsP, inputs)
        # evaluation of Q
        outQ = tf.math.polyval(self.coeffsQ, inputs)
        # compute P/Q
        out = tf.math.divide(outP, outQ)
        return out

    def get_config(self):
        config = {
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(RationalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


# function to create an MLP model

def create_mlp_model(architecture = (2**10, 100, 100, 1),
                     hidden_activation = tf.keras.activations.relu,
                     output_activation = tf.keras.activations.sigmoid,
                     use_bias = True,
                     use_batch_norm_input_hidden_lasthidden = (False, False, False),
                     dropout_rates_input_hidden = (0.0, 0.0),
                     hidden_rational_layers = False,
                     rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                     rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                     rational_parameters_shared_axes = [1],
                     model_name = None):

    """
    Function to create an MLP model, optionally with trainable rational activation function layers.
    rational_parameters_shared_axes = [1] means that weights are shared across each rational layer
    """

    # define variables
    n_hidden = len(architecture)-2
    dropout_rate_input = dropout_rates_input_hidden[0]
    dropout_rate_hidden = dropout_rates_input_hidden[1]
    use_batch_norm_input = use_batch_norm_input_hidden_lasthidden[0]
    use_batch_norm_hidden = use_batch_norm_input_hidden_lasthidden[1]
    use_batch_norm_lasthidden = use_batch_norm_input_hidden_lasthidden[2]

    # define first input layer
    input_layer = Input(shape = architecture[0], name = "input")
    hidden = input_layer

    # if wanted, add batch normalisation layer right after input layer
    if use_batch_norm_input == True:
        hidden = BatchNormalization()(hidden)

    # add dropout layer right after input
    hidden = Dropout(rate = dropout_rate_input, name = "dropout_input")(hidden)

    # define hidden layers
    for h in range(1, n_hidden + 1):

        hidden = Dense(units = architecture[h],
                       activation = hidden_activation,
                       use_bias = use_bias,
                       name = "hidden_" + str(h))(hidden)

        # if wanted, define additional hidden layers with trainable rational activation functions
        if hidden_rational_layers == True:

            hidden = RationalLayer(alpha_initializer = rational_parameters_alpha_initializer,
                                   beta_initializer = rational_parameters_beta_initializer,
                                   shared_axes = rational_parameters_shared_axes,
                                   name = "rational_hidden_" + str(h))(hidden)

        # if wanted, add batch normalisation layer after all hidden layers but the last hidden layer
        if use_batch_norm_hidden == True and h != n_hidden:
            hidden = BatchNormalization()(hidden)

        # if wanted, add batch normalisation also after the last hidden layer
        if use_batch_norm_lasthidden == True and h == n_hidden:
            hidden = BatchNormalization()(hidden)

        # add dropout layer
        hidden = Dropout(rate = dropout_rate_hidden, name = "dropout_hidden" + str(h))(hidden)


    # define final output layer

    if n_hidden >= 0:

        output_layer = Dense(architecture[n_hidden + 1],
                             activation = output_activation,
                             use_bias = use_bias,
                             name = 'output')(hidden)
    else:

        output_layer = hidden

    # build model
    mlp_model = Model(inputs = [input_layer], outputs = [output_layer], name = model_name)

    return mlp_model


# create graph neural network models

def create_gcn_model(gcn_n_node_features = 25,
                     gcn_n_hidden = 2,
                     gcn_channels = 25,
                     gcn_activation = tf.keras.activations.relu,
                     gcn_use_bias = True,
                     gcn_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                     gcn_dropout_rates_input_hidden = (0, 0)):

    # define parameters
    gcn_dropout_rates_input = gcn_dropout_rates_input_hidden[0]
    gcn_dropout_rates_hidden = gcn_dropout_rates_input_hidden[1]
    gcn_use_batch_norm_input = gcn_use_batch_norm_input_hidden_lasthidden[0]
    gcn_use_batch_norm_hidden = gcn_use_batch_norm_input_hidden_lasthidden[1]
    gcn_use_batch_norm_lasthidden = gcn_use_batch_norm_input_hidden_lasthidden[2]

    # define input tensors
    X_mol_graphs = Input(shape=(None, gcn_n_node_features))
    A_mod_mol_graphs = Input(shape=(None, None))

    # define GNN convolutional layers
    X = X_mol_graphs
    A = A_mod_mol_graphs

    # if wanted, batch norm layer right after input
    if gcn_use_batch_norm_input == True:
        X = BatchNormalization()(X)

    # add dropout layer right after input
    X = Dropout(gcn_dropout_rates_input)(X)

    # hidden layers
    for h in range(1, gcn_n_hidden + 1):
        X = GCNConv(channels = gcn_channels, activation = gcn_activation, use_bias = gcn_use_bias)([X, A])

        if gcn_use_batch_norm_hidden == True and h != gcn_n_hidden:
            X = BatchNormalization()(X)

        if gcn_use_batch_norm_lasthidden == True and h == gcn_n_hidden:
            X = BatchNormalization()(X)

        X = Dropout(gcn_dropout_rates_hidden)(X)

    # define global pooling layer to reduce graph to a single vector via node features
    X = GlobalSumPool()(X)

    # define final model
    model = Model(inputs = [X_mol_graphs, A_mod_mol_graphs], outputs = X, name = "gcn_model")

    return model


def create_gcn_mlp_model(gcn_n_node_features = 25,
                         gcn_n_hidden = 2,
                         gcn_channels = 25,
                         gcn_activation = tf.keras.activations.relu,
                         gcn_use_bias = True,
                         gcn_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                         gcn_dropout_rates_input_hidden = (0,0),
                         mlp_architecture = (25, 25, 1),
                         mlp_hidden_activation = tf.keras.activations.relu,
                         mlp_output_activation = tf.keras.activations.sigmoid,
                         mlp_use_bias = True,
                         mlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                         mlp_dropout_rates_input_hidden = (0, 0),
                         mlp_hidden_rational_layers = False,
                         mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                         mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                         mlp_rational_parameters_shared_axes = [1]):

    # define input tensors
    X_mol_graphs = Input(shape=(None, gcn_n_node_features))
    A_mod_mol_graphs = Input(shape=(None, None))

    # define and apply GCN model
    gcn_model = create_gcn_model(gcn_n_node_features = gcn_n_node_features,
                                 gcn_n_hidden = gcn_n_hidden,
                                 gcn_channels = gcn_channels,
                                 gcn_activation = gcn_activation,
                                 gcn_use_bias = gcn_use_bias,
                                 gcn_use_batch_norm_input_hidden_lasthidden = gcn_use_batch_norm_input_hidden_lasthidden,
                                 gcn_dropout_rates_input_hidden = gcn_dropout_rates_input_hidden)

    X = gcn_model([X_mol_graphs, A_mod_mol_graphs])

    # define and apply MLP model
    mlp_model = create_mlp_model(architecture = mlp_architecture,
                                 hidden_activation = mlp_hidden_activation,
                                 output_activation = mlp_output_activation,
                                 use_bias = mlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = mlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = mlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = mlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = mlp_rational_parameters_alpha_initializer,
                                 rational_parameters_beta_initializer = mlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = mlp_rational_parameters_shared_axes)

    X = mlp_model(X)

    # define final model
    model = Model(inputs = [X_mol_graphs, A_mod_mol_graphs], outputs = X)

    return model

# function to create a siamese MLP MLP model

def create_siamese_mlp_mlp_model(smlp_architecture = (2**10, 100),
                                 smlp_hidden_activation = tf.keras.activations.relu,
                                 smlp_output_activation = tf.keras.activations.sigmoid,
                                 smlp_use_bias = True,
                                 smlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                                 smlp_dropout_rates_input_hidden = (0, 0),
                                 smlp_hidden_rational_layers = False,
                                 smlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                 smlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                 smlp_rational_parameters_shared_axes = [1],
                                 mlp_architecture = (100, 100, 1),
                                 mlp_hidden_activation = tf.keras.activations.relu,
                                 mlp_output_activation = tf.keras.activations.sigmoid,
                                 mlp_use_bias = True,
                                 mlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                                 mlp_dropout_rates_input_hidden = (0, 0),
                                 mlp_hidden_rational_layers = False,
                                 mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                 mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                 mlp_rational_parameters_shared_axes = [1]):

    # define input tensors
    X_fp_1 = Input(shape=(smlp_architecture[0],))
    X_fp_2 = Input(shape=(smlp_architecture[0],))


    # define and apply siamese MLP model to create embeddings for both molecules
    smlp_model = create_mlp_model(architecture = smlp_architecture,
                                 hidden_activation = smlp_hidden_activation,
                                 output_activation = smlp_output_activation,
                                 use_bias = smlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = smlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = smlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = smlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = smlp_rational_parameters_alpha_initializer,
                                 rational_parameters_beta_initializer = smlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = smlp_rational_parameters_shared_axes)

    X_emb_1 = smlp_model(X_fp_1)
    X_emb_2 = smlp_model(X_fp_2)

    # combine embeddings to a single vector in a symmetric manner
    X = tf.math.abs(X_emb_1 - X_emb_2)

    # define and apply MLP model
    mlp_model = create_mlp_model(architecture = mlp_architecture,
                                 hidden_activation = mlp_hidden_activation,
                                 output_activation = mlp_output_activation,
                                 use_bias = mlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = mlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = mlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = mlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = mlp_rational_parameters_alpha_initializer,
                                 rational_parameters_beta_initializer = mlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = mlp_rational_parameters_shared_axes)

    X = mlp_model(X)

    # define final model
    model = Model(inputs = [X_fp_1, X_fp_2], outputs = X)

    return (model, smlp_model)





def create_siamese_mlp_mlp_model_with_act_preds(smlp_architecture = (2**10, 100),
                                             smlp_hidden_activation = tf.keras.activations.relu,
                                             smlp_output_activation = tf.keras.activations.sigmoid,
                                             smlp_use_bias = True,
                                             smlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                                             smlp_dropout_rates_input_hidden = (0, 0),
                                             smlp_hidden_rational_layers = False,
                                             smlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                             smlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                             smlp_rational_parameters_shared_axes = [1],
                                             mlp_architecture = (100, 100, 1),
                                             mlp_hidden_activation = tf.keras.activations.relu,
                                             mlp_use_bias = True,
                                             mlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                                             mlp_dropout_rates_input_hidden = (0, 0),
                                             mlp_hidden_rational_layers = False,
                                             mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                             mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                             mlp_rational_parameters_shared_axes = [1]):

    # define siamese MLP model to create embeddings for both molecules
    smlp_model = create_mlp_model(architecture = smlp_architecture,
                                 hidden_activation = smlp_hidden_activation,
                                 output_activation = smlp_output_activation,
                                 use_bias = smlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = smlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = smlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = smlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = smlp_rational_parameters_alpha_initializer,
                                 rational_parameters_beta_initializer = smlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = smlp_rational_parameters_shared_axes)
    
    # define MLP projection heads
    activity_pred_mlp_model = create_mlp_model(architecture = mlp_architecture,
                                             hidden_activation = mlp_hidden_activation,
                                             output_activation = tf.keras.activations.linear,
                                             use_bias = mlp_use_bias,
                                             use_batch_norm_input_hidden_lasthidden = mlp_use_batch_norm_input_hidden_lasthidden,
                                             dropout_rates_input_hidden = mlp_dropout_rates_input_hidden,
                                             hidden_rational_layers = mlp_hidden_rational_layers,
                                             rational_parameters_alpha_initializer = mlp_rational_parameters_alpha_initializer,
                                             rational_parameters_beta_initializer = mlp_rational_parameters_beta_initializer,
                                             rational_parameters_shared_axes = mlp_rational_parameters_shared_axes)

    ac_pred_mlp_model = create_mlp_model(architecture = mlp_architecture,
                                         hidden_activation = mlp_hidden_activation,
                                         output_activation = tf.keras.activations.sigmoid,
                                         use_bias = mlp_use_bias,
                                         use_batch_norm_input_hidden_lasthidden = mlp_use_batch_norm_input_hidden_lasthidden,
                                         dropout_rates_input_hidden = mlp_dropout_rates_input_hidden,
                                         hidden_rational_layers = mlp_hidden_rational_layers,
                                         rational_parameters_alpha_initializer = mlp_rational_parameters_alpha_initializer,
                                         rational_parameters_beta_initializer = mlp_rational_parameters_beta_initializer,
                                         rational_parameters_shared_axes = mlp_rational_parameters_shared_axes)

    # define input tensors
    X_fp_1 = Input(shape=(smlp_architecture[0],))
    X_fp_2 = Input(shape=(smlp_architecture[0],))
    
    # create siamese embeddings
    X_emb_1 = smlp_model(X_fp_1)
    X_emb_2 = smlp_model(X_fp_2)

    # create predictions
    act_1_pred = activity_pred_mlp_model(X_emb_1)
    ac_pred = ac_pred_mlp_model(X_emb_1 + X_emb_2)
    act_2_pred = activity_pred_mlp_model(X_emb_2)

    # define final model
    model = Model(inputs = [X_fp_1, X_fp_2], outputs = [act_1_pred, ac_pred, act_2_pred])

    return (model, smlp_model)




# function to create a siamese MLP cosine similarity model

def create_siamese_mlp_cos_sim_model(smlp_architecture = (2**10, 100),
                                     smlp_hidden_activation = tf.keras.activations.relu,
                                     smlp_output_activation = tf.keras.activations.sigmoid,
                                     smlp_use_bias = True,
                                     smlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                                     smlp_dropout_rates_input_hidden = (0, 0),
                                     smlp_hidden_rational_layers = False,
                                     smlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                     smlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                     smlp_rational_parameters_shared_axes = [1]):

    # define input tensors
    X_fcfp_1 = Input(shape=(smlp_architecture[0],))
    X_fcfp_2 = Input(shape=(smlp_architecture[0],))

    # define and apply siamese MLP model to create embeddings for both molecules
    smlp_model = create_mlp_model(architecture = smlp_architecture,
                                 hidden_activation = smlp_hidden_activation,
                                 output_activation = smlp_output_activation,
                                 use_bias = smlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = smlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = smlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = smlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = smlp_rational_parameters_alpha_initializer,
                                 rational_parameters_beta_initializer = smlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = smlp_rational_parameters_shared_axes)

    X_emb_1 = smlp_model(X_fcfp_1)
    X_emb_2 = smlp_model(X_fcfp_2)

    # compute cos(alpha) for angle alpha between both embeddings
    cos_alpha = tf.math.reduce_sum(X_emb_1 * X_emb_2, axis = 1)/(tf.norm(X_emb_1, axis = 1) * tf.norm(X_emb_2, axis = 1))

    # define final output via sigmoid function
    sigmoid_cos_alpha = tf.math.sigmoid(cos_alpha)

    # define final model
    model = Model(inputs = [X_fcfp_1, X_fcfp_2], outputs = 1 - sigmoid_cos_alpha)

    return (model, smlp_model)

# function to create a siamese GCN MLP model

def create_siamese_gcn_mlp_model(gcn_n_node_features = 25,
                                 gcn_n_hidden = 2,
                                 gcn_channels = 100,
                                 gcn_activation = tf.keras.activations.relu,
                                 gcn_use_bias = True,
                                 gcn_use_batch_norm_input_hidden_lasthidden = (False, True, False),
                                 gcn_dropout_rates_input_hidden = (0, 0),
                                 dgcn_architecture = (100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
                                 dgcn_hidden_activation = tf.keras.activations.relu,
                                 dgcn_output_activation = tf.keras.activations.relu,
                                 dgcn_use_bias = True,
                                 dgcn_use_batch_norm_input_hidden_lasthidden = (True, True, True),
                                 dgcn_dropout_rates_input_hidden = (0, 0),
                                 dgcn_hidden_rational_layers = False,
                                 dgcn_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218] ,
                                 dgcn_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                 dgcn_rational_parameters_shared_axes = [1],
                                 mlp_architecture = (100, 100, 100, 1),
                                 mlp_hidden_activation = tf.keras.activations.relu,
                                 mlp_output_activation = tf.keras.activations.sigmoid,
                                 mlp_use_bias = True,
                                 mlp_use_batch_norm_input_hidden_lasthidden = (True, True, False),
                                 mlp_dropout_rates_input_hidden = (0, 0),
                                 mlp_hidden_rational_layers = False,
                                 mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                 mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                 mlp_rational_parameters_shared_axes = [1]):

    # define input tensors
    X_mol_graphs_1 = Input(shape=(None, gcn_n_node_features))
    A_mod_mol_graphs_1 = Input(shape=(None, None))
    X_mol_graphs_2 = Input(shape=(None, gcn_n_node_features))
    A_mod_mol_graphs_2 = Input(shape=(None, None))

    # define and apply GCN model to vectorise molecular graph arrays
    gcn_model = create_gcn_model(gcn_n_node_features = gcn_n_node_features,
                                 gcn_n_hidden = gcn_n_hidden,
                                 gcn_channels = gcn_channels,
                                 gcn_activation = gcn_activation,
                                 gcn_use_bias = gcn_use_bias,
                                 gcn_use_batch_norm_input_hidden_lasthidden = gcn_use_batch_norm_input_hidden_lasthidden,
                                 gcn_dropout_rates_input_hidden = gcn_dropout_rates_input_hidden)

    X_vec_1 = gcn_model([X_mol_graphs_1, A_mod_mol_graphs_1])
    X_vec_2 = gcn_model([X_mol_graphs_2, A_mod_mol_graphs_2])

    # define and apply DGCN model (dense layer on top of GCN) to create molecular embeddings
    dgcn_model = create_mlp_model(architecture = dgcn_architecture,
                                  hidden_activation = dgcn_hidden_activation,
                                  output_activation = dgcn_output_activation,
                                  use_bias = dgcn_use_bias,
                                  use_batch_norm_input_hidden_lasthidden = dgcn_use_batch_norm_input_hidden_lasthidden,
                                  dropout_rates_input_hidden = dgcn_dropout_rates_input_hidden,
                                  hidden_rational_layers = dgcn_hidden_rational_layers,
                                  rational_parameters_alpha_initializer = dgcn_rational_parameters_alpha_initializer,
                                  rational_parameters_beta_initializer = dgcn_rational_parameters_beta_initializer,
                                  rational_parameters_shared_axes = dgcn_rational_parameters_shared_axes)

    X_emb_1 = dgcn_model(X_vec_1)
    X_emb_2 = dgcn_model(X_vec_2)

    # combine embeddings to a single vector in a symmetric manner
    X = tf.math.abs(X_emb_1 - X_emb_2)

    # define and apply MLP model
    mlp_model = create_mlp_model(architecture = mlp_architecture,
                                 hidden_activation = mlp_hidden_activation,
                                 output_activation = mlp_output_activation,
                                 use_bias = mlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = mlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = mlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = mlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = mlp_rational_parameters_alpha_initializer,
                                 rational_parameters_beta_initializer = mlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = mlp_rational_parameters_shared_axes)

    X = mlp_model(X)

    # define final models
    model = Model(inputs = [X_mol_graphs_1, A_mod_mol_graphs_1, X_mol_graphs_2, A_mod_mol_graphs_2], outputs = X)
    sgcn_model = Model(inputs = [X_mol_graphs_1, A_mod_mol_graphs_1], outputs = X_emb_1)

    return (model, sgcn_model)

# function to create a trained siamese MLP MLP model

def create_trained_siamese_mlp_mlp_model(x_smiles_train,
                                         y_train,
                                         x_smiles_to_fcfp_dict,
                                         X_smiles_mmps,
                                         smlp_architecture = (2**10, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
                                         smlp_hidden_activation = tf.keras.activations.relu,
                                         smlp_output_activation = tf.keras.activations.linear,
                                         smlp_use_bias = True,
                                         smlp_use_batch_norm_input_hidden_lasthidden = (False, True, True),
                                         smlp_dropout_rates_input_hidden = (0, 0),
                                         smlp_hidden_rational_layers = False,
                                         smlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218] ,
                                         smlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                         smlp_rational_parameters_shared_axes = [1],
                                         mlp_architecture = (100, 100, 100, 1),
                                         mlp_hidden_activation = tf.keras.activations.relu,
                                         mlp_output_activation = tf.keras.activations.sigmoid,
                                         mlp_use_bias = True,
                                         mlp_use_batch_norm_input_hidden_lasthidden = (True, True, False),
                                         mlp_dropout_rates_input_hidden = (0, 0),
                                         mlp_hidden_rational_layers = False,
                                         mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218] ,
                                         mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                         mlp_rational_parameters_shared_axes = [1],
                                         batch_size = 2**9,
                                         epochs = 1,
                                         optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
                                         loss = tf.keras.losses.BinaryCrossentropy(),
                                         performance_metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.BinaryAccuracy()],
                                         verbose = 1):

    # get indices for mmps for which both molecules are in training set
    ind_train_mmps = []
    for (k, [smiles_1, smiles_2]) in enumerate(X_smiles_mmps):
        if smiles_1 in x_smiles_train and smiles_2 in x_smiles_train:
            ind_train_mmps.append(k)

    # extract mmps which lie in training set
    X_smiles_mmps_train = X_smiles_mmps[ind_train_mmps]

    # label the extracted mmps
    y_train_smiles_to_label_dict = dict(list(zip(x_smiles_train, y_train)))
    y_mmps_train = np.array([int(y_train_smiles_to_label_dict[smiles_1]!=y_train_smiles_to_label_dict[smiles_2]) for [smiles_1, smiles_2] in X_smiles_mmps_train])

    # construct binary predictor variables X_fcfp_1 and X_fcfp_2 ( = fixed molecular features)
    X_fcfp_1_train = list(range(len(X_smiles_mmps_train)))
    X_fcfp_2_train = list(range(len(X_smiles_mmps_train)))

    for k in range(len(X_smiles_mmps_train)):

        X_fcfp_1_train[k] = x_smiles_to_fcfp_dict[X_smiles_mmps_train[k,0]]
        X_fcfp_2_train[k] = x_smiles_to_fcfp_dict[X_smiles_mmps_train[k,1]]

    X_fcfp_1_train = np.array(X_fcfp_1_train)
    X_fcfp_2_train = np.array(X_fcfp_2_train)


    # create fresh SMLP model
    (smlp_mlp_model, smlp_model) = create_siamese_mlp_mlp_model(smlp_architecture = smlp_architecture,
                                                                smlp_hidden_activation = smlp_hidden_activation,
                                                                smlp_output_activation = smlp_output_activation,
                                                                smlp_use_bias = smlp_use_bias,
                                                                smlp_use_batch_norm_input_hidden_lasthidden = smlp_use_batch_norm_input_hidden_lasthidden,
                                                                smlp_dropout_rates_input_hidden = smlp_dropout_rates_input_hidden,
                                                                smlp_hidden_rational_layers = smlp_hidden_rational_layers,
                                                                smlp_rational_parameters_alpha_initializer = smlp_rational_parameters_alpha_initializer,
                                                                smlp_rational_parameters_beta_initializer = smlp_rational_parameters_beta_initializer,
                                                                smlp_rational_parameters_shared_axes = smlp_rational_parameters_shared_axes,
                                                                mlp_architecture = mlp_architecture,
                                                                mlp_hidden_activation = mlp_hidden_activation,
                                                                mlp_output_activation = mlp_output_activation,
                                                                mlp_use_bias = mlp_use_bias,
                                                                mlp_use_batch_norm_input_hidden_lasthidden = mlp_use_batch_norm_input_hidden_lasthidden,
                                                                mlp_dropout_rates_input_hidden = mlp_dropout_rates_input_hidden,
                                                                mlp_hidden_rational_layers = mlp_hidden_rational_layers,
                                                                mlp_rational_parameters_alpha_initializer = mlp_rational_parameters_alpha_initializer,
                                                                mlp_rational_parameters_beta_initializer = mlp_rational_parameters_beta_initializer,
                                                                mlp_rational_parameters_shared_axes = mlp_rational_parameters_shared_axes)

    # compile the model
    smlp_mlp_model.compile(optimizer = optimizer,
                           loss = loss,
                           metrics = performance_metrics)

    # fit the model
    if epochs > 0:

        smlp_mlp_model.fit([X_fcfp_1_train, X_fcfp_2_train],
                            y_mmps_train,
                            epochs = epochs,
                            batch_size = batch_size,
                            verbose = verbose)

    return (smlp_mlp_model, smlp_model)


def create_siamese_mlp_mlp_molecular_ensemble_classifier(smlp_mlp_model,
                                                         X_training_space_fcfp,
                                                         y_train):

    def siamese_mlp_mlp_molecular_ensemble_classifier(X_fcfp_test):

        n_training_space = len(X_training_space_fcfp)
        n_test = len(X_fcfp_test)
        final_predictions = np.zeros((n_test, 1))

        for (k, training_space_fcfp) in enumerate(list(X_training_space_fcfp)):

            Repeated_training_space_fcfp = np.repeat(np.array([training_space_fcfp]), repeats = n_test, axis = 0)


            intermediate_predictions = smlp_mlp_model([Repeated_training_space_fcfp, X_fcfp_test])

            if y_train[k] == 0:
                final_predictions += intermediate_predictions
            elif y_train[k] == 1:
                final_predictions += 1-intermediate_predictions

        final_predictions = np.array(final_predictions/n_training_space)

        return final_predictions

    return siamese_mlp_mlp_molecular_ensemble_classifier


# function to create dual task siamese mlp mlp model

def create_dual_task_siamese_mlp_mlp_model(smlp_architecture = (2**10, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
                                           smlp_hidden_activation = tf.keras.activations.relu,
                                           smlp_output_activation = tf.keras.activations.sigmoid,
                                           smlp_use_bias = True,
                                           smlp_use_batch_norm_input_hidden_lasthidden = (False, True, False),
                                           smlp_dropout_rates_input_hidden = (0, 0),
                                           smlp_hidden_rational_layers = False,
                                           smlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                           smlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                           smlp_rational_parameters_shared_axes = [1],
                                           ac_mlp_architecture = (100, 100, 100, 1),
                                           ac_mlp_hidden_activation = tf.keras.activations.relu,
                                           ac_mlp_output_activation = tf.keras.activations.sigmoid,
                                           ac_mlp_use_bias = True,
                                           ac_mlp_use_batch_norm_input_hidden_lasthidden = (True, True, False),
                                           ac_mlp_dropout_rates_input_hidden = (0, 0),
                                           ac_mlp_hidden_rational_layers = False,
                                           ac_mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                           ac_mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                           ac_mlp_rational_parameters_shared_axes = [1],
                                           pd_mlp_architecture = (100, 100, 100, 1),
                                           pd_mlp_hidden_activation = tf.keras.activations.tanh,
                                           pd_mlp_output_activation = tf.keras.activations.sigmoid,
                                           pd_mlp_use_bias = False,
                                           pd_mlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                                           pd_mlp_dropout_rates_input_hidden = (0, 0),
                                           pd_mlp_hidden_rational_layers = False,
                                           pd_mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                           pd_mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                           pd_mlp_rational_parameters_shared_axes = [1]):

    # define input tensors
    X_fp_1 = Input(shape=(smlp_architecture[0],))
    X_fp_2 = Input(shape=(smlp_architecture[0],))

    # define and apply siamese MLP model to create embeddings for both molecules
    smlp_model = create_mlp_model(architecture = smlp_architecture,
                                 hidden_activation = smlp_hidden_activation,
                                 output_activation = smlp_output_activation,
                                 use_bias = smlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = smlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = smlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = smlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = smlp_rational_parameters_alpha_initializer,
                                 rational_parameters_beta_initializer = smlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = smlp_rational_parameters_shared_axes)

    X_emb_1 = smlp_model(X_fp_1)
    X_emb_2 = smlp_model(X_fp_2)

    # combine embeddings to two new vectors for ac and pd prediction
    X_ac = tf.math.abs(X_emb_1 - X_emb_2)
    
    X_pd = X_emb_1 - X_emb_2

    # define and apply MLP model for ac prediction
    ac_mlp_model = create_mlp_model(architecture = ac_mlp_architecture,
                                    hidden_activation = ac_mlp_hidden_activation,
                                    output_activation = ac_mlp_output_activation,
                                    use_bias = ac_mlp_use_bias,
                                    use_batch_norm_input_hidden_lasthidden = ac_mlp_use_batch_norm_input_hidden_lasthidden,
                                    dropout_rates_input_hidden = ac_mlp_dropout_rates_input_hidden,
                                    hidden_rational_layers = ac_mlp_hidden_rational_layers,
                                    rational_parameters_alpha_initializer = ac_mlp_rational_parameters_alpha_initializer,
                                    rational_parameters_beta_initializer = ac_mlp_rational_parameters_beta_initializer,
                                    rational_parameters_shared_axes = ac_mlp_rational_parameters_shared_axes,
                                    model_name = "ac_mlp")

    X_ac_pred = ac_mlp_model(X_ac)

    # define and apply MLP model for pd prediction
    pd_mlp_model = create_mlp_model(architecture = pd_mlp_architecture,
                                     hidden_activation = pd_mlp_hidden_activation,
                                     output_activation = pd_mlp_output_activation,
                                     use_bias = pd_mlp_use_bias,
                                     use_batch_norm_input_hidden_lasthidden = pd_mlp_use_batch_norm_input_hidden_lasthidden,
                                     dropout_rates_input_hidden = pd_mlp_dropout_rates_input_hidden,
                                     hidden_rational_layers = pd_mlp_hidden_rational_layers,
                                     rational_parameters_alpha_initializer = pd_mlp_rational_parameters_alpha_initializer,
                                     rational_parameters_beta_initializer = pd_mlp_rational_parameters_beta_initializer,
                                     rational_parameters_shared_axes = pd_mlp_rational_parameters_shared_axes,
                                     model_name = "pd_mlp")

    X_pd_pred = pd_mlp_model(X_pd)

    # define final model
    model = Model(inputs = [X_fp_1, X_fp_2], outputs = [X_ac_pred, X_pd_pred])

    return (model, smlp_model)

# function to create dual task siamese gcn mlp model

def create_dual_task_siamese_gcn_mlp_model(gcn_n_node_features = 25,
                                           gcn_n_hidden = 2,
                                           gcn_channels = 100,
                                           gcn_activation = tf.keras.activations.relu,
                                           gcn_use_bias = True,
                                           gcn_use_batch_norm_input_hidden_lasthidden = (False, True, False),
                                           gcn_dropout_rates_input_hidden = (0, 0),
                                           dgcn_architecture = (100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
                                           dgcn_hidden_activation = tf.keras.activations.relu,
                                           dgcn_output_activation = tf.keras.activations.linear,
                                           dgcn_use_bias = True,
                                           dgcn_use_batch_norm_input_hidden_lasthidden = (True, True, False),
                                           dgcn_dropout_rates_input_hidden = (0, 0),
                                           dgcn_hidden_rational_layers = False,
                                           dgcn_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218] ,
                                           dgcn_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                           dgcn_rational_parameters_shared_axes = [1],
                                           ac_mlp_architecture = (100, 100, 100, 1),
                                           ac_mlp_hidden_activation = tf.keras.activations.relu,
                                           ac_mlp_output_activation = tf.keras.activations.sigmoid,
                                           ac_mlp_use_bias = True,
                                           ac_mlp_use_batch_norm_input_hidden_lasthidden = (True, True, False),
                                           ac_mlp_dropout_rates_input_hidden = (0, 0),
                                           ac_mlp_hidden_rational_layers = False,
                                           ac_mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                           ac_mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                           ac_mlp_rational_parameters_shared_axes = [1],
                                           pd_mlp_architecture = (100, 100, 100, 1),
                                           pd_mlp_hidden_activation = tf.keras.activations.tanh,
                                           pd_mlp_output_activation = tf.keras.activations.sigmoid,
                                           pd_mlp_use_bias = False,
                                           pd_mlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                                           pd_mlp_dropout_rates_input_hidden = (0, 0),
                                           pd_mlp_hidden_rational_layers = False,
                                           pd_mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                           pd_mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                           pd_mlp_rational_parameters_shared_axes = [1]):

    # define input tensors
    X_mol_graphs_1 = Input(shape=(None, gcn_n_node_features))
    A_mod_mol_graphs_1 = Input(shape=(None, None))
    X_mol_graphs_2 = Input(shape=(None, gcn_n_node_features))
    A_mod_mol_graphs_2 = Input(shape=(None, None))

    # define and apply GCN model to vectorise molecular graph arrays
    gcn_model = create_gcn_model(gcn_n_node_features = gcn_n_node_features,
                                 gcn_n_hidden = gcn_n_hidden,
                                 gcn_channels = gcn_channels,
                                 gcn_activation = gcn_activation,
                                 gcn_use_bias = gcn_use_bias,
                                 gcn_use_batch_norm_input_hidden_lasthidden = gcn_use_batch_norm_input_hidden_lasthidden,
                                 gcn_dropout_rates_input_hidden = gcn_dropout_rates_input_hidden)

    X_vec_1 = gcn_model([X_mol_graphs_1, A_mod_mol_graphs_1])
    X_vec_2 = gcn_model([X_mol_graphs_2, A_mod_mol_graphs_2])

    # define and apply DGCN model (dense layer on top of GCN) to create molecular embeddings
    dgcn_model = create_mlp_model(architecture = dgcn_architecture,
                                  hidden_activation = dgcn_hidden_activation,
                                  output_activation = dgcn_output_activation,
                                  use_bias = dgcn_use_bias,
                                  use_batch_norm_input_hidden_lasthidden = dgcn_use_batch_norm_input_hidden_lasthidden,
                                  dropout_rates_input_hidden = dgcn_dropout_rates_input_hidden,
                                  hidden_rational_layers = dgcn_hidden_rational_layers,
                                  rational_parameters_alpha_initializer = dgcn_rational_parameters_alpha_initializer,
                                  rational_parameters_beta_initializer = dgcn_rational_parameters_beta_initializer,
                                  rational_parameters_shared_axes = dgcn_rational_parameters_shared_axes)

    X_emb_1 = dgcn_model(X_vec_1)
    X_emb_2 = dgcn_model(X_vec_2)

    # combine embeddings to two new vectors for ac and pd prediction
    X_ac = tf.math.abs(X_emb_1 - X_emb_2)
    X_pd = X_emb_1 - X_emb_2

    # define and apply MLP model for ac prediction
    ac_mlp_model = create_mlp_model(architecture = ac_mlp_architecture,
                                    hidden_activation = ac_mlp_hidden_activation,
                                    output_activation = ac_mlp_output_activation,
                                    use_bias = ac_mlp_use_bias,
                                    use_batch_norm_input_hidden_lasthidden = ac_mlp_use_batch_norm_input_hidden_lasthidden,
                                    dropout_rates_input_hidden = ac_mlp_dropout_rates_input_hidden,
                                    hidden_rational_layers = ac_mlp_hidden_rational_layers,
                                    rational_parameters_alpha_initializer = ac_mlp_rational_parameters_alpha_initializer,
                                    rational_parameters_beta_initializer = ac_mlp_rational_parameters_beta_initializer,
                                    rational_parameters_shared_axes = ac_mlp_rational_parameters_shared_axes,
                                    model_name = "ac_mlp")

    X_ac_pred = ac_mlp_model(X_ac)

    # define and apply MLP model for pd prediction
    pd_mlp_model = create_mlp_model(architecture = pd_mlp_architecture,
                                     hidden_activation = pd_mlp_hidden_activation,
                                     output_activation = pd_mlp_output_activation,
                                     use_bias = pd_mlp_use_bias,
                                     use_batch_norm_input_hidden_lasthidden = pd_mlp_use_batch_norm_input_hidden_lasthidden,
                                     dropout_rates_input_hidden = pd_mlp_dropout_rates_input_hidden,
                                     hidden_rational_layers = pd_mlp_hidden_rational_layers,
                                     rational_parameters_alpha_initializer = pd_mlp_rational_parameters_alpha_initializer,
                                     rational_parameters_beta_initializer = pd_mlp_rational_parameters_beta_initializer,
                                     rational_parameters_shared_axes = pd_mlp_rational_parameters_shared_axes,
                                     model_name = "pd_mlp")

    X_pd_pred = pd_mlp_model(X_pd)

    # define final model
    model = Model(inputs = [X_mol_graphs_1, A_mod_mol_graphs_1, X_mol_graphs_2, A_mod_mol_graphs_2], outputs = [X_ac_pred, X_pd_pred])
    sgcn_model = Model(inputs = [X_mol_graphs_1, A_mod_mol_graphs_1], outputs = X_emb_1)

    return (model, sgcn_model)



def create_siamese_qsar_mlp_model(smlp_architecture = (1024, 100, 1),
                                  smlp_hidden_activation = tf.keras.activations.relu,
                                  smlp_output_activation = tf.keras.activations.linear,
                                  smlp_use_bias = True,
                                  smlp_use_batch_norm_input_hidden_lasthidden = (False, True, False),
                                  smlp_dropout_rates_input_hidden = (0, 0),
                                  smlp_hidden_rational_layers = False,
                                  smlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                  smlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                  smlp_rational_parameters_shared_axes = [1]):

    # define input tensors
    X_fp_1 = Input(shape=(smlp_architecture[0],))
    X_fp_2 = Input(shape=(smlp_architecture[0],))

    # define and apply siamese MLP model to create embeddings for both molecules
    smlp_model = create_mlp_model(architecture = smlp_architecture,
                                 hidden_activation = smlp_hidden_activation,
                                 output_activation = smlp_output_activation,
                                 use_bias = smlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = smlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = smlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = smlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = smlp_rational_parameters_alpha_initializer,
                                 rational_parameters_beta_initializer = smlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = smlp_rational_parameters_shared_axes)

    activity_1 = smlp_model(X_fp_1)
    activity_2 = smlp_model(X_fp_2)

    # combine embeddings to a single vector in a symmetric manner
    activity_abs_diff = tf.math.abs(activity_1 - activity_2)

    
    # define final model
    model = Model(inputs = [X_fp_1, X_fp_2], outputs = activity_abs_diff)

    return (model, smlp_model)


def create_siamese_qsar_mlp_model_with_act_preds(smlp_architecture = (1024, 100, 1),
                                              smlp_hidden_activation = tf.keras.activations.relu,
                                              smlp_output_activation = tf.keras.activations.linear,
                                              smlp_use_bias = True,
                                              smlp_use_batch_norm_input_hidden_lasthidden = (False, True, False),
                                              smlp_dropout_rates_input_hidden = (0, 0),
                                              smlp_hidden_rational_layers = False,
                                              smlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                              smlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                              smlp_rational_parameters_shared_axes = [1]):

    # define input tensors
    X_fp_1 = Input(shape=(smlp_architecture[0],))
    X_fp_2 = Input(shape=(smlp_architecture[0],))

    # define and apply siamese MLP model to create embeddings for both molecules
    smlp_model = create_mlp_model(architecture = smlp_architecture,
                                 hidden_activation = smlp_hidden_activation,
                                 output_activation = smlp_output_activation,
                                 use_bias = smlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = smlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = smlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = smlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = smlp_rational_parameters_alpha_initializer,
                                 rational_parameters_beta_initializer = smlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = smlp_rational_parameters_shared_axes)

    activity_1 = smlp_model(X_fp_1)
    activity_2 = smlp_model(X_fp_2)

    # combine embeddings to a single vector in a symmetric manner
    activity_abs_diff = tf.math.abs(activity_1 - activity_2)

    
    # define final model
    model = Model(inputs = [X_fp_1, X_fp_2], outputs = [activity_1, activity_abs_diff, activity_2])

    return (model, smlp_model)





def create_siamese_mlp_mlp_model_vars_and_core(smlp_architecture = (2**10, 100),
                                                 smlp_hidden_activation = tf.keras.activations.relu,
                                                 smlp_output_activation = tf.keras.activations.sigmoid,
                                                 smlp_use_bias = True,
                                                 smlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                                                 smlp_dropout_rates_input_hidden = (0, 0),
                                                 smlp_hidden_rational_layers = False,
                                                 smlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                                 smlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                                 smlp_rational_parameters_shared_axes = [1],
                                                 mlp_architecture = (100 + 1024, 100, 1),
                                                 mlp_hidden_activation = tf.keras.activations.relu,
                                                 mlp_output_activation = tf.keras.activations.sigmoid,
                                                 mlp_use_bias = True,
                                                 mlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                                                 mlp_dropout_rates_input_hidden = (0, 0),
                                                 mlp_hidden_rational_layers = False,
                                                 mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                                 mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                                 mlp_rational_parameters_shared_axes = [1]):

    # define input tensors
    X_fp_var_1 = Input(shape=(smlp_architecture[0],))
    X_fp_var_2 = Input(shape=(smlp_architecture[0],))
    X_fp_core = Input(shape=(mlp_architecture[0] - smlp_architecture[-1],))


    # define and apply siamese MLP model to create embeddings for the variable parts of both molecules
    smlp_model = create_mlp_model(architecture = smlp_architecture,
                                 hidden_activation = smlp_hidden_activation,
                                 output_activation = smlp_output_activation,
                                 use_bias = smlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = smlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = smlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = smlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = smlp_rational_parameters_alpha_initializer,
                                 rational_parameters_beta_initializer = smlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = smlp_rational_parameters_shared_axes)

    X_emb_var_1 = smlp_model(X_fp_var_1)
    X_emb_var_2 = smlp_model(X_fp_var_2)

    # combine variable part embeddings to a single vector in a symmetric manner
    X_var_sym = X_emb_var_1 + X_emb_var_2
    
    # concatenate X_var_sym and X_fp_core
    X = tf.concat([X_var_sym, X_fp_core], axis = 1)

    # define and apply MLP model
    mlp_model = create_mlp_model(architecture = mlp_architecture,
                                 hidden_activation = mlp_hidden_activation,
                                 output_activation = mlp_output_activation,
                                 use_bias = mlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = mlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = mlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = mlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = mlp_rational_parameters_alpha_initializer,
                                 rational_parameters_beta_initializer = mlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = mlp_rational_parameters_shared_axes)

    X = mlp_model(X)

    # define final model
    model = Model(inputs = [X_fp_var_1, X_fp_var_2, X_fp_core], outputs = X)

    return (model, smlp_model)
