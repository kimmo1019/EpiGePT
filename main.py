import tensorflow as tf
import numpy as np 
import random
import sys, os, gc
import pandas as pd
from absl import app, flags, logging
from geformer import Geformer

DTYPE_MAP = {
    "fp16": tf.float16,
    "bf16": tf.bfloat16,
    "fp32": tf.float32,
}

#devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(devices[0], True)

def one_hot_encoding(seq):
    d = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'N':4, 'n':4}
    mat = np.zeros((len(seq),5))  
    for i in range(len(seq)):
        mat[i, d[seq[i]]] = 1
    mat = mat[:,:4]
    return mat

class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        #self.model.save_weights('checkpoints/model_epoch%d.h5' % epoch)
        current = logs.get("val_loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

def make_gen_callable(params, batch_size, data_queue, seq_mat, expr_mat, 
                    motifscore_mat, targets_mat):
    def generate_batch():
        batch_seq = np.empty((batch_size, params["seq_length"]*2**params["num_cb"], 1, 4))
        #after conv shape [batch_size, 1000, 2048-711]
        batch_expr = np.empty((batch_size, params["seq_length"], params["hidden_size"]-params["num_channels"]))
        batch_motifscore = np.empty((batch_size, params["seq_length"], params["hidden_size"]-params["num_channels"]))
        batch_targets = np.empty((batch_size, params["seq_length"], params["num_targets"]))
        empty_array = np.empty((batch_size, params["num_heads"],
                                params["seq_length"],params["seq_length"]))
        batch_dummy_att_weights = {'layer_%d' % i : empty_array for i in range(params["num_hidden_layers"])}
        for i in range(params["num_hidden_layers"]):
            batch_dummy_att_weights['layer_%d' % i] = np.empty((batch_size, params["num_heads"],
                                                        params["seq_length"],params["seq_length"]))
        while True:
            for step in range(len(data_queue) // batch_size):
                present_queue = data_queue[step*batch_size : (step+1)*batch_size]
                for i in range(len(present_queue)):
                    cell_idx, seq_idx = present_queue[i]
                    batch_seq[i] = seq_mat[seq_idx]
                    batch_expr[i] = np.tile(expr_mat[:, cell_idx], (params["seq_length"], 1))
                    batch_motifscore[i] = motifscore_mat[seq_idx]
                    batch_targets[i] = targets_mat[cell_idx, seq_idx, :, :]
                yield [batch_seq, batch_expr * batch_motifscore], [batch_targets, batch_dummy_att_weights]
    return generate_batch
        

def load_inputs(params,
                seq_file = 'data/encode/overlap_count_gt50.128k.bin',
                expr_file = 'data/encode/aggregated_tf_expr.csv',
                motifscore_file = 'data/motif/motifscore.npy',
                targets_file = 'data/encode/targets_data.npy',
                genome_file = '../hg19.fa'):
    """Returns:
        seq_mat: DNA sequences with shape [13300, 128k, 1, 4]
        expr_mat: TFs expression with shape [711, 28]
        motifscore_mat: motif score matrix with shape [13300, 1000, 711]
        targets_mat: target readscount (DNase/ChIP-seq) with shape [28, 13300, 1000, 8]
    """

    #load one-seq encoding of DNA sequences
    from pyfasta import Fasta
    genome = Fasta(genome_file)
    dna_seqs = open(seq_file).readlines()
    seq_mat = np.empty((len(dna_seqs), params["seq_length"]*2**params["num_cb"], 1, 4))
    for i in range(len(dna_seqs)):
        line = dna_seqs[i]
        chrom,start,end = line.split('\t')[0], int(line.split('\t')[1]), int(line.split('\t')[2].strip())
        seq = genome[chrom][start : end]
        seq_mat[i, :, 0, :] = one_hot_encoding(seq)
        
    expr_mat = pd.read_csv(expr_file, header=0, index_col=[0], sep='\t').values
    expr_mat = np.log(expr_mat + 1)

    motifscore_mat = np.load(motifscore_file).reshape(len(dna_seqs), params["seq_length"], expr_mat.shape[0])
    targets_mat = np.load(targets_file)
    for i in range(params["num_targets"]):
        N_i = np.sum(targets_mat[:,:,i], axis = 1)
        N = np.min(N_i)
        for j in range(targets_mat.shape[0]):
            targets_mat[j,:,i] = np.log(1. + targets_mat[j,:,i]*N/N_i[j])
    targets_mat = targets_mat.reshape((expr_mat.shape[1], len(dna_seqs), params["seq_length"], params["num_targets"]))
    #raw readscount, normalize, log(1+xij*N/Ni)
    return seq_mat, expr_mat, motifscore_mat, targets_mat

def evaluate(params, model, test_cell_idx, seq_mat, expr_mat, motifscore_mat, 
            targets_mat, batch_size = 1000):
    from scipy.stats import pearsonr
    corr_all = np.empty((len(test_cell_idx), targets_mat.shape[-1]))
    for cell_idx in test_cell_idx:
        print('cell idx',cell_idx)
        target_pre = np.empty((motifscore_mat.shape[0], params["seq_length"], targets_mat.shape[-1]))
        for batch_idx in range(int(np.ceil(motifscore_mat.shape[0] / batch_size))):
            if (batch_idx+1)*batch_size > motifscore_mat.shape[0]:
                ind = np.arange(batch_idx * batch_size, motifscore_mat.shape[0])
            else:
                ind = np.arange(batch_idx * batch_size, (batch_idx+1)*batch_size)
            expr_mat_batch = np.tile(expr_mat[:, cell_idx], (len(ind), params["seq_length"], 1))
            motifscore_mat_batch = motifscore_mat[ind, :, :]
            seq_mat_batch = seq_mat[ind, :, :, :]
            #seq_mat_batch = tf.convert_to_tensor(seq_mat_batch)
            #expr_motif_batch = tf.convert_to_tensor(expr_mat_batch*motifscore_mat_batch)
            batch_pre, batch_att_weights = model([seq_mat_batch, expr_mat_batch*motifscore_mat_batch], training = False)
            batch_pre, batch_att_weights = model.predict_on_batch([seq_mat_batch, expr_mat_batch*motifscore_mat_batch])
            #batch_pre, batch_att_weights = model.predict([seq_mat_batch, expr_mat_batch*motifscore_mat_batch], 
            #                        batch_size = 4,
            #                        verbose=1,
            #                        callbacks=[ClearMemory()])
                                    #max_queue_size=10,
                                    #workers=1,
                                    #use_multiprocessing=False)
            #_ = gc.collect()
            target_pre[ind, :, :] = batch_pre
        #target_pre with shape [13300, 1000, 8]
        for i in range(targets_mat.shape[-1]):
            corr_all[test_cell_idx.index(cell_idx)][i] = pearsonr(targets_mat[cell_idx,:,:,i].flatten(), target_pre[:,:,i].flatten())[0]
        np.save('results/target_pre_cell_%d.npy' % cell_idx, target_pre) 
        np.save('results/corr_cell_%d.npy' %cell_idx, corr_all[test_cell_idx.index(cell_idx), :])
        print(corr_all[test_cell_idx.index(cell_idx), :])
    print(np.mean(corr_all,axis = 0), np.mean(corr_all,axis = 1))
    #np.save('corr.npy', corr_all)

def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

def dummy_loss(y_true, y_pred):
    return 0.0

def train(params):
    #strategy = tf.distribute.MirroredStrategy()
    #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    #with strategy.scope():
    model = Geformer(params)
    #model.compile(optimizer, loss = tf.keras.losses.MeanSquaredError())
    model.compile(tf.keras.optimizers.Adam(learning_rate = params["lr"]), 
                loss = [tf.keras.losses.MeanSquaredError(), dummy_loss],
                loss_weights=[1., 0.0], run_eagerly=True)
    # seq_mat, expr_mat, motifscore_mat, targets_mat = load_inputs(params)
    # seq_mat = seq_mat[:3000,:,:,:]
    # motifscore_mat = motifscore_mat[:3000, :, :]
    # targets_mat = targets_mat[:,:3000,:,:]
    seq_mat = np.random.normal(size=(20, 128000, 1, 4))
    expr_mat = np.random.normal(size=([711, 28]))
    motifscore_mat = np.random.normal(size=(20, 1000, 711))
    targets_mat = np.random.normal(size=(28, 20, 1000, 8))
    print('load data done')
    print(seq_mat.shape, expr_mat.shape, motifscore_mat.shape, targets_mat.shape)
    np.random.seed(0)
    train_cell_idx = np.random.choice(targets_mat.shape[0], int(params["tr_frac"]*targets_mat.shape[0]), replace = False)
    test_cell_idx = [item for item in np.arange(targets_mat.shape[0]) if item not in train_cell_idx]
    data_queue = []
    for i in train_cell_idx:
        for j in range(seq_mat.shape[0]):
            data_queue.append([i, j])
    random.seed(0)
    random.shuffle(data_queue)
    train_queue = data_queue[:-int(params["val_frac"]*len(data_queue))]
    valid_queue = data_queue[-int(params["val_frac"]*len(data_queue)):]
    print(train_cell_idx, test_cell_idx)
    print(len(train_queue),len(valid_queue))
    train_generator = make_gen_callable(params, params["bs"], train_queue, 
                    seq_mat, expr_mat, motifscore_mat, targets_mat)
    validation_generator = make_gen_callable(params, len(valid_queue), valid_queue, 
                    seq_mat, expr_mat, motifscore_mat, targets_mat)
    validation_data = next(validation_generator())
    # dataset = tf.data.Dataset.from_generator(train_generator,
    #                 output_types = ((tf.float32, tf.float32),
    #                                 (tf.float32, tf.float32)),
    #                 output_shapes = (([128000,1,4], [1000,711]),
    #                                 ([1000,8], [3,2,1000,1000])))
    # dataset = dataset.batch(params["bs"])

    print(validation_data[0][0].shape,validation_data[0][1].shape,validation_data[1][0].shape)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = './checkpoints/best_ckpt.h5',
                             monitor='val_loss', save_weights_only=True),
                EarlyStoppingAtMinLoss(patience=2)]
    history = model.fit(x = train_generator(), steps_per_epoch = len(train_queue)//params["bs"], 
                epochs = 1, verbose = 1, batch_size = params["bs"], callbacks = callbacks,
                validation_data = validation_data,
                max_queue_size = 100, workers = 28,use_multiprocessing = True)
                #)
    model.load_weights('checkpoints/ckpt.h5')
    print(model.summary())
    print('Model training done')
    #model.save_weights("checkpoints/model.h5")
    seq_mat, expr_mat, motifscore_mat, targets_mat = load_inputs(params)
    seq_mat = seq_mat[:3000,:,:,:]
    motifscore_mat = motifscore_mat[:3000, :, :]
    targets_mat = targets_mat[:,:3000,:,:]
    # seq_mat = np.random.normal(size=(1000, 128000, 1, 4))
    # expr_mat = np.random.normal(size=([711, 28]))
    # motifscore_mat = np.random.normal(size=(1000, 1000, 711))
    # targets_mat = np.random.normal(size=(28, 1000, 1000, 8))
    evaluate(params, model, test_cell_idx, seq_mat, expr_mat, motifscore_mat, targets_mat, batch_size = 4)
    


def main(argv):
    del argv
    params = {}
    params["num_channels"] = FLAGS.num_channels
    params["num_cb"] = FLAGS.num_cb
    params["seq_length"] = FLAGS.seq_length
    params["hidden_size"] = FLAGS.hidden_size
    params["num_hidden_layers"] = FLAGS.num_hidden_layers
    params["num_heads"] = FLAGS.num_heads
    params["num_targets"] = FLAGS.num_targets
    params["attention_dropout"] = FLAGS.attention_dropout
    params["filter_size"] = FLAGS.filter_size
    params["relu_dropout"] = FLAGS.relu_dropout
    params["layer_postprocess_dropout"] = FLAGS.layer_postprocess_dropout

    params["bs"] = FLAGS.bs
    params["lr"] = FLAGS.lr
    params["epochs"] = FLAGS.epochs
    params["dtype"] = DTYPE_MAP[FLAGS.dtype]
    params["tr_frac"] = FLAGS.tr_frac
    params["val_frac"] = FLAGS.val_frac
    print(params)
    train(params)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    #5,4,8,0.1,512
    flags.DEFINE_integer("num_channels", 313, "Output channels number for convolution stack")
    flags.DEFINE_integer("num_cb", 7, "Number of convolution blocks")
    flags.DEFINE_integer("seq_length", 1000, "number of words in a sequence")
    flags.DEFINE_integer("hidden_size", 1024, "word embedding dimension")#2048
    flags.DEFINE_integer("num_hidden_layers", 3, "number of MultiHeadAttention") #10
    flags.DEFINE_integer("num_heads", 2, "number of heads in each MHA") #8
    flags.DEFINE_integer("num_targets", 8, "number of targets in multi-task prediction")
    flags.DEFINE_float("attention_dropout", 0.1, "attention dropout rate")
    flags.DEFINE_integer("filter_size", 128, "filter size for the inner (first) dense layer in ffn")#512
    flags.DEFINE_float("relu_dropout", 0.1, "ffn dropout rate")
    flags.DEFINE_float("layer_postprocess_dropout", 0.1, "encoder inputs dropout rate")
    flags.DEFINE_float("lr", 1e-4, "learning rate")
    flags.DEFINE_integer("bs", 4, "batch size")
    flags.DEFINE_integer("epochs", 100, "epochs for training")
    flags.DEFINE_string("dtype", "fp32", "Data type for tensor")
    flags.DEFINE_float("tr_frac", 0.75, "fraction of training cell types/tissues")
    flags.DEFINE_float("val_frac", 0.01, "fraction of validation set")
    app.run(main) 
