import tensorflow as tf
import numpy as np 
import random
import sys, os 
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
        self.model.save_weights('checkpoints/model_epoch%d.h5' % epoch)
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

def generate_batch(params, batch_size, data_queue, seq_mat, expr_mat, 
                    motifscore_mat, targets_mat):
    batch_seq = np.empty((batch_size, 1000*2**params["num_cb"], 1, 4))
    #after conv shape [batch_size, 1000, 2048-711]
    batch_expr = np.empty((batch_size, 1000, params["hidden_size"]-params["num_channels"]))
    batch_motifscore = np.empty((batch_size, 1000, params["hidden_size"]-params["num_channels"]))
    batch_targets = np.empty((batch_size, 1000, params["num_targets"]))
    while True:
        for step in range(len(data_queue) // batch_size):
            present_queue = data_queue[step*batch_size : (step+1)*batch_size]
            for i in range(len(present_queue)):
                cell_idx, seq_idx = present_queue[i]
                batch_seq[i] = seq_mat[seq_idx]
                batch_expr[i] = np.tile(expr_mat[:, cell_idx], (1000, 1))
                batch_motifscore[i] = motifscore_mat[seq_idx]
                batch_targets[i] = targets_mat[cell_idx, seq_idx, :, :]
            yield [batch_seq, batch_expr * batch_motifscore], batch_targets
        

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
    seq_mat = np.empty((len(dna_seqs), 1000*2**params["num_cb"], 1, 4))
    for i in range(len(dna_seqs)):
        line = dna_seqs[i]
        chrom,start,end = line.split('\t')[0], int(line.split('\t')[1]), int(line.split('\t')[2].strip())
        seq = genome[chrom][start : end]
        seq_mat[i, :, 0, :] = one_hot_encoding(seq)
        
    expr_mat = pd.read_csv(expr_file, header=0, index_col=[0], sep='\t').values
    expr_mat = np.log(expr_mat + 1)

    motifscore_mat = np.load(motifscore_file).reshape(len(dna_seqs), 1000, expr_mat.shape[0])
    targets_mat = np.load(targets_file)
    for i in range(params["num_targets"]):
        N = np.max(targets_mat[:,:,i])
        N_i = np.max(targets_mat[:,:,i], axis = 1)
        for j in range(targets_mat.shape[0]):
            targets_mat[j,:,i] = np.log(1. + targets_mat[j,:,i]*N/N_i[j])
    targets_mat = targets_mat.reshape((expr_mat.shape[1], len(dna_seqs), 1000, params["num_targets"]))
    #raw readscount, normalize, log(1+xij*N/Ni)
    return seq_mat, expr_mat, motifscore_mat, targets_mat

def evaluate(model, test_cell_idx, seq_mat, expr_mat, motifscore_mat, targets_mat):
    from scipy.stats import pearsonr
    corr_all = np.empty((len(test_cell_idx), targets_mat.shape[-1]))
    for cell_idx in test_cell_idx:
        print(cell_idx)
        seq_test_mat = seq_mat
        expr_test_mat = np.tile(expr_mat[:, cell_idx], (motifscore_mat.shape[0], 1000, 1))
        motifscore_test_mat = motifscore_mat
        targets_test_mat = targets_mat[cell_idx, :, :, :]
        pre = model.predict([seq_test_mat, expr_test_mat*motifscore_test_mat], batch_size = 4)
        #pre with shape [13300, 1000, 8]
        for i in range(targets_test_mat.shape[-1]):
            corr_all[list(test_cell_idx).index(cell_idx)][i] = pearsonr(targets_test_mat[:,:,i].flatten(), pre[:,:,i].flatten())[0]
    print(np.mean(corr_all,axis = 0), np.mean(corr_all,axis = 1))
    np.save('corr.npy', corr_all)
    

def train(params):
    model = Geformer(params)
    optimizer = tf.keras.optimizers.Adam(learning_rate = FLAGS.lr)
    model.compile(optimizer, loss = tf.keras.losses.MeanSquaredError())
    seq_mat, expr_mat, motifscore_mat, targets_mat = load_inputs(params)
    #seq_mat = np.random.normal(size=(20, 128000, 1, 4))
    #expr_mat = np.random.normal(size=([711, 28]))
    #motifscore_mat = np.random.normal(size=(20, 1000, 711))
    #targets_mat = np.random.normal(size=(28, 20, 1000, 8))
    print('load data done')
    print(seq_mat.shape, expr_mat.shape, motifscore_mat.shape, targets_mat.shape)
    np.random.seed(0)
    train_cell_idx = np.random.choice(targets_mat.shape[0], int(0.75*targets_mat.shape[0]), replace = False)
    test_cell_idx = [item for item in np.arange(targets_mat.shape[0]) if item not in train_cell_idx]
    data_queue = []
    for i in train_cell_idx:
        for j in range(seq_mat.shape[0]):
            data_queue.append([i, j])
    random.seed(0)
    random.shuffle(data_queue)
    train_queue = data_queue[:int(0.99*len(data_queue))]
    valid_queue = data_queue[int(0.99*len(data_queue)):]
    print(len(train_queue),len(valid_queue))
    train_generator = generate_batch(params, params["bs"], train_queue, 
                    seq_mat, expr_mat, motifscore_mat, targets_mat)
    validation_generator = generate_batch(params, len(valid_queue), valid_queue, 
                    seq_mat, expr_mat, motifscore_mat, targets_mat)
    validation_data = next(validation_generator)
    # seq_test = np.random.normal(size = (None, 1000*2**params["num_cb"], 1, 4))
    # expr_test = np.random.normal(size = (None, 1000, 711))
    # motifscore_test = np.random.normal(size = (None, 1000, params["hidden_size"]-params["num_channels"]))
    # target_test = np.random.normal(size = (None, 1000, 1000))
    print(validation_data[0][0].shape,validation_data[0][1].shape,validation_data[1].shape)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
                EarlyStoppingAtMinLoss(patience=2)]
    history = model.fit(x = train_generator, steps_per_epoch = len(train_queue)//params["bs"], 
                epochs = 100, verbose = 1, batch_size = params["bs"], callbacks = callbacks,
                validation_data = validation_data,
                max_queue_size = 100, workers = 28,use_multiprocessing = True)
                #)
    #model.load_weights('model_epoch2.h5')
    print('Model training done')
    model.save_weights("checkpoints/model.h5")
    evaluate(model, test_cell_idx, seq_mat, expr_mat, motifscore_mat, targets_mat)
    


def main(argv):
    del argv
    params = {}
    params["num_channels"] = FLAGS.num_channels
    params["num_cb"] = FLAGS.num_cb
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
    print(params)
    train(params)

if __name__ == '__main__':
    FLAGS = flags.FLAGS

    flags.DEFINE_integer("num_channels", 313, "Output channels number for convolution stack")
    flags.DEFINE_integer("num_cb", 7, "Number of convolution blocks")
    flags.DEFINE_integer("hidden_size", 1024, "word embedding dimension")#2048
    flags.DEFINE_integer("num_hidden_layers", 10, "number of MultiHeadAttention") #10
    flags.DEFINE_integer("num_heads", 8, "number of heads in each MHA") #8
    flags.DEFINE_integer("num_targets", 8, "number of targets in multi-task prediction")
    flags.DEFINE_float("attention_dropout", 0.1, "attention dropout rate")
    flags.DEFINE_integer("filter_size", 512, "filter size for the inner (first) dense layer in ffn")#512
    flags.DEFINE_float("relu_dropout", 0.1, "ffn dropout rate")
    flags.DEFINE_float("layer_postprocess_dropout", 0.1, "encoder inputs dropout rate")
    flags.DEFINE_float("lr", 1e-4, "learning rate")
    flags.DEFINE_integer("bs", 4, "batch size")
    flags.DEFINE_integer("epochs", 100, "epochs for training")
    flags.DEFINE_string("dtype", "fp32", "Data type for tensor")
    app.run(main) 
