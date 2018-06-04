import time
from keras.callbacks import Callback


class LoggingCallback(Callback):
    """
        Callback that logs learning progress at end of epoch
    """

    def __init__(self, print_fn=print):
        super().__init__()
        self.print_fn = print_fn
        self.batch_start_time = 0

    def on_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        elapsed_time = time.time() - self.batch_start_time
        msg = "batch: {0:6}, elapsed_time: {1:10.4}".format(batch, elapsed_time)
        prepared = [(k, logs[k]) for k in sorted(logs.keys()) if k not in ('batch',
                                                                           'size')]
        for k, v in prepared:
            msg += ", {0}: {1:8.6f}".format(k, v)

        self.print_fn(msg)

    def on_epoch_end(self, epoch, logs=None):
        msg = "epoch: {:6}".format(epoch + 1)
        prepared = [(k, logs[k]) for k in sorted(logs.keys())]
        for k, v in prepared:
            msg += ", {}: {:8.6f}".format(k, v)

        self.print_fn(msg)
