import os
# Set this before importing TensorFlow to suppress CUDA/XLA INFO logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO off, 2=INFO+WARNING off, 3=ERROR only
import json
import time
import logging
import tensorflow as tf
import numpy as np
import random
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import layers, Model

# Suppress remaining TensorFlow logs
tf.get_logger().setLevel(logging.ERROR)

# Use GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

FEATURE_NAMES = [
    'std_rush_order', 'avg_rush_order', 'std_trades', 'std_volume', 'avg_volume',
    'std_price', 'avg_price', 'avg_price_max', 'hour_sin', 'hour_cos',
    'minute_sin', 'minute_cos', 'delta_minutes'
]

class ConvLSTM(Model):
    def __init__(self, num_feats, conv_kernel_size, embedding_size, num_layers,
                 dropout=0.0, out_norm=False, cell_norm=False):
        super(ConvLSTM, self).__init__()
        self.num_feats = num_feats
        self.conv_kernel_size = conv_kernel_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_norm = out_norm
        self.cell_norm = cell_norm

        self.conv1 = layers.Conv1D(filters=embedding_size, kernel_size=conv_kernel_size, padding='valid')
        self.relu1 = layers.ReLU()
        self.pool = layers.MaxPool1D(pool_size=2, strides=1, padding='valid')
        self.lstm = layers.LSTM(embedding_size, return_sequences=True,
                               dropout=dropout if not cell_norm else 0.0, recurrent_dropout=0.0)
        self.ln = layers.LayerNormalization() if out_norm else None
        self.cell_ln = layers.LayerNormalization() if cell_norm else None
        self.o_proj = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        y = inputs
        pad_size = self.conv_kernel_size - 1
        
        # Left padding with replication
        left_pad = tf.repeat(y[:, :1, :], pad_size, axis=1)
        y = tf.concat([left_pad, y], axis=1)
        
        y = self.conv1(y)
        y = self.relu1(y)
        
        # Right padding to maintain seq_len after pooling
        right_pad = tf.repeat(y[:, -1:, :], 1, axis=1)
        y = tf.concat([y, right_pad], axis=1)
        y = self.pool(y)
        
        if self.cell_norm:
            y = self.cell_ln(y)
        y = self.lstm(y, training=training)
        if self.out_norm:
            y = self.ln(y)
        y = self.o_proj(y)
        return y

def create_loader(data, batch_size, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_data(path):
    return pd.read_csv(path, compression='gzip', parse_dates=['date'])

PLACEHOLDER_TIMEDELTA = pd.Timedelta(minutes=0)
MIN_PUMP_SIZE = 100

def get_pumps(data, segment_length, pad=True):
    pumps = []
    skipped_row_count = 0
    for pump_index in np.unique(data['pump_index'].values):
        pump_i = data[data['pump_index'] == pump_index].copy()
        if len(pump_i) < MIN_PUMP_SIZE:
            print(f'Pump {pump_index} has {len(pump_i)} rows, skipping')
            skipped_row_count += len(pump_i)
            continue
        pump_i['delta_minutes'] = (pump_i['date'] - pump_i['date'].shift(1)).fillna(PLACEHOLDER_TIMEDELTA)
        pump_i['delta_minutes'] = pump_i['delta_minutes'].apply(lambda x: x.total_seconds() / 60)
        pump_i = pump_i[FEATURE_NAMES + ['gt']]
        pump_i = pump_i.values.astype(np.float32)
        if pad:
            pump_i = np.pad(pump_i, ((segment_length - 1, 0), (0, 0)), mode='reflect')
        pumps.append(pump_i)
    print(f'Skipped {skipped_row_count} rows total')
    print(f'{len(pumps)} pumps')
    return pumps

def process_data(data, segment_length=15):
    print('Processing data...')
    print(f'Segment length: {segment_length}')
    pumps = get_pumps(data, segment_length)
    segments = []
    for pump in pumps:
        for i, window in enumerate(np.lib.stride_tricks.sliding_window_view(pump, segment_length, axis=0)):
            segment = window.transpose()
            segments.append(segment)
    print(f'{len(segments)} rows of data after processing')
    return np.stack(segments)

def undersample_data(data, undersample_ratio):
    with_anomalies = data[:, :, -1].sum(axis=1) > 0
    mask = with_anomalies | (np.random.rand(data.shape[0]) < undersample_ratio)
    return data[mask]

def get_data(path, batch_size, train_ratio, undersample_ratio, segment_length):
    cached_data_path = f'{path}_{segment_length}.npy'
    if not os.path.exists(cached_data_path):
        data = process_data(load_data(path), segment_length=segment_length)
        np.save(cached_data_path, data)
    else:
        print(f'Loading cached data from {cached_data_path}')
        data = np.load(cached_data_path)
    
    if train_ratio is not None:
        train_data, test_data = train_test_split(data, train_size=train_ratio, shuffle=False)
        train_data = undersample_data(train_data, undersample_ratio)
        return train_data, test_data
    return data

def train(model, dataloader, optimizer, criterion, feature_count=13):
    epoch_loss = 0
    num_batches = 0
    iterator = iter(dataloader)
    while True:
        try:
            batch = next(iterator)
            x = batch[:, :, :feature_count]
            y = batch[:, :, -1]  # Full sequence target
            with tf.GradientTape() as tape:
                preds = model(x, training=True)[:, :, 0]
                loss = criterion(y, preds)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()
            num_batches += 1
        except StopIteration:
            break  # Exit when dataset is exhausted
    return epoch_loss / num_batches if num_batches > 0 else 0

def validate(model, dataloader, verbose=True, pr_threshold=0.7, feature_count=13):
    preds_1 = []
    preds_0 = []
    all_ys = []
    all_preds = []
    iterator = iter(dataloader)
    while True:
        try:
            batch = next(iterator)
            x = batch[:, :, :feature_count]
            y = batch[:, -1, -1]
            preds = model(x, training=False)[:, -1, 0]
            if verbose:
                preds_0.extend(preds[y == 0].numpy())
                preds_1.extend(preds[y == 1].numpy())
            all_ys.append(y.numpy())
            all_preds.append(preds.numpy())
        except StopIteration:
            break
    if verbose and preds_0 and preds_1:
        print(f'Mean output at 0: {np.mean(preds_0):0.5f} at 1: {np.mean(preds_1):0.5f}')
    y = np.concatenate(all_ys, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    preds = preds >= pr_threshold
    acc = accuracy_score(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    return acc, precision, recall, f1

def pick_threshold(model, dataloader, undersample_ratio, feature_count=13):
    all_ys = []
    all_preds = []
    iterator = iter(dataloader)
    while True:
        try:
            batch = next(iterator)
            x = batch[:, :, :feature_count]
            y = batch[:, -1, -1]
            preds = model(x, training=False)[:, -1, 0]
            all_ys.append(y.numpy())
            all_preds.append(preds.numpy())
        except StopIteration:
            break
    y = np.concatenate(all_ys, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    precision, recall, thresholds = precision_recall_curve(y, preds)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f'Best threshold: {best_threshold} (train f1: {np.max(f1_scores):0.5f})')
    return best_threshold

def collect_metrics_n_epochs(model, train_loader, test_loader, optimizer, criterion, config, feature_count=13):
    best_metrics = np.array([0.0]*4)
    for epoch in range(config.n_epochs):
        start = time.time()
        loss = train(model, train_loader, optimizer, criterion, feature_count)
        if (epoch + 1) % config.train_output_every_n == 0:
            print(f'Epoch {epoch + 1}{f" ({(time.time()-start):0.2f}s)" if config.time_epochs else ""} -- Train Loss: {loss:0.5f}')
        if (epoch + 1) % config.validate_every_n == 0 or config.final_run:
            if config.prthreshold > 0:
                prthreshold = config.prthreshold
            else:
                prthreshold = pick_threshold(model, train_loader, config.undersample_ratio, feature_count)
            acc, precision, recall, f1 = validate(model, test_loader, verbose=config.verbose, pr_threshold=prthreshold, feature_count=feature_count)
            if f1 > best_metrics[-1]:
                best_metrics = [acc, precision, recall, f1]
            print(f'Val   -- Acc: {acc:0.5f} -- Precision: {precision:0.5f} -- Recall: {recall:0.5f} -- F1: {f1:0.5f}')
    return best_metrics

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        self.train_output_every_n = 5
        self.validate_every_n = 10
        self.time_epochs = True
        self.final_run = False
        self.verbose = False
        self.seed = 0

def main():
    with open('conv_lstm_config.json', 'r') as f:
        configs = json.load(f)
    config_name = "15S_optimal_5Folds"
    config = Config(configs[config_name])
    
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    data_path = os.path.join('data', os.path.basename(config.dataset))
    train_data, test_data = get_data(data_path, config.batch_size, 0.8, config.undersample_ratio, config.segment_length)
    feature_count = train_data.shape[-1] - 1
    
    train_loader = create_loader(train_data, config.batch_size, shuffle=True)
    test_loader = create_loader(test_data, config.batch_size)
    
    model = ConvLSTM(
        num_feats=feature_count,
        conv_kernel_size=config.kernel_size,
        embedding_size=config.embedding_size,
        num_layers=config.n_layers,
        dropout=config.dropout,
        out_norm=config.out_norm,
        cell_norm=config.cell_norm
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
    criterion = tf.keras.losses.BinaryCrossentropy()
    
    fold_metrics = np.array([0.0]*4)
    if config.kfolds > 1:
        kf = KFold(n_splits=config.kfolds, shuffle=True, random_state=config.seed)
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(train_data)):
            print(f'##### Fold {fold_i+1} #####')
            fold_train_data, fold_test_data = train_data[train_idx], train_data[test_idx]
            fold_train_loader = create_loader(fold_train_data, config.batch_size, shuffle=True)
            fold_test_loader = create_loader(fold_test_data, config.batch_size)
            fold_model = ConvLSTM(
                num_feats=feature_count,
                conv_kernel_size=config.kernel_size,
                embedding_size=config.embedding_size,
                num_layers=config.n_layers,
                dropout=config.dropout,
                out_norm=config.out_norm,
                cell_norm=config.cell_norm
            )
            fold_optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
            best_metrics = collect_metrics_n_epochs(
                fold_model, fold_train_loader, fold_test_loader, fold_optimizer, criterion, config, feature_count
            )
            fold_metrics += best_metrics
            print(f'Best F1 for this fold: {best_metrics[-1]:0.5f}\n')
    else:
        best_metrics = collect_metrics_n_epochs(
            model, train_loader, test_loader, optimizer, criterion, config, feature_count
        )
        fold_metrics += best_metrics
        print(f'Best F1 this run: {best_metrics[-1]:0.5f}\n')
    
    acc, precision, recall, f1 = fold_metrics / config.kfolds
    print(f'Final metrics ({config.kfolds} folds)')
    print(f'Val   -- Acc: {acc:0.5f} -- Precision: {precision:0.5f} -- Recall: {recall:0.5f} -- F1: {f1:0.5f}')

if __name__ == '__main__':
    main()
