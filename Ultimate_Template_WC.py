# ==========================================
# 0. IMPORTS & GPU SETUP
# ==========================================
import os, gc, warnings, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, applications, mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        CSVLogger, LambdaCallback)
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.utils import class_weight
from sklearn.metrics import (classification_report, confusion_matrix, log_loss,
                             roc_auc_score, f1_score, cohen_kappa_score,
                             average_precision_score)

warnings.filterwarnings('ignore')
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
mixed_precision.set_global_policy('mixed_float16')


# ==========================================
# 1. CFG — শুধু এই 4 BLOCK UPDATE করো
# ==========================================
class CFG:
    seed = 42

    num_classes = 4          
    img_size    = (224, 224)
    batch_size  = 32  

    data_format = 'csv'
    task_type = 'multiclass'

    monitor_metric = 'val_accuracy'
    monitor_mode   = 'max'   

    sub_type = 'probability'

    h_flip = True
━
    train_dir = '/kaggle/input/dog-breed-identification/train/'
    test_dir  = '/kaggle/input/dog-breed-identification/test/'
    data_csv  = '/kaggle/input/dog-breed-identification/labels.csv'
    test_csv  = '/kaggle/input/dog-breed-identification/sample_submission.csv'
    img_col   = 'id'       
    label_col = 'breed'

    val_split = 0.2

    group_col = None

    add_class_dir = False
    multilabel_cols = None

    epochs_warmup    = 5
    epochs_finetune  = 25
    label_smoothing  = 0.1
    warmup_epochs_lr = 3
    use_mixup        = True
    mixup_alpha      = 0.2
    use_tta          = True
    n_tta            = 6
    use_concat_pool  = True 
    dense_units      = 512
    dropout_rate_1   = 0.4
    dropout_rate_2   = 0.3

tf.keras.utils.set_random_seed(CFG.seed)
np.random.seed(CFG.seed)

if CFG.task_type == 'multiclass':
    ACTIVATION = 'softmax'
    LOSS_BASE  = tf.keras.losses.CategoricalCrossentropy(
                     label_smoothing=CFG.label_smoothing)
    CLASS_MODE = 'categorical'
else:
    ACTIVATION = 'sigmoid'
    LOSS_BASE  = tf.keras.losses.BinaryCrossentropy(
                     label_smoothing=CFG.label_smoothing)
    CLASS_MODE = 'raw' 

print(f"Mode:{CFG.task_type} | Act:{ACTIVATION} | ClassMode:{CLASS_MODE}")
print(f"Monitoring: {CFG.monitor_metric} ({CFG.monitor_mode})")


# ==========================================
# 2. DATA LOADING
# ==========================================
print(f"\n{'='*50}\nDATA LOADING ({CFG.data_format.upper()})\n{'='*50}")

class_indices        = None
index_to_class       = None
class_weights_dict   = None
FILE_COL = Y_COL     = None
train_df = val_df = df_full = None
cw_arr_list          = None
_orig_label_col_list = None 

if CFG.data_format == 'csv':
    df_full = pd.read_csv(CFG.data_csv)

    if CFG.task_type == 'multiclass' and isinstance(CFG.label_col, list):
        print(f"  [Auto-Convert] One-Hot {CFG.label_col} → Single Label")
        df_full['__auto_label__'] = df_full[CFG.label_col].idxmax(axis=1)
        _orig_label_col_list = CFG.label_col
        CFG.label_col        = '__auto_label__'

    if CFG.add_class_dir:
        df_full['filename'] = (df_full[CFG.label_col].astype(str)
                               + '/' + df_full[CFG.img_col].astype(str))
        FILE_COL = 'filename'
    elif not df_full[CFG.img_col].astype(str).str.contains(
            r'\.\w+$', regex=True).any():
        df_full['filename'] = df_full[CFG.img_col].astype(str) + '.jpg'
        FILE_COL = 'filename'
    else:
        FILE_COL = CFG.img_col

    if CFG.task_type == 'multiclass':
        df_full[CFG.label_col] = df_full[CFG.label_col].astype(str)
        Y_COL = CFG.label_col
    else:
        if CFG.multilabel_cols is None:
            raise ValueError(
                " task_type='multilabel' হলে CFG.multilabel_cols list দিতে হবে!\n"
                "  Example: multilabel_cols = ['label1', 'label2', 'label3']")
        Y_COL = CFG.multilabel_cols
        print(f"  Multilabel columns: {Y_COL}")

    if CFG.group_col and CFG.group_col in df_full.columns:
        print(f"GroupShuffleSplit → '{CFG.group_col}' (data leakage রোখা হচ্ছে)")
        gss = GroupShuffleSplit(n_splits=1, test_size=CFG.val_split,
                                random_state=CFG.seed)
        tr_idx, va_idx = next(gss.split(df_full, groups=df_full[CFG.group_col]))
        train_df = df_full.iloc[tr_idx].reset_index(drop=True)
        val_df   = df_full.iloc[va_idx].reset_index(drop=True)
    else:
        print("✅ Standard Stratified Split")
        stratify = df_full[CFG.label_col] if CFG.task_type == 'multiclass' else None
        train_df, val_df = train_test_split(
            df_full, test_size=CFG.val_split,
            random_state=CFG.seed, stratify=stratify)

    CFG.num_classes = train_df[CFG.label_col].nunique()
    print(f"✅ Auto-detected Classes: {CFG.num_classes}")
    print(f"   Train:{len(train_df)} | Val:{len(val_df)}")

    if CFG.task_type == 'multiclass':
        cw_arr      = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(train_df[CFG.label_col]),
            y=train_df[CFG.label_col])
        cw_arr_list = list(cw_arr)

elif CFG.data_format == 'directory':
    FILE_COL = None
    Y_COL    = None
    print(f"  Train dir: {CFG.train_dir}")


# ==========================================
# 3. DATA GENERATORS
# ==========================================

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.10,
    zoom_range=0.15,
    horizontal_flip=CFG.h_flip,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator()


def make_gen(datagen, df, shuffle=True, directory=None):
    if CFG.data_format == 'csv':
        return datagen.flow_from_dataframe(
            dataframe=df,
            directory=directory or CFG.train_dir,
            x_col=FILE_COL,
            y_col=Y_COL,
            target_size=CFG.img_size,
            batch_size=CFG.batch_size,
            class_mode=CLASS_MODE,
            shuffle=shuffle,
            seed=CFG.seed
        )
    else:
        aug = ImageDataGenerator(
            rotation_range=15, width_shift_range=0.10,
            height_shift_range=0.10, shear_range=0.10,
            zoom_range=0.15, horizontal_flip=CFG.h_flip,
            brightness_range=[0.85, 1.15], fill_mode='nearest',
            validation_split=CFG.val_split)
        val_aug = ImageDataGenerator(validation_split=CFG.val_split)
        if shuffle:
            return aug.flow_from_directory(
                CFG.train_dir, target_size=CFG.img_size,
                batch_size=CFG.batch_size, class_mode=CLASS_MODE,
                shuffle=True, seed=CFG.seed, subset='training')
        else:
            return val_aug.flow_from_directory(
                CFG.train_dir, target_size=CFG.img_size,
                batch_size=CFG.batch_size, class_mode=CLASS_MODE,
                shuffle=False, subset='validation')


train_gen = make_gen(train_datagen, train_df, shuffle=True)
val_gen   = make_gen(val_datagen,   val_df,   shuffle=False)

if CFG.task_type == 'multiclass':
    class_indices  = train_gen.class_indices
    index_to_class = {v: k for k, v in class_indices.items()}

    if CFG.data_format == 'csv':
        class_weights_dict = {
            class_indices[lbl]: cw_arr_list[i]
            for i, lbl in enumerate(np.unique(train_df[CFG.label_col]))
        }
    else:
        print("  Computing class weights from directory...")
        dir_labels = train_gen.classes
        cw_dir     = class_weight.compute_class_weight(
            'balanced', classes=np.unique(dir_labels), y=dir_labels)
        class_weights_dict = {i: float(w) for i, w in enumerate(cw_dir)}
        print(f"  ✅ Class weights: {class_weights_dict}")

if df_full is not None and CFG.task_type == 'multiclass':
    counts = df_full[CFG.label_col].value_counts()
    plt.figure(figsize=(20, 4))
    sns.barplot(x=counts.index[:40].astype(str),
                y=counts.values[:40], palette='viridis')
    plt.xticks(rotation=90, fontsize=6)
    plt.title(f'Class Distribution (top 40 of {len(counts)})')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150)
    plt.show()
    ratio = counts.values[0] / max(counts.values[-1], 1)
    print(f"  Most : {counts.index[0]} ({counts.values[0]})")
    print(f"  Least: {counts.index[-1]} ({counts.values[-1]})")
    print(f"  Imbalance: {ratio:.1f}x {'⚠️ HIGH' if ratio > 5 else '✅ OK'}")


# ==========================================
# 4. MIXUP — IN-BATCH (Zero Data Waste)
# ==========================================
def mixup_generator(generator, alpha=CFG.mixup_alpha):
    while True:
        x, y = next(generator)
        bs = len(x)
        if bs <= 1:
            yield x, y
            continue
        y   = y.astype(np.float32)
        idx = np.random.permutation(bs)
        x2, y2 = x[idx], y[idx]
        lam   = np.random.beta(alpha, alpha)
        lam_x = np.reshape(lam, [-1] + [1] * (len(x.shape) - 1))
        lam_y = np.reshape(lam, [-1] + [1] * (len(y.shape) - 1))
        yield lam_x * x + (1 - lam_x) * x2, lam_y * y + (1 - lam_y) * y2

def apply_sample_weights(generator, weights_dict):
    for x, y in generator:
        class_ids = np.argmax(y, axis=1)
        sw = np.array(
            [weights_dict.get(int(c), 1.0) for c in class_ids],
            dtype='float32')
        yield x, y, sw


# ==========================================
# 5. COSINE LR WITH WARMUP — SAFE VERSION
# ==========================================
@tf.keras.utils.register_keras_serializable()
class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr      = float(base_lr)
        self.total_steps  = float(total_steps)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step        = tf.cast(step, tf.float32)
        warmup_lr   = self.base_lr * (step / tf.maximum(1.0, self.warmup_steps))
        decay_steps = tf.maximum(1.0, self.total_steps - self.warmup_steps)
        progress    = (step - self.warmup_steps) / decay_steps
        cosine_lr   = 0.5 * self.base_lr * (
            1.0 + tf.cos(math.pi * tf.clip_by_value(progress, 0.0, 1.0)))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            'base_lr':      self.base_lr,
            'total_steps':  self.total_steps,
            'warmup_steps': self.warmup_steps,
        }


# ==========================================
# 6. MODEL BUILDER
# ==========================================
def build_model(base_fn):
    base = base_fn(weights='imagenet', include_top=False,
                   input_shape=(*CFG.img_size, 3))
    base.trainable = False

    inputs = tf.keras.Input(shape=(*CFG.img_size, 3))
    name   = base_fn.__name__

    if 'EfficientNet' in name:
        x = inputs
    elif 'ResNet' in name or 'MobileNet' in name:
        x = layers.Rescaling(1. / 127.5, offset=-1.0)(inputs)
    elif 'DenseNet' in name:
        x    = layers.Rescaling(1. / 255.)(inputs)
        mean = tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 3])
        std  = tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 3])
        x    = (x - mean) / std
    elif 'Xception' in name or 'Inception' in name:
        x = layers.Rescaling(1. / 127.5, offset=-1.0)(inputs)
    else:
        x = layers.Rescaling(1. / 255.)(inputs)

    x = base(x, training=False)

    if CFG.use_concat_pool:
        gap = layers.GlobalAveragePooling2D()(x)
        gmp = layers.GlobalMaxPooling2D()(x)
        x   = layers.Concatenate()([gap, gmp])
    else:
        x = layers.GlobalAveragePooling2D()(x)

    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(CFG.dropout_rate_1)(x)
    x       = layers.Dense(CFG.dense_units, activation='swish')(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(CFG.dropout_rate_2)(x)
    outputs = layers.Dense(CFG.num_classes, activation=ACTIVATION,
                           dtype='float32')(x)

    return tf.keras.Model(inputs, outputs), base


# ==========================================
# 7. METRICS — ALL-IN-ONE UNIVERSAL
# ==========================================
def get_metrics():
 
    m = [
        'accuracy',
        tf.keras.metrics.AUC(multi_label=True, name='auc'),
        tf.keras.metrics.AUC(curve='PR', multi_label=True, name='map'),
        tf.keras.metrics.Precision(name='pre'),
        tf.keras.metrics.Recall(name='rec'),
    ]

    try:
        m.append(tf.keras.metrics.F1Score(average='macro', name='f1'))
    except AttributeError:
        print("  F1Score unavailable (TF < 2.16) — skipped")

    # Top-5: num_classes > 5 হলে যোগ করো
    if CFG.task_type == 'multiclass' and CFG.num_classes > 5:
        m.append(tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc'))

    return m


# ==========================================
# 8. TRAINING PIPELINE
# ==========================================
def run_pipeline(model_func, model_name):
    print(f"\n{'='*50}\nTRAINING: {model_name}\n{'='*50}")
    tf.keras.backend.clear_session()
    gc.collect()

    model, base = build_model(model_func)
    spe = len(train_gen) 
    # spe = min(500, len(train_gen))

    monitor_metric = getattr(CFG, 'monitor_metric', 'val_accuracy')
    monitor_mode   = getattr(CFG, 'monitor_mode',   'max')

    print(f"\n[Phase 1] Warm-up — {CFG.epochs_warmup} epochs")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=LOSS_BASE,
                  metrics=get_metrics())

    gen_p1_raw = (mixup_generator(make_gen(train_datagen, train_df, shuffle=True))
                  if CFG.use_mixup else train_gen)
    gen_p1     = (apply_sample_weights(gen_p1_raw, class_weights_dict)
                  if (CFG.task_type == 'multiclass' and class_weights_dict)
                  else gen_p1_raw)

    model.fit(gen_p1,
              steps_per_epoch=spe,
              validation_data=val_gen,
              epochs=CFG.epochs_warmup,
              verbose=1) 

    print(f"\n[Phase 2] Fine-tune — {CFG.epochs_finetune} epochs")
    base.trainable = True

    lr_sched = CosineDecayWithWarmup(
        base_lr=1e-5,
        total_steps=spe * CFG.epochs_finetune,
        warmup_steps=spe * CFG.warmup_epochs_lr)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr_sched),
                  loss=LOSS_BASE,
                  metrics=get_metrics())

    save_path = f'best_{model_name}.keras'

    def _safe_lr(epoch, logs):
        try:
            opt    = model.optimizer
            lr_val = (float(opt.learning_rate(opt.iterations))
                      if hasattr(opt.learning_rate, '__call__')
                      else float(opt.learning_rate))
            print(f"  LR: {lr_val:.2e}")
        except Exception:
            pass

    callbacks = [
        ModelCheckpoint(save_path,
                        monitor=monitor_metric,
                        save_best_only=True,
                        mode=monitor_mode,
                        verbose=1),
        EarlyStopping(monitor=monitor_metric,
                      patience=6,
                      restore_best_weights=True,
                      verbose=1),
        CSVLogger(f'{model_name}_log.csv'),
        LambdaCallback(on_epoch_end=_safe_lr),
    ]

    gen_p2_raw = (mixup_generator(make_gen(train_datagen, train_df, shuffle=True))
                  if CFG.use_mixup else train_gen)
    gen_p2     = (apply_sample_weights(gen_p2_raw, class_weights_dict)
                  if (CFG.task_type == 'multiclass' and class_weights_dict)
                  else gen_p2_raw)

    history = model.fit(gen_p2,
                        steps_per_epoch=spe,
                        validation_data=val_gen,
                        epochs=CFG.epochs_finetune,
                        callbacks=callbacks,
                        verbose=1) 

    _plot_history(history, model_name)
    _evaluate_model(model, val_df, model_name)

    print(f"\nSaved: {save_path}")
    del model, base
    tf.keras.backend.clear_session()
    gc.collect()
    return save_path


def _plot_history(history, name):
    keys = [k for k in history.history if not k.startswith('val_')]
    n    = max(len(keys), 1)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        ax.plot(history.history[k],                      label='Train')
        ax.plot(history.history.get(f'val_{k}', []),    label='Val')
        ax.set_title(f'{name} — {k}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{name}_curves.png', dpi=150)
    plt.show()


def _evaluate_model(model, val_df_eval, name):
    val_gen_eval = make_gen(val_datagen, val_df_eval, shuffle=False)
    y_pred_prob  = model.predict(val_gen_eval, verbose=0)

    if CFG.task_type == 'multiclass':
        y_pred    = np.argmax(y_pred_prob, axis=1)
        y_true    = val_gen_eval.classes
        y_true_oh = tf.keras.utils.to_categorical(y_true, CFG.num_classes)

        print(f"\n📊 ── Evaluation: {name} ────────────────────────")

        # Log Loss (probability submission-এ গুরুত্বপূর্ণ)
        if CFG.sub_type == 'probability':
            try:
                ll = log_loss(y_true_oh, y_pred_prob)
                print(f"  Log Loss     : {ll:.5f}  (lower = better)")
            except Exception:
                pass

        class_names = [index_to_class[i] for i in range(CFG.num_classes)]
        print(classification_report(y_true, y_pred,
                                    target_names=class_names,
                                    zero_division=0))

        try:
            auc_s = roc_auc_score(y_true_oh, y_pred_prob,
                                  multi_class='ovr', average='macro')
            print(f"  Macro AUC    : {auc_s:.5f}")
        except Exception:
            pass

        try:
            f1_s = f1_score(y_true, y_pred, average='macro')
            print(f"  Macro F1     : {f1_s:.5f}")
        except Exception:
            pass

        try:
            kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
            print(f"  QW Kappa     : {kappa:.5f}")
        except Exception:
            pass

        try:
            map_s = average_precision_score(y_true_oh, y_pred_prob,
                                            average='macro')
            print(f"  mAP          : {map_s:.5f}")
        except Exception:
            pass

        print("  " + "─" * 44)

        n_show = min(20, CFG.num_classes)
        cm     = confusion_matrix(y_true, y_pred)[:n_show, :n_show]
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names[:n_show],
                    yticklabels=class_names[:n_show])
        plt.title(f'Confusion Matrix — {name}')
        plt.tight_layout()
        plt.savefig(f'{name}_cm.png', dpi=150)
        plt.show()

    else:
        if val_df_eval is None or CFG.multilabel_cols is None:
            print("⚠️  Multilabel evaluate skipped (directory mode or no multilabel_cols)")
            return
        y_true = val_df_eval[CFG.multilabel_cols].values.astype(int)
        y_pred = (y_pred_prob > 0.5).astype(int)
        print("\n📊 Multilabel Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=CFG.multilabel_cols,
                                    zero_division=0))


# ==========================================
# 9. MODEL SELECTION
# ==========================================
# def build_model_medical(base_fn):
#     base = base_fn(
#         weights='imagenet21k',    # ← শুধু এটা বদলান
#         include_top=False,
#         input_shape=(*CFG.img_size, 3)
#     )


models_to_train = {
    'EfficientNetV2S': applications.EfficientNetV2S,   
    # 'ResNet50V2':    applications.ResNet50V2,        
    # 'DenseNet121':   applications.DenseNet121,        
    # 'EfficientNetV2M': applications.EfficientNetV2M, 
    # 'Xception':      applications.Xception,          
}


# ==========================================
# 10. RUN TRAINING
# ==========================================
saved_paths = []
for name, fn in models_to_train.items():
    p = run_pipeline(fn, name)
    saved_paths.append(p)
print(f"\n✅ All trained: {saved_paths}")


# ==========================================
# 11. TTA PREDICTION
# ==========================================
def predict_with_tta(model_path, test_files, n_tta=CFG.n_tta):
    print(f"\nPredicting: {Path(model_path).name}")
    model  = models.load_model(model_path, compile=False)
    df_tmp = pd.DataFrame({'filename': test_files})

    def _gen(dg):
        return dg.flow_from_dataframe(
            df_tmp,
            directory=CFG.test_dir,
            x_col='filename',
            y_col=None,
            target_size=CFG.img_size,
            batch_size=CFG.batch_size,
            class_mode=None,
            shuffle=False)

    preds = model.predict(_gen(val_datagen), verbose=1)

    if CFG.use_tta and n_tta > 1:
        tta_dg = ImageDataGenerator(
            horizontal_flip=CFG.h_flip,
            zoom_range=0.10,
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05)
        for i in range(n_tta - 1):
            tta_dg.seed = i
            preds = preds + model.predict(_gen(tta_dg), verbose=0)
        preds = preds / n_tta

    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return preds


# ==========================================
# 12. ENSEMBLE + SUBMISSION
# ==========================================
print(f"\n{'='*50}\nUNIVERSAL ENSEMBLE + SUBMISSION\n{'='*50}")

sample_sub = pd.read_csv(CFG.test_csv)
test_ids   = sample_sub[CFG.img_col].astype(str).values

test_files = ([t + '.jpg' for t in test_ids]
              if not any('.' in t for t in test_ids[:5])
              else list(test_ids))

all_preds = []
for path in saved_paths:
    p = predict_with_tta(path, test_files)
    all_preds.append(p.astype('float16'))
    print(f"  {Path(path).name}: shape={p.shape} | "
          f"conf_mean={p.max(axis=1).mean():.4f}")


ensemble_preds = np.mean(all_preds, axis=0)

target_cols = [c for c in sample_sub.columns if c != CFG.img_col]
out_col     = target_cols[0] if len(target_cols) > 0 else 'label'

if CFG.sub_type == 'class':
    idx_list = np.argmax(ensemble_preds, axis=1)
    
    sample_target_dtype = sample_sub[out_col].dtype
    
    first_val = index_to_class[0]
    is_first_val_numeric = False
    try:
        float(first_val)
        is_first_val_numeric = True
    except (ValueError, TypeError):
        pass
        
    if np.issubdtype(sample_target_dtype, np.number) and not is_first_val_numeric:
        print(f"Detected: Sample submission expects numeric IDs for '{out_col}' but classes are strings.")
        print("   Automatically mapping to raw indices (0 to 100) for TW Food 101 consistency.")
        labels = idx_list
    else:
        labels = [index_to_class[i] for i in idx_list]
        try:
            labels = [int(float(l)) for l in labels]
        except (ValueError, TypeError):
            pass

    submission = pd.DataFrame({CFG.img_col: test_ids, out_col: labels})

elif CFG.sub_type == 'probability':
    pred_df = pd.DataFrame(
        ensemble_preds,
        columns=[index_to_class[i] for i in range(CFG.num_classes)])
    missing_cols = [c for c in target_cols if c not in pred_df.columns]
    if len(missing_cols) == 0:
        pred_df = pred_df[target_cols]
    else:
        print(f"Warning: Target columns in sample submission do not match class names.")
        
    submission = pd.concat(
        [pd.DataFrame({CFG.img_col: test_ids}), pred_df], axis=1)

submission.to_csv('submission.csv', index=False)

print(f"\n submission.csv saved!")
print(f"   Shape     : {submission.shape}  (expected: {sample_sub.shape})")
print(f"   Col match : {list(submission.columns) == list(sample_sub.columns)}")
print(f"   NaN count : {submission.isnull().sum().sum()}")
print(submission.head(3).iloc[:, :min(6, len(submission.columns))])

max_conf = ensemble_preds.max(axis=1)
print(f"\nConfidence: mean={max_conf.mean():.3f} | "
      f">90%: {(max_conf > 0.9).sum()} | <30%: {(max_conf < 0.3).sum()}")

plt.figure(figsize=(8, 4))
plt.hist(max_conf, bins=50, color='steelblue', edgecolor='white')
plt.axvline(0.5, color='red',   ls='--', label='50%')
plt.axvline(0.9, color='green', ls='--', label='90%')
plt.title('Prediction Confidence Distribution')
plt.xlabel('Max Class Probability')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('confidence.png', dpi=150)
plt.show()

print("\n" + "=" * 50)
print("DONE! Pipeline successfully completed. Good luck!")
print("=" * 50)