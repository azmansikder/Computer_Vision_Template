# ==============================================================================
# 🏆 ULTIMATE IMAGE SEGMENTATION TEMPLATE v2.0 — KAGGLE & RESEARCH READY
# Architecture: Multi-Model U-Net (Ensemble) | Loss: BCE + Dice Loss
# Features: tf.data Pipeline | Synchronized Augmentation | TTA | Ensemble
# Metrics: IoU, Dice, Precision, Recall | Output: RLE Submission CSV
# ==============================================================================

import os, gc, cv2, warnings, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LambdaCallback
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
print("✅ TensorFlow Version:", tf.__version__)
print("✅ GPUs Found:", tf.config.list_physical_devices('GPU'))

# Mixed Precision: GPU-তে 1.5-2x স্পিড এবং কম মেমরি
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


# ==========================================
# 1. CFG — শুধু এই ব্লক আপডেট করুন
# ==========================================
class CFG:
    seed = 42
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Dataset & Training Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    img_size    = (256, 256) 
    batch_size  = 16         # OOM হলে 8 বা 4 করুন
    epochs      = 25
    val_split   = 0.2
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Paths (ডেটাসেট ফোল্ডার লিংক)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    images_dir  = '/kaggle/input/medical-segmentation/images/'
    masks_dir   = '/kaggle/input/medical-segmentation/masks/'
    test_dir    = '/kaggle/input/medical-segmentation/test_images/'
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Advanced Options
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    use_tta      = True   # Test Time Augmentation
    use_ensemble = True   # একাধিক মডেলের রেজাল্ট এভারেজ করা

tf.keras.utils.set_random_seed(CFG.seed)


# ==========================================
# 2. DATA LOADING & SPLIT
# ==========================================
print(f"\n{'='*50}\n📂 DATA LOADING & SPLITTING\n{'='*50}")

# ইমেজ এবং মাস্ক পাথ কালেক্ট করা
all_images = sorted([os.path.join(CFG.images_dir, f) for f in os.listdir(CFG.images_dir) if f.endswith(('.png', '.jpg', '.tif'))])
all_masks  = sorted([os.path.join(CFG.masks_dir, f) for f in os.listdir(CFG.masks_dir) if f.endswith(('.png', '.jpg', '.tif'))])

assert len(all_images) == len(all_masks), "❌ Image and Mask count mismatch!"

train_imgs, val_imgs, train_msks, val_msks = train_test_split(
    all_images, all_masks, test_size=CFG.val_split, random_state=CFG.seed
)
print(f"✅ Total: {len(all_images)} | Train: {len(train_imgs)} | Val: {len(val_imgs)}")


# ==========================================
# 3. TF.DATA PIPELINE & SYNCHRONIZED AUGMENTATION
# ==========================================
# Segmentation-এ ইমেজ এবং মাস্ক দুটিকেই একসাথে সমানভাবে Augment করতে হয়!

def load_image_and_mask(img_path, mask_path):
    img = tf.image.decode_image(tf.io.read_file(img_path), channels=3, expand_animations=False)
    img = tf.image.resize(img, CFG.img_size)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.image.decode_image(tf.io.read_file(mask_path), channels=1, expand_animations=False)
    mask = tf.image.resize(mask, CFG.img_size, method='nearest') # মাস্ক ব্লার করা যাবে না
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.where(mask > 0.5, 1.0, 0.0) # Binarize
    
    return img, mask

def augment(img, mask):
    """Synchronized Augmentation"""
    # Random Horizontal Flip
    if tf.random.uniform(()) > 0.5:
        img  = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    
    # Random Vertical Flip
    if tf.random.uniform(()) > 0.5:
        img  = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
        
    # Brightness (শুধুমাত্র ইমেজে, মাস্কে নয়!)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    return img, mask

# Train Dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_msks))
train_ds = train_ds.shuffle(len(train_imgs), seed=CFG.seed)
train_ds = train_ds.map(load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE) # Augmentation
train_ds = train_ds.batch(CFG.batch_size).prefetch(tf.data.AUTOTUNE)

# Validation Dataset (No Augmentation)
val_ds = tf.data.Dataset.from_tensor_slices((val_imgs, val_msks))
val_ds = val_ds.map(load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(CFG.batch_size).prefetch(tf.data.AUTOTUNE)


# ==========================================
# 4. METRICS & LOSSES
# ==========================================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def bce_dice_loss(y_true, y_pred):
    bce  = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = 1.0 - dice_coef(y_true, y_pred)
    return bce + dice

SEG_METRICS = [
    dice_coef, 
    iou_coef, 
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]


# ==========================================
# 5. COSINE LR SCHEDULE
# ==========================================
@tf.keras.utils.register_keras_serializable()
class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = float(base_lr); self.total_steps = float(total_steps); self.warmup_steps = float(warmup_steps)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step / tf.maximum(1.0, self.warmup_steps))
        progress = (step - self.warmup_steps) / tf.maximum(1.0, self.total_steps - self.warmup_steps)
        cosine_lr = 0.5 * self.base_lr * (1.0 + tf.cos(math.pi * tf.clip_by_value(progress, 0.0, 1.0)))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
    def get_config(self): return {'base_lr': self.base_lr, 'total_steps': self.total_steps, 'warmup_steps': self.warmup_steps}


# ==========================================
# 6. MODEL BUILDER (Multi-Architectures)
# ==========================================
def conv_block(inputs, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x

def build_unet(base_filters=64, name="UNet_Standard"):
    """
    base_filters চেঞ্জ করে আলাদা আলাদা মডেল (Ensemble এর জন্য) বানানো যায়।
    """
    inputs = layers.Input(shape=(*CFG.img_size, 3))

    s1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling2D((2, 2))(s1)
    s2 = conv_block(p1, base_filters * 2)
    p2 = layers.MaxPooling2D((2, 2))(s2)
    s3 = conv_block(p2, base_filters * 4)
    p3 = layers.MaxPooling2D((2, 2))(s3)

    b1 = conv_block(p3, base_filters * 8)

    u1 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=2, padding="same")(b1)
    c1 = layers.concatenate([u1, s3])
    d1 = conv_block(c1, base_filters * 4)

    u2 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=2, padding="same")(d1)
    c2 = layers.concatenate([u2, s2])
    d2 = conv_block(c2, base_filters * 2)

    u3 = layers.Conv2DTranspose(base_filters, (2, 2), strides=2, padding="same")(d2)
    c3 = layers.concatenate([u3, s1])
    d3 = conv_block(c3, base_filters)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid", dtype='float32')(d3)
    return models.Model(inputs, outputs, name=name)


# ==========================================
# 7. TRAINING PIPELINE & MULTI-MODEL
# ==========================================
models_dict = {
    'UNet_Base': lambda: build_unet(base_filters=32, name='UNet_Base'),
    'UNet_Large': lambda: build_unet(base_filters=64, name='UNet_Large'), # Ensemble এর জন্য ২য় মডেল
}

saved_models = []

for model_name, model_fn in models_dict.items():
    print(f"\n{'='*50}\n🚀 TRAINING: {model_name}\n{'='*50}")
    tf.keras.backend.clear_session(); gc.collect()
    
    model = model_fn()
    
    # LR Schedule
    steps_per_epoch = len(train_imgs) // CFG.batch_size
    lr_sched = CosineDecayWithWarmup(base_lr=1e-3, total_steps=steps_per_epoch * CFG.epochs, warmup_steps=steps_per_epoch * 2)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr_sched), loss=bce_dice_loss, metrics=SEG_METRICS)
    
    save_path = f"{model_name}_best.keras"
    callbacks = [
        ModelCheckpoint(save_path, monitor='val_dice_coef', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_dice_coef', patience=6, restore_best_weights=True, mode='max', verbose=1),
        CSVLogger(f"{model_name}_log.csv")
    ]
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=CFG.epochs, callbacks=callbacks, verbose=1)
    saved_models.append(save_path)

    # Plot Curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss'); plt.title('Loss'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history.history['dice_coef'], label='Train Dice'); plt.plot(history.history['val_dice_coef'], label='Val Dice'); plt.title('Dice Coef'); plt.legend()
    plt.savefig(f"{model_name}_curves.png"); plt.show()


# ==========================================
# 8. VISUAL EVALUATION (Qualitative Analysis)
# ==========================================
def show_predictions(model, dataset, num_samples=3):
    for images, masks in dataset.take(1):
        preds = model.predict(images, verbose=0)
        
        plt.figure(figsize=(12, num_samples * 4))
        for i in range(min(num_samples, len(images))):
            plt.subplot(num_samples, 3, i*3 + 1); plt.imshow(images[i]); plt.title("Input Image"); plt.axis('off')
            plt.subplot(num_samples, 3, i*3 + 2); plt.imshow(masks[i].numpy().squeeze(), cmap='gray'); plt.title("True Mask"); plt.axis('off')
            
            pred_mask = (preds[i].squeeze() > 0.5).astype(np.float32)
            plt.subplot(num_samples, 3, i*3 + 3); plt.imshow(pred_mask, cmap='gray'); plt.title("Predicted Mask"); plt.axis('off')
        plt.tight_layout(); plt.savefig('segmentation_results.png'); plt.show()

print("\n🔍 Generating Qualitative Visuals (Best Model)...")
best_model = models.load_model(saved_models[0], custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef, 'iou_coef': iou_coef})
show_predictions(best_model, val_ds)


# ==========================================
# 9. TTA (Test Time Augmentation) PREDICTION
# ==========================================
def predict_with_tta(model, image_tensor):
    """
    Segmentation TTA:
    1. আসল ছবির প্রেডিকশন
    2. ছবিকে Left-Right উল্টিয়ে প্রেডিকশন -> মাস্ককেও আবার Left-Right উল্টিয়ে সোজা করা
    3. দুটোর Average
    """
    # ১. Original Prediction
    pred_orig = model.predict(image_tensor, verbose=0)
    
    if CFG.use_tta:
        # ২. Flipped Prediction
        img_flipped = tf.image.flip_left_right(image_tensor)
        pred_flipped = model.predict(img_flipped, verbose=0)
        # মাস্ক সোজা করা
        pred_flipped_rev = tf.image.flip_left_right(pred_flipped)
        
        # Average
        return (pred_orig + pred_flipped_rev) / 2.0
    return pred_orig


# ==========================================
# 10. ENSEMBLE & RLE SUBMISSION (KAGGLE STANDARD)
# ==========================================
print(f"\n{'='*50}\n🎯 ENSEMBLE PREDICTION & RLE SUBMISSION\n{'='*50}")

def rle_encode(mask):
    """Binarized মাস্ককে Kaggle Submission-এর জন্য RLE String-এ কনভার্ট করা"""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Test ইমেজের ফোল্ডার স্ক্যান
if os.path.exists(CFG.test_dir):
    test_files = sorted(os.listdir(CFG.test_dir))
    submission_dict = {'id': [], 'rle_mask': []}

    loaded_models = [models.load_model(p, custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef, 'iou_coef': iou_coef}) for p in saved_models]

    print(f"Predicting {len(test_files)} test images with {len(loaded_models)} models...")
    for f in test_files:
        img_path = os.path.join(CFG.test_dir, f)
        img_id   = f.split('.')[0]
        
        # Test Image Load
        img = tf.image.decode_image(tf.io.read_file(img_path), channels=3)
        img = tf.image.resize(img, CFG.img_size)
        img = tf.cast(img, tf.float32) / 255.0
        img_tensor = tf.expand_dims(img, axis=0) # Batch dimension
        
        # Multi-Model TTA Ensemble
        ensemble_pred = 0
        for m in loaded_models:
            ensemble_pred += predict_with_tta(m, img_tensor)
        ensemble_pred /= len(loaded_models)
        
        # Thresholding
        binary_mask = (ensemble_pred[0].numpy().squeeze() > 0.5).astype(np.uint8)
        
        # RLE Conversion
        rle_str = rle_encode(binary_mask)
        submission_dict['id'].append(img_id)
        submission_dict['rle_mask'].append(rle_str)
        
    # Save CSV
    sub_df = pd.DataFrame(submission_dict)
    sub_df.to_csv('submission.csv', index=False)
    print("\n✅ submission.csv saved successfully!")
    print(sub_df.head())
else:
    print("⚠️ Test directory not found. Skipping submission generation.")

print("\n🏆 SEGMENTATION PIPELINE COMPLETED! BEST OF LUCK! 🚀")
# ==============================================================================