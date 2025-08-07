import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import Sequence
from transformers import BertTokenizer, TFBertModel
from PIL import Image
from sklearn.utils import class_weight

# --- Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32
BERT_MODEL_NAME = 'bert-base-uncased'
NUM_CLASSES = 7

# --- 1. Custom Data Generator ---
# This class is essential for feeding paired image and text data to the model.
class MultimodalDataGenerator(Sequence):
    def __init__(self, df, tokenizer, batch_size, shuffle=True):
        self.df = df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = self.df.index.tolist()
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.df.loc[batch_indices]

        # Get image data
        image_batch = np.array([self.load_image(path) for path in batch_df['image_path']])
        
        # Get text data
        text_batch = self.tokenizer(
            batch_df['symptoms'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='tf'
        )
        
        # Get labels
        label_batch = tf.keras.utils.to_categorical(batch_df['label'], num_classes=NUM_CLASSES)

        return [image_batch, text_batch['input_ids'], text_batch['attention_mask']], label_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        return np.array(img) / 255.0

# --- 2. Build Model Components ---
def build_model_components():
    # Image Stream (CNN)
    cnn_base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
    for layer in cnn_base.layers:
        layer.trainable = False # Freeze the base
    
    # Text Stream (BERT)
    bert_base = TFBertModel.from_pretrained(BERT_MODEL_NAME)
    for layer in bert_base.layers:
        layer.trainable = False # Freeze the base

    return cnn_base, bert_base

# --- 3. Build the Full Multimodal Model ---
def build_multimodal_model(cnn_base, bert_base):
    # Image Input
    image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    cnn_features = cnn_base(image_input)

    # Text Inputs
    text_input_ids = Input(shape=(64,), dtype=tf.int32, name='text_input_ids')
    text_attention_mask = Input(shape=(64,), dtype=tf.int32, name='text_attention_mask')
    bert_output = bert_base(input_ids=text_input_ids, attention_mask=text_attention_mask)
    bert_features = bert_output.pooler_output

    # Fusion
    concatenated_features = Concatenate()([cnn_features, bert_features])
    
    # Classifier Head
    x = Dropout(0.4)(concatenated_features)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    # Create model
    model = Model(inputs=[image_input, text_input_ids, text_attention_mask], outputs=output)
    return model

# --- Main Training Script ---
if __name__ == '__main__':
    # Load processed data
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # Create data generators
    train_generator = MultimodalDataGenerator(train_df, tokenizer, BATCH_SIZE)
    val_generator = MultimodalDataGenerator(val_df, tokenizer, BATCH_SIZE, shuffle=False)

    # Build the model
    cnn_base, bert_base = build_model_components()
    model = build_multimodal_model(cnn_base, bert_base)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Calculate class weights for imbalanced data
    class_weights = class_weight.compute_class_weight('balanced',
                                                     classes=np.unique(train_df['label']),
                                                     y=train_df['label'])
    class_weights_dict = dict(enumerate(class_weights))
    print("Class Weights:", class_weights_dict)

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10, # Increase epochs for better results, e.g., 20-30
        class_weight=class_weights_dict,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )

    # Save the final model
    model.save('multimodal_derm_model.h5')
    print("Model saved as multimodal_derm_model.h5")
