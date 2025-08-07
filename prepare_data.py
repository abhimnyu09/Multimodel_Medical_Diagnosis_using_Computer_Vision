import pandas as pd
import os
from sklearn.model_selection import train_test_split

# --- 1. Load Metadata ---
metadata_path = 'data/HAM10000_metadata.csv'
df = pd.read_csv(metadata_path)
print("Dataframe loaded. Shape:", df.shape)

# --- 2. Create Image Path Column ---
image_dir = 'data/images'
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(image_dir, x + '.jpg'))

# --- 3. Create Synthetic Symptom Text ---
# This is a crucial step for the multimodal part.
# In a real-world scenario, you'd have real patient data.
# Here, we generate plausible text based on the diagnosis ('dx').
symptom_templates = {
    'nv': ["It's a common mole that has been there for a while.", "A dark spot, looks like a regular mole.", "A small, unchanging mole on my skin."],
    'mel': ["A mole that has changed in size, shape, or color recently.", "An asymmetrical dark lesion that is growing.", "A new, suspicious-looking dark spot that sometimes bleeds."],
    'bkl': ["A waxy, 'stuck-on' growth that looks like a wart.", "A brownish, rough patch of skin.", "A benign lesion, feels a bit scaly."],
    'bcc': ["A pearly or waxy bump, sometimes with visible blood vessels.", "A flat, flesh-colored or brown scar-like lesion.", "A sore that heals and then returns."],
    'akiec': ["A dry, scaly patch on sun-exposed skin.", "A rough patch that can sometimes be felt more than seen.", "A pre-cancerous spot, feels like sandpaper."],
    'vasc': ["A small, red or purple bump, looks like a blood blister.", "A vascular lesion, a collection of small blood vessels.", "A reddish spot on the skin."],
    'df': ["A small, hard lump under the skin.", "A firm bump, often on the legs.", "A dermatofibroma, feels like a small stone under the skin."]
}

def generate_symptom_text(dx):
    import random
    return random.choice(symptom_templates[dx])

df['symptoms'] = df['dx'].apply(generate_symptom_text)
print("\nGenerated symptom text for each entry.")
print(df.head())

# --- 4. Encode Labels ---
# Convert string labels ('nv', 'mel', etc.) to numbers (0, 1, etc.)
label_mapping = {label: i for i, label in enumerate(df['dx'].unique())}
df['label'] = df['dx'].map(label_mapping)
print("\nLabel Mapping:", label_mapping)

# --- 5. Split Data into Training, Validation, and Test sets ---
train_df, test_val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df['label'], random_state=42)

print(f"\nTraining set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# --- 6. Save the Processed Dataframes ---
# We will use these files in our model training script.
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
print("\nProcessed dataframes saved to 'data/' directory.")

# Save label mapping for later use in the app
import json
with open('label_mapping.json', 'w') as f:
    json.dump(label_mapping, f)
