from huggingface_hub import HfApi, HfFolder

# --- Configuration ---
# Your Hugging Face username
HF_USERNAME = "abhimnyu09"
# The name of the repository you just created
REPO_NAME = "derm-detect-multimodal-model"

# --- Login and Upload ---
# You'll be prompted to enter a Hugging Face token. You can get one from
# your Hugging Face profile -> Settings -> Access Tokens -> New token.
print("Logging in to Hugging Face Hub...")
HfFolder.save_token(input("Enter your Hugging Face write token: "))

api = HfApi()

print("Uploading model files...")
api.upload_file(
    path_or_fileobj="multimodal_derm_model.h5",
    path_in_repo="multimodal_derm_model.h5",
    repo_id=f"{HF_USERNAME}/{REPO_NAME}",
    repo_type="model",
    create_pr=True
)

api.upload_file(
    path_or_fileobj="label_mapping.json",
    path_in_repo="label_mapping.json",
    repo_id=f"{HF_USERNAME}/{REPO_NAME}",
    repo_type="model",
    create_pr=True
)

print("✅ Files uploaded successfully!")
print(f"Find them at: https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")
