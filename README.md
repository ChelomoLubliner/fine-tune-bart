# fine-tune-bart
Fine-tune Bart for Summarize Medical Articles

This repo contains materials for final project of Computational Learning (Master @HUJI).
We will do a fine-tune of Bart, a denoising autoencoder for pretraining sequence-to-sequence models, for summarize medical articles.
We will use hugging face, so this code is reusable for any fine-tune with hugging face.

It includes the summary of the project, a Python file, the data as a JSON file, and the requirements for running the script.

Note: You have to put the JSON file in a folder and to set the path in the code. You have also to put your access token of your hugging face account (free) and run this commande in your console : !python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('YOU_USER_ACCESS_TOKEN')"
