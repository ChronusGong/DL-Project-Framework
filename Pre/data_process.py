import os
import pandas as pd
import torch
from tqdm import tqdm
from torchvision.io import read_image
from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPModel

# Have to specify device here or CPU will used by default
clip_model = CLIPModel.from_pretrained("../clip-vit-large-patch14", from_tf=False, local_files_only=True,
                                       device_map="cuda:0")
image_processor = CLIPImageProcessor.from_pretrained("../clip-vit-large-patch14", from_tf=False, local_files_only=True,
                                                     device_map="cuda:0")


# get embedding
def get_embedding(file_name):
    img_path = os.path.join('../street_view', file_name)
    image = read_image(img_path)

    inputs = image_processor(images=image, return_tensors="pt")
    inputs.to("cuda")
    embedding =clip_model.get_image_features(**inputs)
    embedding = embedding.to("cpu")
    embedding = embedding.detach().numpy()

    return embedding


if __name__ == '__main__':
    df = pd.read_csv('../income.csv')

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        fn = row[2]
        emb = get_embedding(fn)
        torch.save(emb, f'./data/street_view_emb/' + fn.replace('.jpg', '.pt'))
