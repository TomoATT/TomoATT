# %%
# download src_ref_files from Zenodo
import os
import requests


url = 'https://zenodo.org/records/18195841/files/src_rec_Turkey.tar.gz?download=1'

path = "./input_data/src_rec_Turkey.tar.gz"

# check file existencetar.gz
if not os.path.exists(path):
    try:
        os.mkdir("./input_data")
    except:
        pass
    print("Downloading src_rec_Turkey.tar.gz from Zenodo...")
    response = requests.get(url, stream=True)
    with open(path, 'wb') as out_file:
        out_file.write(response.content)
    print("Download complete.")
else:
    print("src_rec_Turkey.tar.gz already exists.")

# %%
# download initial model from Zenodo

url = 'https://zenodo.org/records/18195841/files/model_1d_crust1.0_N61_121_61.h5?download=1'

path = "../2_models/model_1d_crust1.0_N61_121_61.h5"

# check file existence
if not os.path.exists(path):
    try:
        os.mkdir("../2_models")
    except:
        pass
    print("Downloading model_1d_crust1.0_N61_121_61.h5 from Zenodo...")
    response = requests.get(url, stream=True)
    with open(path, 'wb') as out_file:
        out_file.write(response.content)
    print("Download complete.")
else:
    print("model_1d_crust1.0_N61_121_61.h5 already exists.")


