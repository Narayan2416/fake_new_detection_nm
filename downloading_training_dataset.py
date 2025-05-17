import gdown

file_id = "1klN3WXILu4YVvVMHIl6h_fryzJ0pD73R"
url = f"https://drive.google.com/uc?id={file_id}"
output = "train.csv"


gdown.download(url, output, quiet=False)
