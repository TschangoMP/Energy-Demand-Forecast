import pandas as pd
import requests
import zipfile
import io

# Dataset URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"

# Download and extract the ZIP file
response = requests.get(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))
zip_file.extractall("../data/bronze/LD2011_2014.txt") 