print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from PIL import Image
import requests
from io import BytesIO
import requests, io, sys

n_colors = 5
img_num = int(sys.argv[1]) or 1

url1 = "https://personal.utdallas.edu/~axn112530/cs6375/unsupervised/images/image1.jpg"
url2 = "https://personal.utdallas.edu/~axn112530/cs6375/unsupervised/images/image2.jpg"
url3 = "https://personal.utdallas.edu/~axn112530/cs6375/unsupervised/images/image3.jpg"
url4 = "https://personal.utdallas.edu/~axn112530/cs6375/unsupervised/images/image4.jpg"
url5 = "https://personal.utdallas.edu/~axn112530/cs6375/unsupervised/images/image5.jpg"

if img_num == 1:
     selected_response = requests.get(url1)
elif img_num == 2:
     selected_response = requests.get(url2)
elif img_num == 3:
     selected_response = requests.get(url3)
elif img_num == 4:
     selected_response = requests.get(url4)
elif img_num == 5: 
     selected_response = requests.get(url5)


# Load the Summer Palace photo
# img = load_sample_image("img.jpg")
# img = urllib.request.urlopen(data_url)
response = selected_response
img = Image.open(BytesIO(response.content))

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
img = np.array(img, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(img.shape)
assert d == 3
image_array = np.reshape(img, (w * h, d))

print("Fitting model on a small sub-sample of the data")
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)


# Get labels for all points
print("Predicting color indices on the full image (k-means)")
labels = kmeans.predict(image_array)



def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# Display all results, alongside original image
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image')
plt.imshow(img)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized image (K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
plt.show()