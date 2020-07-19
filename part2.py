import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from PIL import Image
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
     response = requests.get(url1)
elif img_num == 2:
     response = requests.get(url2)
elif img_num == 3:
     response = requests.get(url3)
elif img_num == 4:
     response = requests.get(url4)
elif img_num == 5: 
     response = requests.get(url5)

img = Image.open(BytesIO(response.content))
img = np.array(img, dtype=np.float64) / 255

w, h, d = original_shape = tuple(img.shape)
image_array = np.reshape(img, (w * h, d))

image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

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

plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original Image')
plt.imshow(img)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized Image (K-Means)')
img_data = recreate_image(kmeans.cluster_centers_, labels, w, h)
plt.imshow(img_data)
rescaled = (255.0 / img_data.max() * (img_data - img_data.min())).astype(np.uint8)
im = Image.fromarray(rescaled).convert('RGB')
# im.save('quantized' + str(img_num) + '.jpg')  # Save quantized image to jpg file
plt.show()