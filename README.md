# Image Retrieval System using Similarity Metrics and Deep Learning

This project implements an image retrieval system that finds and displays images similar to a given query image from a dataset. It explores both basic pixel-level comparison methods and advanced techniques using pre-trained deep learning models (CLIP) and a vector database (ChromaDB) for more accurate and efficient semantic search.

## 📝 Project Overview

Image Retrieval is a system that, given a query image, searches a large database of images and returns a ranked list of the most visually similar ones. This project builds such a system from the ground up, demonstrating two main pipelines:

1.  **Basic Retrieval**: Compares images directly based on their raw pixel values using mathematical distance metrics.
2.  **Advanced Retrieval**: Leverages a deep learning model to understand the semantic content of images, converting them into feature vectors for comparison. This approach is further optimized using a specialized vector database.

### General Pipeline
![General Pipeline](https://github.com/HeigatVu/image-retrieval/blob/main/pipeline.png)

## ⚙️ Methodology

The project is divided into two main approaches for retrieving images.

### 1. Basic Image Retrieval (Pixel-based)

This method operates directly on the pixel values of the images. The query image and all images in the database are resized to a uniform dimension. Then, their similarity is calculated using various distance metrics.

**Similarity Metrics Used:**
* **L1 Distance (Manhattan Distance)**: Calculates the sum of the absolute differences between pixel values.
    $L1(\vec{a}, \vec{b}) = \sum_{i=1}^{N} |a_i - b_i|$
* **L2 Distance (Euclidean Distance)**: Calculates the square root of the sum of squared differences between pixel values.
    $L2(\vec{a}, \vec{b}) = \sqrt{\sum_{i=1}^{N} (a_i - b_i)^2}$
* **Cosine Similarity**: Measures the cosine of the angle between two image vectors, focusing on orientation rather than magnitude.
    $cosine\_similarity(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{||\vec{a}|| ||\vec{b}||}$
* **Correlation Coefficient**: Measures the linear relationship between two image vectors.

While simple, this approach often fails on complex images where semantic meaning is more important than pixel layout (e.g., finding a picture of a crocodile by its features, not its exact position and color in the frame).

### 2. Advanced Image Retrieval (Feature-based)

To overcome the limitations of pixel-based methods, this approach uses a pre-trained deep learning model to extract high-level semantic features from images.

**Pipeline:**
1.  **Feature Extraction**: The **CLIP (Contrastive Language-Image Pre-Training)** model is used to convert each image (both the query and the database images) into a dense feature vector (embedding). This vector represents the image's semantic content.
2.  **Similarity Calculation**: The same distance metrics (L1, L2, Cosine Similarity) are then applied to these feature vectors instead of the raw pixel data. This results in a much more accurate, content-aware search.

![Advanced Pipeline](https://i.imgur.com/u3gQ2iB.png)

### 3. Optimized Retrieval with a Vector Database

To make the search process faster and more scalable, the advanced approach is optimized by using **ChromaDB**, a vector database.

**Process:**
1.  **Indexing (Offline)**: All images in the database are converted into feature vectors using CLIP. These vectors are then stored and indexed in a ChromaDB "collection". This is a one-time process.
2.  **Querying (Online)**: When a new query image is provided, it is converted into a feature vector using CLIP. This vector is then used to query the ChromaDB collection, which efficiently finds the most similar vectors (and thus, the most similar images) using its optimized search algorithms (like HNSW).

This method is significantly faster for large datasets as it avoids re-calculating feature vectors for the entire database on every query.

## 📈 Results

The advanced method using CLIP provides significantly better results, especially for complex images. The model understands the objects and context within the image, leading to semantically relevant matches.

**Basic Method (L1) - Complex Image:**
![Basic L1 Result](https://i.imgur.com/9y0h4uS.png)
*The results are based on color and texture, not the subject.*

**Advanced Method (CLIP + L1) - Complex Image:**
![Advanced CLIP Result](https://i.imgur.com/5A0O6tI.png)
*The results are all images of the same subject (African crocodile), demonstrating a clear understanding of the image content.*

## 🛠️ Technologies Used

* **Python**
* **NumPy**: For numerical operations on image matrices and vectors.
* **Pillow (PIL)**: For opening, resizing, and manipulating images.
* **Matplotlib**: For displaying the query and result images.
* **open-clip-torch**: Provides the pre-trained CLIP model for feature extraction.
* **chromadb**: An open-source vector database used for efficient storage and retrieval of image embeddings.

## 🚀 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install numpy Pillow matplotlib
    pip install chromadb open-clip-torch
    ```
2.  **Prepare Data**: Organize your dataset into `data/train` and `data/test` directories. Each subdirectory within `train` and `test` should represent an image class.
3.  **Run the Notebook**: Open the project's `.ipynb` notebook and execute the cells sequentially. You can switch between the basic, advanced, and optimized methods to compare their performance.
S
