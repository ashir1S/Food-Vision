# Food-101 Classifier

This project is a Food Image Classifier built using PyTorch and deployed as a Gradio web application on Hugging Face Spaces. The model is trained on the Food-101 dataset, which consists of 101 different food categories.

![A user uploading an image to the Food vision web app](https://github.com/ashir1S/Dog-Breed-Classifier/blob/main/demo/img.png)

## Model Architecture

The project leverages transfer learning with the Vision Transformer (ViT-B/32) model from torchvision.models.

* **Framework:** PyTorch
* **Core Library:** torchvision
* **Backbone Model:** ViT-B/32 (Vision Transformer)
* **Head:** Custom classifier with LayerNorm, Dropout, and a Linear layer for 101 classes.
* **Transforms:** Standard preprocessing (resize, normalization, augmentation).

## Results

The model demonstrates strong performance on the Food-101 benchmark:

| Metric         | Value                            |
| :------------- | :------------------------------- |
| Top-1 Accuracy | 80.97%                           |
| Dataset        | Food-101 (25,250 test images)    |
| Model          | ViT-B/32                         |
| Training       | Fine-tuned with transfer learning|

## Deployment on Hugging Face Spaces

The trained model has been deployed as an interactive demo using Gradio.

* **Hugging Face Space:** [https://huggingface.co/spaces/Ashirwad12/Food-101-Classifier](https://huggingface.co/spaces/Ashirwad12/food-vision)
* **Gradio Integration:** Provides an upload interface to test predictions directly in the browser.

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ashir1S/Food-101-Classifier.git](https://github.com/ashir1S/Food-101-Classifier.git)
    cd Food-101-Classifier
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the dataset automatically via torchvision:**
    In a Python session or notebook cell, run:
    ```python
    from torchvision.datasets import Food101
    dataset = Food101(root="data", download=True)
    ```

4.  **Run the Jupyter Notebook:**
    Open and execute `Food1O1.ipynb` to train the model, or load a pre-trained checkpoint from Google Drive.

## Contributing

Contributions are welcome! Feel free to fork the repo, open an issue, or submit a pull request with improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
