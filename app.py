# app.py (Optimized with FAISS IVF & Batch Inference)
from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
import faiss
from waitress import serve
import os
import uuid
from werkzeug.utils import secure_filename

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

# ------------------------------
# MobileNet Embedder Definition
# ------------------------------
class MobileNetEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.model.classifier = torch.nn.Identity()

    def forward(self, x):
        return self.model(x)

# ------------------------------
# UploadAPI Class Definition
# ------------------------------
class UploadAPI:
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Load embeddings and paths
        data = np.load('embeddings.npz')
        self.embeddings = data['embeddings'].astype(np.float32)
        self.paths = data['paths']
        
        # Normalize embeddings for cosine similarity
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        num_embeddings, emb_dim = self.embeddings.shape

        # Dynamically choose FAISS index based on dataset size
        if num_embeddings < 1000:
            # Small dataset: use flat index (exact search)
            self.index = faiss.IndexFlatL2(emb_dim)
            self.index.add(self.embeddings)
            print(f"Using IndexFlatL2 with {num_embeddings} embeddings.")
        else:
            # Larger dataset: use IVF index with dynamic clustering
            nlist = max(1, min(num_embeddings // 39, 256))  # FAISS recommendation (~39 points per centroid)
            quantizer = faiss.IndexFlatL2(emb_dim)
            self.index = faiss.IndexIVFFlat(quantizer, emb_dim, nlist)
            self.index.train(self.embeddings)
            self.index.add(self.embeddings)
            print(f"Using IndexIVFFlat with {nlist} clusters for {num_embeddings} embeddings.")

    def image_to_embedding(self, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(image_tensor).cpu().numpy()
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)  # Normalize embedding
        return embedding

    def search_similar(self, query_embedding, k=3):
        dists, idxs = self.index.search(query_embedding.astype(np.float32), k)
        scores = 1 - dists[0] / 2  # Convert L2 distance to cosine similarity approximation
        similar_images = [{
            'path': self.paths[idx],
            'score': float(score)
        } for idx, score in zip(idxs[0], scores)]
        
        return similar_images

# Initialize API instance globally
api = UploadAPI(MobileNetEmbedder())

# ------------------------------
# Flask Routes and Handlers
# ------------------------------

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        
        # Create uploads directory if not exists
        uploads_dir = os.path.join(app.root_path, 'static', 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        # Save uploaded file securely with unique name
        original_filename = secure_filename(file.filename)
        ext = os.path.splitext(original_filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        
        filepath = os.path.join(uploads_dir, unique_filename)
        file.save(filepath)

        # Process uploaded image to embedding and search similar images
        image = Image.open(filepath).convert('RGB')
        
        embedding = api.image_to_embedding(image)
        
        results = api.search_similar(embedding)

        # Apply similarity threshold (e.g., 90%)
        filtered_results = [res for res in results if res['score'] >= 0.9]

        similar_paths = [{
            'path': url_for('static', filename=res['path'].replace('\\', '/').replace('static/', '')),
            'score': res['score']
        } for res in filtered_results]

        uploaded_image_url = url_for('static', filename=f'uploads/{unique_filename}')

        return jsonify({
            'uploaded_image': uploaded_image_url,
            'similar_images': similar_paths
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ------------------------------
# Main Entry Point (Waitress Server)
# ------------------------------
if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)
