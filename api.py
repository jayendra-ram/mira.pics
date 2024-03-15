import modal
from fastapi import FastAPI, UploadFile, File
from deepface import DeepFace
import itertools
import uuid
import os

stub = modal.Stub("face-similarity-api")
image_dir = "/tmp"

app = FastAPI()

@stub.function(
    image=modal.Image.debian_slim().pip_install(["deepface"]),
    # secret=modal.Secret.from_name("my-secret"),
    gpu="any",
)
def calculate_similarity(file1_path: str, file2_path: str):
    # Calculate the similarity between the two faces using DeepFace
    result = DeepFace.verify(img1_path=file1_path, img2_path=file2_path)
    
    # Extract the similarity percentage from the result
    similarity_percentage = result['distance']
    
    return similarity_percentage

@app.post("/compare")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Generate unique filenames for the uploaded images
    file1_name = f"{uuid.uuid4()}.jpg"
    file2_name = f"{uuid.uuid4()}.jpg"
    
    # Save the uploaded images to the image directory
    file1_path = os.path.join(image_dir, file1_name)
    file2_path = os.path.join(image_dir, file2_name)
    
    with open(file1_path, "wb") as f:
        f.write(await file1.read())
    
    with open(file2_path, "wb") as f:
        f.write(await file2.read())
    
    # Calculate the similarity between the two faces
    similarity = calculate_similarity.call(file1_path, file2_path)
    
    # Remove the temporary image files
    os.remove(file1_path)
    os.remove(file2_path)
    
    return {"similarity": similarity}

@stub.local_entrypoint()
def main():
    ## run on two files
    calculate_similarity("rg.jpeg","Ryan Gosling.jpeg") 

if __name__ == "__main__":
    with stub.run():
        main()