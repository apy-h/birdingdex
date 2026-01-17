"""
FastAPI backend for Birdingdex application.
Provides REST APIs for image upload, bird classification, and image augmentation.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import io
import base64
from typing import Optional, List
import uvicorn

from ml_service import BirdClassifier, ImageAugmenter

app = FastAPI(title="Birdingdex API", version="1.0.0")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML models (lazy loading)
bird_classifier = None
image_augmenter = None


def get_bird_classifier():
    """Lazy initialization of bird classifier."""
    global bird_classifier
    if bird_classifier is None:
        bird_classifier = BirdClassifier()
    return bird_classifier


def get_image_augmenter():
    """Lazy initialization of image augmenter."""
    global image_augmenter
    if image_augmenter is None:
        image_augmenter = ImageAugmenter()
    return image_augmenter


class ClassificationResult(BaseModel):
    """Bird classification result."""
    species: str
    confidence: float
    top_predictions: List[dict]


class AugmentationRequest(BaseModel):
    """Request for image augmentation."""
    image_base64: str
    edit_type: str  # "hat", "bowtie", "glasses", etc.
    prompt: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to Birdingdex API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/classify", response_model=ClassificationResult)
async def classify_bird(file: UploadFile = File(...)):
    """
    Classify bird species from uploaded image.

    Args:
        file: Uploaded image file

    Returns:
        Classification result with species and confidence
    """
    try:
        # Read image data
        image_data = await file.read()

        # Get classifier and perform inference
        classifier = get_bird_classifier()
        result = classifier.classify(image_data)

        return ClassificationResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/api/augment")
async def augment_image(request: AugmentationRequest):
    """
    Apply cosmetic edits to bird image using Stable Diffusion inpainting.

    Args:
        request: Augmentation request with image and edit type

    Returns:
        Augmented image as base64
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_base64)

        # Get augmenter and apply edit
        augmenter = get_image_augmenter()
        augmented_image = augmenter.apply_edit(
            image_bytes,
            request.edit_type,
            request.prompt
        )

        # Encode result as base64
        buffered = io.BytesIO()
        augmented_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {"augmented_image": img_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Augmentation failed: {str(e)}")


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and validate bird image.

    Args:
        file: Uploaded image file

    Returns:
        Upload confirmation with image metadata
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image data
        image_data = await file.read()

        # Convert to base64 for frontend
        img_base64 = base64.b64encode(image_data).decode()

        return {
            "success": True,
            "filename": file.filename,
            "size": len(image_data),
            "image_base64": img_base64
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/species")
async def get_species_list():
    """
    Get list of all supported bird species.

    Returns:
        List of bird species
    """
    try:
        classifier = get_bird_classifier()
        species_list = classifier.get_species_list()
        return {"species": species_list, "count": len(species_list)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get species list: {str(e)}")


@app.get("/api/model/metrics")
async def get_model_metrics():
    """
    Get model training metrics and performance statistics.

    Returns:
        Model metrics including accuracy, hyperparameters, and per-class performance
    """
    try:
        classifier = get_bird_classifier()
        metrics = classifier.get_model_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model metrics: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
