export interface Bird {
  species: string;
  confidence: number;
  imageUrl: string;
  topPredictions?: {
    species: string;
    confidence: number;
  }[];
  augmentedImageUrl?: string;
}

export interface ClassificationResult {
  species: string;
  confidence: number;
  top_predictions: {
    species: string;
    confidence: number;
  }[];
}

export interface UploadResponse {
  success: boolean;
  filename: string;
  size: number;
  image_base64: string;
}

export interface AugmentationResponse {
  augmented_image: string;
}
