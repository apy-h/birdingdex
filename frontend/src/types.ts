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

// Utility function to clean bird species names for display
export function cleanBirdName(name: string): string {
  // Remove number prefix (e.g., "097.") and replace underscores with spaces
  let cleaned = name;
  if (cleaned.includes('.')) {
    cleaned = cleaned.split('.', 2)[1] || cleaned;
  }
  cleaned = cleaned.replace(/_/g, ' ');
  return cleaned;
}
