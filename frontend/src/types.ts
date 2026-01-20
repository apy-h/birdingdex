export interface BirdImage {
  imageUrl: string;
  confidence: number;
  topPredictions?: {
    species: string;
    confidence: number;
  }[];
  augmentedImages?: string[]; // Array of augmented image data URLs
}

export interface Bird {
  species: string;
  confidence: number;
  imageUrl: string;
  topPredictions?: {
    species: string;
    confidence: number;
  }[];
  // For collection view with multiple images per species
  images?: BirdImage[];
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
  // Remove number prefix (e.g., "097.")
  let cleaned = name;
  if (cleaned.includes('.')) {
    cleaned = cleaned.split('.', 2)[1] || cleaned;
  }

  // Connect uppercase word followed by lowercase word with hyphen (e.g., "Red_tailed_Hawk" → "Red-tailed_Hawk")
  cleaned = cleaned.replace(/([A-Z][a-z]*)_([a-z])/g, '$1-$2');

  // Replace remaining underscores with spaces
  cleaned = cleaned.replace(/_/g, ' ');

  return cleaned;
}
