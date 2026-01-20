import React, { useState, useRef } from 'react';
import { api } from '../api';
import { Bird } from '../types';
import './ImageUpload.css';

interface ImageUploadProps {
  onBirdDiscovered: (bird: Bird) => void;
  isAugmenting?: boolean;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onBirdDiscovered, isAugmenting = false }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isDisabled = isUploading || isAugmenting;

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);

    // Upload and classify
    setIsUploading(true);
    setError(null);

    try {
      // Classify the bird
      const result = await api.classifyBird(file);

      // Create bird object
      const bird: Bird = {
        species: result.species,
        confidence: result.confidence,
        imageUrl: preview || '',
        topPredictions: result.top_predictions,
      };

      // Set the actual image URL
      const uploadResult = await api.uploadImage(file);
      bird.imageUrl = `data:image/jpeg;base64,${uploadResult.image_base64}`;

      onBirdDiscovered(bird);

      // Reset
      setPreview(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err) {
      console.error('Error processing image:', err);
      setError('Failed to classify bird. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      if (fileInputRef.current) {
        fileInputRef.current.files = dataTransfer.files;
        handleFileSelect({ target: fileInputRef.current } as any);
      }
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  return (
    <div className="image-upload">
      <div
        className={`upload-area ${isDisabled ? 'disabled' : ''}`}
        onDrop={isDisabled ? undefined : handleDrop}
        onDragOver={isDisabled ? undefined : handleDragOver}
        onClick={isDisabled ? undefined : () => fileInputRef.current?.click()}
      >
        {preview ? (
          <img src={preview} alt="Preview" className="preview-image" />
        ) : (
          <div className="upload-prompt">
            <div className="upload-icon">📸</div>
            <h3>Upload a Bird Photo</h3>
            <p>Click to browse or drag and drop</p>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={isDisabled ? undefined : handleFileSelect}
          disabled={isDisabled}
          style={{ display: 'none' }}
        />
      </div>

      {isUploading && (
        <div className="loading">
          <div className="spinner"></div>
          <p>Identifying bird species...</p>
        </div>
      )}

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
