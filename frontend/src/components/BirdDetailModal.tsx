import React, { useState } from 'react';
import { Bird, cleanBirdName } from '../types';
import { api } from '../api';
import './BirdDetailModal.css';

interface BirdDetailModalProps {
  bird: Bird;
  isOpen: boolean;
  onClose: () => void;
  onAugmented?: (augmentedImageUrl: string, birdSpecies?: string) => void;
  onAugmentingStateChange?: (isAugmenting: boolean) => void;
}

const BirdDetailModal: React.FC<BirdDetailModalProps> = ({ bird, isOpen, onClose, onAugmented, onAugmentingStateChange }) => {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [showAugmented, setShowAugmented] = useState<{ [key: number]: boolean }>({});
  const [isAugmenting, setIsAugmenting] = useState(false);

  if (!isOpen || !bird.images || bird.images.length === 0) {
    return null;
  }

  const images = bird.images;
  const currentBirdImage = images[currentImageIndex];

  const toggleAugmented = (index: number) => {
    setShowAugmented(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const hasAugmentedImages = currentBirdImage.augmentedImages && currentBirdImage.augmentedImages.length > 0;
  const displayImage =
    showAugmented[currentImageIndex] && hasAugmentedImages
      ? currentBirdImage.augmentedImages![0]
      : currentBirdImage.imageUrl;

  const goToPrevious = () => {
    setCurrentImageIndex(prev => (prev > 0 ? prev - 1 : images.length - 1));
  };

  const goToNext = () => {
    setCurrentImageIndex(prev => (prev < images.length - 1 ? prev + 1 : 0));
  };

  const handleAugment = async (editType: string) => {
    // Don't allow augmentation if this image already has one
    if (currentBirdImage.augmentedImages && currentBirdImage.augmentedImages.length > 0) {
      console.log('Image already augmented');
      return;
    }

    setIsAugmenting(true);
    onAugmentingStateChange?.(true);
    try {
      // Extract base64 from data URL
      const base64 = currentBirdImage.imageUrl.split(',')[1];
      const result = await api.augmentImage(base64, editType);
      const augmentedUrl = `data:image/png;base64,${result.augmented_image}`;
      setShowAugmented(prev => ({ ...prev, [currentImageIndex]: true }));

      // Notify parent component about augmented image
      if (onAugmented) {
        onAugmented(augmentedUrl, bird.species);
      }
    } catch (err) {
      console.error('Augmentation error:', err);
    } finally {
      setIsAugmenting(false);
      onAugmentingStateChange?.(false);
    }
  };

  return (
    <div className="bird-modal-overlay" onClick={onClose}>
      <div className="bird-modal-content" onClick={e => e.stopPropagation()}>
        <button className="modal-close-btn" onClick={onClose}>✕</button>

        <div className="modal-header">
          <h2>{cleanBirdName(bird.species)}</h2>
          <p className="image-counter">{currentImageIndex + 1} of {images.length}</p>
        </div>

        <div className="modal-body">
          <div className="modal-left-section">
            <div className="modal-predictions-section">
              <h3>Top Predictions</h3>
              {currentBirdImage.topPredictions && currentBirdImage.topPredictions.length > 0 ? (
                <div className="predictions-list">
                  {currentBirdImage.topPredictions.slice(0, 3).map((pred, index) => (
                    <div key={index} className="prediction-item">
                      <span className="pred-species">{cleanBirdName(pred.species)}</span>
                      <span className="pred-confidence">{(pred.confidence * 100).toFixed(1)}%</span>
                      <div className="pred-bar">
                        <div
                          className="pred-fill"
                          style={{ width: `${pred.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="no-predictions">No predictions available</p>
              )}
            </div>

            {images.length > 1 && (
              <div className="modal-sightings-section">
                <h3>Sightings</h3>
                <div className="image-thumbnails-grid">
                  {images.map((img, index) => (
                    <button
                      key={index}
                      className={`thumbnail ${index === currentImageIndex ? 'active' : ''}`}
                      onClick={() => {
                        setCurrentImageIndex(index);
                      }}
                      title={`Sighting ${index + 1}`}
                    >
                      <img src={img.imageUrl} alt={`${bird.species} ${index + 1}`} />
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="modal-image-section">
            <div className="modal-image-container">
              <img src={displayImage} alt={bird.species} className="modal-bird-image" />
            </div>

            {images.length > 1 && (
              <div className="slideshow-controls">
                <button onClick={goToPrevious} className="nav-btn prev-btn">
                  ← Previous
                </button>
                <button onClick={goToNext} className="nav-btn next-btn">
                  Next →
                </button>
              </div>
            )}

            <div className="augmentation-section">
              <div className="augment-buttons">
                <button
                  onClick={() => handleAugment('hat')}
                  disabled={isAugmenting || hasAugmentedImages}
                  className="augment-btn"
                  title="Add a hat to the bird"
                >
                  🎩 Hat
                </button>
                <button
                  onClick={() => handleAugment('bowtie')}
                  disabled={isAugmenting || hasAugmentedImages}
                  className="augment-btn"
                  title="Add a bowtie to the bird"
                >
                  🎀 Bowtie
                </button>
                <button
                  onClick={() => handleAugment('glasses')}
                  disabled={isAugmenting || hasAugmentedImages}
                  className="augment-btn"
                  title="Add glasses to the bird"
                >
                  🕶️ Glasses
                </button>
              </div>

              {hasAugmentedImages && (
                <button
                  onClick={() => toggleAugmented(currentImageIndex)}
                  className="augmented-toggle-btn"
                >
                  {showAugmented[currentImageIndex] ? 'Show Original' : 'Show Augmented'}
                </button>
              )}
            </div>

            <p className="confidence-text">
              Confidence: {(currentBirdImage.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        <div className="modal-footer">
        </div>
      </div>
    </div>
  );
};

export default BirdDetailModal;
