import React, { useState } from 'react';
import { Bird, BirdImage, cleanBirdName } from '../types';
import './BirdDetailModal.css';

interface BirdDetailModalProps {
  bird: Bird;
  isOpen: boolean;
  onClose: () => void;
}

const BirdDetailModal: React.FC<BirdDetailModalProps> = ({ bird, isOpen, onClose }) => {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [showAugmented, setShowAugmented] = useState<{ [key: number]: boolean }>({});

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

  return (
    <div className="bird-modal-overlay" onClick={onClose}>
      <div className="bird-modal-content" onClick={e => e.stopPropagation()}>
        <button className="modal-close-btn" onClick={onClose}>✕</button>

        <div className="modal-header">
          <h2>{cleanBirdName(bird.species)}</h2>
          <p className="image-counter">{currentImageIndex + 1} of {images.length}</p>
        </div>

        <div className="modal-body">
          <div className="modal-image-section">
            <div className="modal-image-container">
              <img src={displayImage} alt={bird.species} className="modal-bird-image" />
            </div>

            <div className="slideshow-controls">
              <button onClick={goToPrevious} className="nav-btn prev-btn">
                ← Previous
              </button>
              <button onClick={goToNext} className="nav-btn next-btn">
                Next →
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

            <p className="confidence-text">
              Confidence: {(currentBirdImage.confidence * 100).toFixed(1)}%
            </p>
          </div>

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
        </div>

        <div className="modal-footer">
          <p className="thumbnail-label">Sightings</p>
          <div className="image-thumbnails">
            {images.map((img, index) => (
              <button
                key={index}
                className={`thumbnail ${index === currentImageIndex ? 'active' : ''}`}
                onClick={() => {
                  setCurrentImageIndex(index);
                }}
                title={`Image ${index + 1}`}
              >
                <img src={img.imageUrl} alt={`${bird.species} ${index + 1}`} />
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BirdDetailModal;
