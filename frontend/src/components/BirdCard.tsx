import React, { useState } from 'react';
import { Bird, cleanBirdName } from '../types';
import { api } from '../api';
import './BirdCard.css';

interface BirdCardProps {
  bird: Bird;
  compact?: boolean;
  onCompactCardClick?: () => void;
  onAugmented?: (augmentedImageUrl: string, birdSpecies?: string) => void;
  onAugmentingStateChange?: (isAugmenting: boolean) => void;
}

const BirdCard: React.FC<BirdCardProps> = ({ bird, compact = false, onCompactCardClick, onAugmented, onAugmentingStateChange }) => {
  const [isAugmenting, setIsAugmenting] = useState(false);
  const [showAugmented, setShowAugmented] = useState(false);

  const handleAugment = async (editType: string) => {
    // Don't allow augmentation if this image already has one
    if (bird.images && bird.images.length > 0) {
      const lastImage = bird.images[bird.images.length - 1];
      if (lastImage.augmentedImages && lastImage.augmentedImages.length > 0) {
        console.log('Image already augmented');
        return;
      }
    }

    setIsAugmenting(true);
    onAugmentingStateChange?.(true);
    try {
      // Extract base64 from data URL
      const base64 = bird.imageUrl.split(',')[1];
      const result = await api.augmentImage(base64, editType);
      const augmentedUrl = `data:image/png;base64,${result.augmented_image}`;
      setShowAugmented(true);

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

  const handleCardClick = () => {
    if (compact && onCompactCardClick) {
      onCompactCardClick();
    }
  };

  // Check if current bird's latest image has been augmented
  const hasAugmentedImage = bird.images && bird.images.length > 0 &&
    bird.images[bird.images.length - 1].augmentedImages &&
    bird.images[bird.images.length - 1].augmentedImages!.length > 0;

  const augmentedImage = hasAugmentedImage ? bird.images![bird.images!.length - 1].augmentedImages![0] : null;
  const displayImage = showAugmented && augmentedImage ? augmentedImage : bird.imageUrl;
  const isAlreadyAugmented = hasAugmentedImage;

  return (
    <div
      className={`bird-card ${compact ? 'compact' : ''} ${compact ? 'clickable' : ''}`}
      onClick={handleCardClick}
    >
      <div className="bird-image-container">
        <img src={displayImage} alt={bird.species} className="bird-image" />
        {!compact && (
          <div className="confidence-badge">
            {(bird.confidence * 100).toFixed(1)}% confident
          </div>
        )}
        {isAugmenting && (
          <div className="augmenting-overlay">
            <div className="spinner"></div>
            <p>Applying magic...</p>
          </div>
        )}
      </div>

      <div className="bird-info">
        <h3 className="bird-species">{cleanBirdName(bird.species)}</h3>

        {!compact && bird.topPredictions && (
          <div className="predictions">
            <h4>Top Predictions:</h4>
            <ul>
              {bird.topPredictions.slice(0, 3).map((pred, index) => (
                <li key={index}>
                  {cleanBirdName(pred.species)}: {(pred.confidence * 100).toFixed(1)}%
                </li>
              ))}
            </ul>
          </div>
        )}

        {!compact && (
          <div className="augmentation-controls">
            <h4>Add Accessories:</h4>
            <div className="augment-buttons">
              <button
                onClick={() => handleAugment('hat')}
                disabled={isAugmenting || isAlreadyAugmented}
                className="augment-btn"
              >
                🎩 Hat
              </button>
              <button
                onClick={() => handleAugment('bowtie')}
                disabled={isAugmenting || isAlreadyAugmented}
                className="augment-btn"
              >
                🎀 Bowtie
              </button>
              <button
                onClick={() => handleAugment('glasses')}
                disabled={isAugmenting || isAlreadyAugmented}
                className="augment-btn"
              >
                🕶️ Glasses
              </button>
            </div>
            {augmentedImage && (
              <button
                onClick={() => setShowAugmented(!showAugmented)}
                className="toggle-btn"
              >
                {showAugmented ? 'Show Original' : 'Show Augmented'}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default BirdCard;
