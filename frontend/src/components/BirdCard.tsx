import React, { useState } from 'react';
import { Bird, cleanBirdName } from '../types';
import { api } from '../api';
import './BirdCard.css';

interface BirdCardProps {
  bird: Bird;
  compact?: boolean;
  onCompactCardClick?: () => void;
  onAugmented?: (augmentedImageUrl: string) => void;
}

const BirdCard: React.FC<BirdCardProps> = ({ bird, compact = false, onCompactCardClick, onAugmented }) => {
  const [isAugmenting, setIsAugmenting] = useState(false);
  const [showAugmented, setShowAugmented] = useState(false);
  const [augmentedImage, setAugmentedImage] = useState<string | null>(null);

  const handleAugment = async (editType: string) => {
    setIsAugmenting(true);
    try {
      // Extract base64 from data URL
      const base64 = bird.imageUrl.split(',')[1];
      const result = await api.augmentImage(base64, editType);
      const augmentedUrl = `data:image/png;base64,${result.augmented_image}`;
      setAugmentedImage(augmentedUrl);
      setShowAugmented(true);
      // Notify parent component about augmented image
      if (onAugmented) {
        onAugmented(augmentedUrl);
      }
    } catch (err) {
      console.error('Augmentation error:', err);
    } finally {
      setIsAugmenting(false);
    }
  };

  const handleCardClick = () => {
    if (compact && onCompactCardClick) {
      onCompactCardClick();
    }
  };

  const displayImage = showAugmented && augmentedImage ? augmentedImage : bird.imageUrl;

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
                disabled={isAugmenting}
                className="augment-btn"
              >
                🎩 Hat
              </button>
              <button
                onClick={() => handleAugment('bowtie')}
                disabled={isAugmenting}
                className="augment-btn"
              >
                🎀 Bowtie
              </button>
              <button
                onClick={() => handleAugment('glasses')}
                disabled={isAugmenting}
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
