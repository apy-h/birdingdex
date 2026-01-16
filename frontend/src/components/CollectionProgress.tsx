import React from 'react';
import { Bird } from '../types';
import './CollectionProgress.css';

interface CollectionProgressProps {
  discoveredBirds: Bird[];
  totalSpecies: number;
}

const CollectionProgress: React.FC<CollectionProgressProps> = ({ 
  discoveredBirds, 
  totalSpecies 
}) => {
  const progress = (discoveredBirds.length / totalSpecies) * 100;

  return (
    <div className="collection-progress">
      <h2>Collection Progress</h2>
      
      <div className="progress-stats">
        <div className="stat">
          <div className="stat-value">{discoveredBirds.length}</div>
          <div className="stat-label">Discovered</div>
        </div>
        <div className="stat">
          <div className="stat-value">{totalSpecies}</div>
          <div className="stat-label">Total Species</div>
        </div>
        <div className="stat">
          <div className="stat-value">{progress.toFixed(1)}%</div>
          <div className="stat-label">Complete</div>
        </div>
      </div>

      <div className="progress-bar-container">
        <div 
          className="progress-bar-fill" 
          style={{ width: `${progress}%` }}
        >
          <span className="progress-text">{progress.toFixed(1)}%</span>
        </div>
      </div>

      {discoveredBirds.length === totalSpecies && (
        <div className="completion-message">
          🎉 Congratulations! You've discovered all species! 🎉
        </div>
      )}
    </div>
  );
};

export default CollectionProgress;
