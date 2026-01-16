import React, { useState, useEffect } from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';
import BirdCard from './components/BirdCard';
import CollectionProgress from './components/CollectionProgress';
import { Bird } from './types';
import { api } from './api';

const App: React.FC = () => {
  const [discoveredBirds, setDiscoveredBirds] = useState<Bird[]>([]);
  const [currentBird, setCurrentBird] = useState<Bird | null>(null);
  const [totalSpecies, setTotalSpecies] = useState<number>(25);

  useEffect(() => {
    // Fetch total species count on mount
    const fetchSpeciesCount = async () => {
      try {
        const species = await api.getSpeciesList();
        setTotalSpecies(species.length);
      } catch (error) {
        console.error('Failed to fetch species count:', error);
      }
    };
    fetchSpeciesCount();
  }, []);

  const handleBirdDiscovered = (bird: Bird) => {
    setCurrentBird(bird);
    
    // Add to collection if not already discovered
    const isNew = !discoveredBirds.some(b => b.species === bird.species);
    if (isNew) {
      setDiscoveredBirds(prev => [...prev, bird]);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🦅 Birdingdex</h1>
        <p className="subtitle">Catch 'em all! A Pokédex for Birds</p>
      </header>

      <main className="App-main">
        <div className="upload-section">
          <ImageUpload onBirdDiscovered={handleBirdDiscovered} />
        </div>

        {currentBird && (
          <div className="current-bird-section">
            <h2>Latest Discovery</h2>
            <BirdCard bird={currentBird} />
          </div>
        )}

        <div className="collection-section">
          <CollectionProgress 
            discoveredBirds={discoveredBirds}
            totalSpecies={totalSpecies}
          />
        </div>

        {discoveredBirds.length > 0 && (
          <div className="collection-grid">
            <h2>Your Collection</h2>
            <div className="bird-grid">
              {discoveredBirds.map((bird, index) => (
                <BirdCard key={index} bird={bird} compact />
              ))}
            </div>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>Built with React, TypeScript, FastAPI, and AI 🤖</p>
      </footer>
    </div>
  );
};

export default App;
