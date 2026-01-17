import React, { useState, useEffect } from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';
import BirdCard from './components/BirdCard';
import CollectionProgress from './components/CollectionProgress';
import ModelStats from './components/ModelStats';
import { Bird } from './types';
import { api } from './api';

type Page = 'home' | 'stats';

const App: React.FC = () => {
  const [discoveredBirds, setDiscoveredBirds] = useState<Bird[]>([]);
  const [currentBird, setCurrentBird] = useState<Bird | null>(null);
  const [totalSpecies, setTotalSpecies] = useState<number>(25);
  const [currentPage, setCurrentPage] = useState<Page>('home');

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
        <div className="header-content">
          <img src="/logo.png" alt="Birdingdex Logo" className="header-logo" />
          <div className="header-text">
            <h1>Birdingdex</h1>
            <p className="subtitle">Catch 'em all! A Pokédex for Birds</p>
          </div>
        </div>
        <nav className="nav-menu">
          <button
            className={`nav-button ${currentPage === 'home' ? 'active' : ''}`}
            onClick={() => setCurrentPage('home')}
          >
            🏠 Home
          </button>
          <button
            className={`nav-button ${currentPage === 'stats' ? 'active' : ''}`}
            onClick={() => setCurrentPage('stats')}
          >
            📊 Model Stats
          </button>
        </nav>
      </header>

      <main className="App-main">
        {currentPage === 'home' ? (
          <>
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
          </>
        ) : (
          <ModelStats />
        )}
      </main>

      <footer className="App-footer">
        <p>Built with React, TypeScript, FastAPI, and AI 🤖</p>
      </footer>
    </div>
  );
};

export default App;
