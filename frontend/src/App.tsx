import React, { useState, useEffect } from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';
import BirdCard from './components/BirdCard';
import BirdDetailModal from './components/BirdDetailModal';
import CollectionProgress from './components/CollectionProgress';
import ModelStats from './components/ModelStats';
import { Bird, BirdImage } from './types';
import { api } from './api';

type Page = 'home' | 'stats';

const App: React.FC = () => {
  const [discoveredBirds, setDiscoveredBirds] = useState<Bird[]>([]);
  const [currentBird, setCurrentBird] = useState<Bird | null>(null);
  const [selectedBirdForModal, setSelectedBirdForModal] = useState<Bird | null>(null);
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

    // Add to collection or update existing species
    setDiscoveredBirds(prev => {
      const existingIndex = prev.findIndex(b => b.species === bird.species);

      if (existingIndex >= 0) {
        // Update existing bird with new image
        const updated = [...prev];
        const existingBird = updated[existingIndex];

        // Initialize images array if needed
        if (!existingBird.images) {
          existingBird.images = [{
            imageUrl: existingBird.imageUrl,
            confidence: existingBird.confidence,
            topPredictions: existingBird.topPredictions,
            augmentedImages: []
          }];
        }

        // Add new image
        existingBird.images.push({
          imageUrl: bird.imageUrl,
          confidence: bird.confidence,
          topPredictions: bird.topPredictions,
          augmentedImages: []
        });

        // Update main bird info to latest
        updated[existingIndex] = {
          ...existingBird,
          imageUrl: bird.imageUrl,
          confidence: bird.confidence,
          topPredictions: bird.topPredictions
        };

        return updated;
      } else {
        // New species
        const newBird: Bird = {
          ...bird,
          images: [{
            imageUrl: bird.imageUrl,
            confidence: bird.confidence,
            topPredictions: bird.topPredictions,
            augmentedImages: []
          }]
        };
        return [...prev, newBird];
      }
    });
  };

  const handleAugmentImage = (augmentedImageUrl: string) => {
    // Add augmented image to the current bird's latest image
    if (currentBird) {
      setDiscoveredBirds(prev => {
        const updated = prev.map(bird => {
          if (bird.species === currentBird.species && bird.images) {
            const lastIndex = bird.images.length - 1;
            if (!bird.images[lastIndex].augmentedImages) {
              bird.images[lastIndex].augmentedImages = [];
            }
            bird.images[lastIndex].augmentedImages!.push(augmentedImageUrl);
          }
          return bird;
        });
        return updated;
      });
    }
  };

  const openBirdDetail = (bird: Bird) => {
    // Ensure the bird has images array for the modal
    if (!bird.images || bird.images.length === 0) {
      const birdWithImages: Bird = {
        ...bird,
        images: [{
          imageUrl: bird.imageUrl,
          confidence: bird.confidence,
          topPredictions: bird.topPredictions,
          augmentedImages: []
        }]
      };
      setSelectedBirdForModal(birdWithImages);
    } else {
      setSelectedBirdForModal(bird);
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
                <BirdCard
                  bird={currentBird}
                  onAugmented={handleAugmentImage}
                />
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
                    <BirdCard
                      key={index}
                      bird={bird}
                      compact
                      onCompactCardClick={() => openBirdDetail(bird)}
                      onAugmented={handleAugmentImage}
                    />
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <ModelStats />
        )}
      </main>

      <BirdDetailModal
        bird={selectedBirdForModal || currentBird || { species: '', confidence: 0, imageUrl: '' }}
        isOpen={selectedBirdForModal !== null}
        onClose={() => setSelectedBirdForModal(null)}
      />

      <footer className="App-footer">
        <p>Built with React, TypeScript, FastAPI, and AI 🤖</p>
      </footer>
    </div>
  );
};

export default App;
