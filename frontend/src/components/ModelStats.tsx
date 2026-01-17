import React, { useState, useEffect } from 'react';
import './ModelStats.css';
import { api } from '../api';

interface ModelMetrics {
  model_name?: string;
  training_date?: string;
  num_classes?: number;
  num_train_samples?: number;
  num_test_samples?: number;
  status?: string;
  message?: string;
  hyperparameters?: {
    num_epochs?: number | string;
    batch_size?: number | string;
    learning_rate?: number | string;
    optimizer?: string;
    warmup_steps?: number;
    weight_decay?: number;
  };
  results?: {
    test_accuracy?: number;
    test_precision?: number;
    test_recall?: number;
    test_f1?: number;
    train_loss?: number;
  };
  per_class_metrics?: {
    [key: string]: {
      accuracy: number;
      samples: number;
    };
  };
  class_names?: string[];
  error?: string;
}

const ModelStats: React.FC = () => {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        const data = await api.getModelMetrics();
        setMetrics(data);
        setError(null);
      } catch (err) {
        setError('Failed to load model metrics. The model may not be trained yet.');
        console.error('Error fetching model metrics:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, []);

  if (loading) {
    return (
      <div className="model-stats">
        <div className="loading">Loading model statistics...</div>
      </div>
    );
  }

  if (error || metrics?.error) {
    return (
      <div className="model-stats">
        <div className="error-message">
          <h3>⚠️ Model Not Trained</h3>
          <p>{error || metrics?.error}</p>
          <p className="training-instructions">
            To train the model, run the following command in the backend directory:
          </p>
          <code className="command">python train_model.py</code>
          <p className="note">
            This will fine-tune the Vision Transformer model on the OpenML bird dataset.
          </p>
        </div>
      </div>
    );
  }

  if (metrics?.status === 'Not trained - using demo model') {
    return (
      <div className="model-stats">
        <div className="demo-notice">
          <h2>📊 Model Statistics</h2>
          <div className="demo-message">
            <h3>🔧 Demo Mode Active</h3>
            <p>{metrics.message}</p>
            <p className="training-instructions">
              To train a fine-tuned model, run:
            </p>
            <code className="command">cd backend && python train_model.py</code>
            <div className="demo-info">
              <p><strong>Current Status:</strong> Using base Vision Transformer model</p>
              <p><strong>Species Count:</strong> {metrics.num_classes}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Calculate top and bottom performing classes
  const perClassMetrics = metrics?.per_class_metrics || {};
  const sortedClasses = Object.entries(perClassMetrics).sort(
    (a, b) => b[1].accuracy - a[1].accuracy
  );
  const topClasses = sortedClasses.slice(0, 5);
  const bottomClasses = sortedClasses.slice(-5).reverse();

  // Format date
  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  // Format percentage
  const formatPercent = (value?: number) => {
    if (value === undefined || value === null) return 'N/A';
    return `${(value * 100).toFixed(2)}%`;
  };

  return (
    <div className="model-stats">
      <div className="stats-header">
        <h2>📊 Model Statistics</h2>
        <p className="subtitle">Fine-tuned Vision Transformer Performance Metrics</p>
      </div>

      {/* Model Overview */}
      <div className="stats-section">
        <h3>🤖 Model Overview</h3>
        <div className="info-grid">
          <div className="info-item">
            <span className="label">Model Type:</span>
            <span className="value">{metrics?.model_name || 'N/A'}</span>
          </div>
          <div className="info-item">
            <span className="label">Training Date:</span>
            <span className="value">{formatDate(metrics?.training_date)}</span>
          </div>
          <div className="info-item">
            <span className="label">Number of Classes:</span>
            <span className="value">{metrics?.num_classes || 'N/A'}</span>
          </div>
          <div className="info-item">
            <span className="label">Training Samples:</span>
            <span className="value">{metrics?.num_train_samples || 'N/A'}</span>
          </div>
          <div className="info-item">
            <span className="label">Test Samples:</span>
            <span className="value">{metrics?.num_test_samples || 'N/A'}</span>
          </div>
        </div>
      </div>

      {/* Hyperparameters */}
      <div className="stats-section">
        <h3>⚙️ Hyperparameters</h3>
        <div className="info-grid">
          <div className="info-item">
            <span className="label">Epochs:</span>
            <span className="value">{metrics?.hyperparameters?.num_epochs || 'N/A'}</span>
          </div>
          <div className="info-item">
            <span className="label">Batch Size:</span>
            <span className="value">{metrics?.hyperparameters?.batch_size || 'N/A'}</span>
          </div>
          <div className="info-item">
            <span className="label">Learning Rate:</span>
            <span className="value">{metrics?.hyperparameters?.learning_rate || 'N/A'}</span>
          </div>
          <div className="info-item">
            <span className="label">Optimizer:</span>
            <span className="value">{metrics?.hyperparameters?.optimizer || 'N/A'}</span>
          </div>
          <div className="info-item">
            <span className="label">Warmup Steps:</span>
            <span className="value">{metrics?.hyperparameters?.warmup_steps || 'N/A'}</span>
          </div>
          <div className="info-item">
            <span className="label">Weight Decay:</span>
            <span className="value">{metrics?.hyperparameters?.weight_decay || 'N/A'}</span>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="stats-section">
        <h3>📈 Performance Metrics</h3>
        <div className="metrics-grid">
          <div className="metric-card accuracy">
            <div className="metric-label">Accuracy</div>
            <div className="metric-value">{formatPercent(metrics?.results?.test_accuracy)}</div>
            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{ width: formatPercent(metrics?.results?.test_accuracy) }}
              />
            </div>
          </div>
          <div className="metric-card precision">
            <div className="metric-label">Precision</div>
            <div className="metric-value">{formatPercent(metrics?.results?.test_precision)}</div>
            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{ width: formatPercent(metrics?.results?.test_precision) }}
              />
            </div>
          </div>
          <div className="metric-card recall">
            <div className="metric-label">Recall</div>
            <div className="metric-value">{formatPercent(metrics?.results?.test_recall)}</div>
            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{ width: formatPercent(metrics?.results?.test_recall) }}
              />
            </div>
          </div>
          <div className="metric-card f1">
            <div className="metric-label">F1 Score</div>
            <div className="metric-value">{formatPercent(metrics?.results?.test_f1)}</div>
            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{ width: formatPercent(metrics?.results?.test_f1) }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Per-Class Performance */}
      {topClasses.length > 0 && (
        <div className="stats-section">
          <h3>🏆 Top Performing Classes</h3>
          <div className="class-list">
            {topClasses.map(([className, classMetrics]) => (
              <div key={className} className="class-item">
                <span className="class-name">{className}</span>
                <div className="class-metrics">
                  <span className="class-accuracy">{formatPercent(classMetrics.accuracy)}</span>
                  <span className="class-samples">({classMetrics.samples} samples)</span>
                </div>
                <div className="class-bar">
                  <div
                    className="class-fill"
                    style={{ width: formatPercent(classMetrics.accuracy) }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {bottomClasses.length > 0 && (
        <div className="stats-section">
          <h3>📉 Needs Improvement</h3>
          <div className="class-list">
            {bottomClasses.map(([className, classMetrics]) => (
              <div key={className} className="class-item low-performance">
                <span className="class-name">{className}</span>
                <div className="class-metrics">
                  <span className="class-accuracy">{formatPercent(classMetrics.accuracy)}</span>
                  <span className="class-samples">({classMetrics.samples} samples)</span>
                </div>
                <div className="class-bar">
                  <div
                    className="class-fill"
                    style={{ width: formatPercent(classMetrics.accuracy) }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelStats;
