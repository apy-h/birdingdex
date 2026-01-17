import axios from 'axios';
import { ClassificationResult, UploadResponse, AugmentationResponse } from './types';

const API_BASE_URL = 'http://localhost:8000';

export const api = {
  async uploadImage(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  },

  async classifyBird(file: File): Promise<ClassificationResult> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`${API_BASE_URL}/api/classify`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  },

  async augmentImage(imageBase64: string, editType: string): Promise<AugmentationResponse> {
    const response = await axios.post(`${API_BASE_URL}/api/augment`, {
      image_base64: imageBase64,
      edit_type: editType,
    });

    return response.data;
  },

  async getSpeciesList(): Promise<string[]> {
    const response = await axios.get(`${API_BASE_URL}/api/species`);
    return response.data.species;
  },

  async getModelMetrics(): Promise<any> {
    const response = await axios.get(`${API_BASE_URL}/api/model/metrics`);
    return response.data;
  },
};
