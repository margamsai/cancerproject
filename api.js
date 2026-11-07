import axios from 'axios';

// Base URL points to FastAPI backend
const API_BASE = 'http://127.0.0.1:8000/api';

const api = axios.create({
  baseURL: API_BASE,
});

export const getStatus = async () => (await api.get('/status')).data;
export const getDatasetSummary = async () => (await api.get('/dataset/summary')).data;
export const loadSample = async () => (await api.post('/dataset/sample')).data;

export const uploadDataset = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return (await api.post('/dataset/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })).data;
};

export const configureTarget = async (target_col, cols_to_drop = []) => {
  return (await api.post('/dataset/configure', { target_col, cols_to_drop })).data;
};

export const getEDA = async () => (await api.get('/dataset/eda')).data;

export const runPreprocess = async (params) => {
  return (await api.post('/preprocess', params)).data;
};

export const trainModels = async (model_name = 'All', perform_cv = true, cv_folds = 5) => {
  return (await api.post('/train', { model_name, perform_cv, cv_folds })).data;
};

export const getModels = async () => (await api.get('/models')).data;

export const getPredictFeatures = async () => (await api.get('/predict/features')).data;

export const predictSingle = async (features, model_name = null, patient_meta = null) => {
  return (await api.post('/predict/single', { features, model_name, patient_meta })).data;
};

export const predictBatch = async (file, model_name = null) => {
  const formData = new FormData();
  formData.append('file', file);
  if (model_name) formData.append('model_name', model_name);
  return (await api.post('/predict/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })).data;
};

export const getExportUrl = (model_name = null) => {
  return model_name ? `${API_BASE}/model/export?model_name=${encodeURIComponent(model_name)}` : `${API_BASE}/model/export`;
};

export default api;
