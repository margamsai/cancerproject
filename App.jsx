import React, { useState, useEffect } from 'react';
import {
  Activity,
  Database,
  Cpu,
  Target,
  Upload,
  RefreshCw,
  Play,
  Download,
  AlertCircle,
  CheckCircle2,
  BarChart3,
  Layers,
  ShieldAlert,
  Printer,
  FileText,
  FileSpreadsheet
} from 'lucide-react';
import * as api from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('data');
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [notification, setNotification] = useState(null);

  // Data State
  const [summary, setSummary] = useState(null);
  const [eda, setEda] = useState(null);
  const [targetCol, setTargetCol] = useState('');

  // Preprocess State
  const [prepConfig, setPrepConfig] = useState({
    scaling_strategy: 'StandardScaler (Z-Score)',
    num_impute_strategy: 'median',
    test_size: 0.2,
    stratify: true,
    drop_duplicates: true,
  });

  // Model State
  const [modelResults, setModelResults] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);

  // Prediction State
  const [predictFeatures, setPredictFeatures] = useState([]);
  const [patientInputs, setPatientInputs] = useState({});
  const [patientMeta, setPatientMeta] = useState({
    patient_id: 'PT-101',
    patient_name: 'Jane Doe',
    age: '52',
    gender: 'Female',
    physician: 'Dr. Sarah Lin'
  });
  const [clinicalReport, setClinicalReport] = useState(null);
  const [predictModel, setPredictModel] = useState('');
  const [batchResult, setBatchResult] = useState(null);

  // Toast notification
  const notify = (msg, type = 'success') => {
    setNotification({ msg, type });
    setTimeout(() => setNotification(null), 4000);
  };

  // Initial load
  const fetchInitialData = async () => {
    try {
      const [st, sum, edaData] = await Promise.all([
        api.getStatus(),
        api.getDatasetSummary(),
        api.getEDA()
      ]);
      setStatus(st);
      setSummary(sum);
      setTargetCol(sum.target_col || '');
      setEda(edaData);

      const pFeat = await api.getPredictFeatures();
      setPredictFeatures(pFeat.features || []);

      const initialInputs = {};
      (pFeat.features || []).forEach(f => {
        initialInputs[f.name] = f.mean;
      });
      setPatientInputs(initialInputs);

      const modelsData = await api.getModels();
      setModelResults(modelsData);
      if (modelsData.leaderboard && modelsData.leaderboard.length > 0) {
        setSelectedModel(modelsData.leaderboard[0].Model);
        setPredictModel(modelsData.leaderboard[0].Model);
      }
    } catch (err) {
      console.error(err);
      notify('Unable to connect to backend server. Make sure it is running.', 'error');
    }
  };

  useEffect(() => {
    fetchInitialData();
  }, []);

  // Upload dataset
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setLoading(true);
    try {
      const sum = await api.uploadDataset(file);
      setSummary(sum);
      setTargetCol(sum.target_col);
      const [st, edaData, pFeat] = await Promise.all([
        api.getStatus(),
        api.getEDA(),
        api.getPredictFeatures()
      ]);
      setStatus(st);
      setEda(edaData);
      setPredictFeatures(pFeat.features || []);
      notify(`Dataset loaded: ${file.name}`);
    } catch (err) {
      notify('Upload failed: ' + (err.response?.data?.detail || err.message), 'error');
    } finally {
      setLoading(false);
    }
  };

  // Reset sample dataset
  const handleLoadSample = async () => {
    setLoading(true);
    try {
      await api.loadSample();
      await fetchInitialData();
      notify('Reset to default sample dataset');
    } catch (err) {
      notify('Failed to load sample dataset', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Target change
  const handleTargetChange = async (newTarget) => {
    setTargetCol(newTarget);
    try {
      await api.configureTarget(newTarget);
      const [st, sum, edaData] = await Promise.all([
        api.getStatus(),
        api.getDatasetSummary(),
        api.getEDA()
      ]);
      setStatus(st);
      setSummary(sum);
      setEda(edaData);
      notify(`Target updated to ${newTarget}`);
    } catch (err) {
      notify('Error updating target: ' + err.message, 'error');
    }
  };

  // Preprocess
  const handlePreprocess = async () => {
    setLoading(true);
    try {
      await api.runPreprocess(prepConfig);
      const [st, pFeat] = await Promise.all([api.getStatus(), api.getPredictFeatures()]);
      setStatus(st);
      setPredictFeatures(pFeat.features || []);
      notify('Dataset preprocessed successfully');
    } catch (err) {
      notify('Preprocessing failed: ' + (err.response?.data?.detail || err.message), 'error');
    } finally {
      setLoading(false);
    }
  };

  // Train models
  const handleTrainAll = async () => {
    setLoading(true);
    try {
      const res = await api.trainModels('All', true, 5);
      setModelResults(res);
      if (res.leaderboard && res.leaderboard.length > 0) {
        setSelectedModel(res.leaderboard[0].Model);
        setPredictModel(res.leaderboard[0].Model);
      }
      const st = await api.getStatus();
      setStatus(st);
      notify('All models trained successfully with 5-Fold Cross Validation');
    } catch (err) {
      notify('Training failed: ' + (err.response?.data?.detail || err.message), 'error');
    } finally {
      setLoading(false);
    }
  };

  // Single prediction report
  const handleGenerateReport = async () => {
    setLoading(true);
    try {
      const report = await api.predictSingle(patientInputs, predictModel, patientMeta);
      setClinicalReport(report);
      notify(`Prediction report generated (${report.report_id})`);
    } catch (err) {
      notify('Prediction error: ' + (err.response?.data?.detail || err.message), 'error');
    } finally {
      setLoading(false);
    }
  };

  // Preset values
  const handleLoadPreset = (type) => {
    const updated = {};
    predictFeatures.forEach(f => {
      if (type === 'benign') updated[f.name] = f.benign_preset;
      else if (type === 'malignant') updated[f.name] = f.malignant_preset;
      else updated[f.name] = f.mean;
    });
    setPatientInputs(updated);
    notify(`Loaded ${type.toUpperCase()} preset values`);
  };

  // Download text report
  const handleDownloadTextReport = () => {
    if (!clinicalReport) return;
    const textContent = `================================================================================
CANCER PREDICTION REPORT
Report ID: ${clinicalReport.report_id}
Date: ${clinicalReport.timestamp}
================================================================================

PATIENT INFORMATION:
Patient ID: ${clinicalReport.patient_info?.patient_id}
Patient Name: ${clinicalReport.patient_info?.patient_name}
Age: ${clinicalReport.patient_info?.age}
Gender: ${clinicalReport.patient_info?.gender}
Physician: ${clinicalReport.patient_info?.referring_physician}

PREDICTION SUMMARY:
Classification: ${clinicalReport.diagnosis}
Confidence: ${clinicalReport.confidence}%
Risk Level: ${clinicalReport.risk_tier}
Model Used: ${clinicalReport.model_used}

ANALYSIS:
${clinicalReport.narrative}

RECOMMENDATIONS:
${clinicalReport.recommendations?.map((r, i) => `${i + 1}. ${r}`).join('\n')}

================================================================================
DISCLAIMER: Educational machine learning project. Predictions must not be used for 
medical diagnosis or treatment decisions.
================================================================================
`;
    const blob = new Blob([textContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${clinicalReport.report_id}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Batch prediction upload
  const handleBatchUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setLoading(true);
    try {
      const res = await api.predictBatch(file, predictModel);
      setBatchResult(res);
      notify(`Batch prediction completed for ${res.total_records} records`);
    } catch (err) {
      notify('Batch prediction error: ' + (err.response?.data?.detail || err.message), 'error');
    } finally {
      setLoading(false);
    }
  };

  const currentModelData = modelResults?.models?.[selectedModel];

  return (
    <div className="app-container">
      {/* SIDEBAR NAVIGATION */}
      <aside className="sidebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '2rem' }}>
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #2563eb, #06b6d4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1.25rem'
          }}>
            🎗️
          </div>
          <div>
            <div style={{ fontWeight: 800, fontSize: '1.15rem', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              OncoML
              <span className="pulse-dot" />
            </div>
            <div style={{ fontSize: '0.7rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
              Cancer Prediction System
            </div>
          </div>
        </div>

        <nav style={{ flex: 1 }}>
          <div
            className={`nav-item ${activeTab === 'data' ? 'active' : ''}`}
            onClick={() => setActiveTab('data')}
          >
            <Database size={18} />
            <span>1. Dataset</span>
          </div>

          <div
            className={`nav-item ${activeTab === 'preprocess' ? 'active' : ''}`}
            onClick={() => setActiveTab('preprocess')}
          >
            <Layers size={18} />
            <span>2. Preprocess</span>
          </div>

          <div
            className={`nav-item ${activeTab === 'benchmark' ? 'active' : ''}`}
            onClick={() => setActiveTab('benchmark')}
          >
            <Cpu size={18} />
            <span>3. Train Models</span>
          </div>

          <div
            className={`nav-item ${activeTab === 'predict' ? 'active' : ''}`}
            onClick={() => setActiveTab('predict')}
          >
            <FileText size={18} />
            <span>4. Prediction & Report</span>
          </div>
        </nav>

        {/* WORKSPACE STATUS */}
        <div style={{
          background: '#0b1120',
          border: '1px solid var(--border-subtle)',
          borderRadius: '10px',
          padding: '1rem',
          marginTop: '1rem'
        }}>
          <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: '#64748b', fontWeight: 700, marginBottom: '0.5rem' }}>
            Status
          </div>

          {status?.dataset_loaded ? (
            <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                <span className="badge badge-green">Ready</span>
                <span style={{ color: '#38bdf8', fontWeight: 600 }}>{status.rows} rows</span>
              </div>
              <div style={{ fontSize: '0.72rem', color: '#64748b' }}>
                Target: <code style={{ color: '#e2e8f0' }}>{status.target_col}</code>
              </div>
            </div>
          ) : (
            <div style={{ fontSize: '0.8rem', color: '#64748b' }}>No dataset loaded</div>
          )}

          {status?.trained_models?.length > 0 && (
            <div style={{ marginTop: '0.4rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="badge badge-blue">{status.trained_models.length} Trained</span>
              <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>5-Fold CV</span>
            </div>
          )}

          <button
            onClick={handleLoadSample}
            disabled={loading}
            className="btn btn-secondary"
            style={{ width: '100%', marginTop: '0.8rem', fontSize: '0.74rem', padding: '0.4rem' }}
          >
            <RefreshCw size={12} />
            <span>Reset Sample Data</span>
          </button>
        </div>
      </aside>

      {/* MAIN CONTENT AREA */}
      <main className="main-content">
        {/* TOP HEADER */}
        <header className="top-header">
          <div>
            <h1 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#ffffff' }}>
              Cancer Prediction & Machine Learning Platform
            </h1>
            <p style={{ fontSize: '0.82rem', color: '#94a3b8', marginTop: '0.15rem' }}>
              Train machine learning models and generate patient risk reports
            </p>
          </div>

          <div>
            <div className="badge badge-amber" style={{ padding: '0.4rem 0.8rem', fontSize: '0.74rem' }}>
              <ShieldAlert size={14} />
              <span>Educational Project &bull; Not for Clinical Decisions</span>
            </div>
          </div>
        </header>

        {/* NOTIFICATION TOAST */}
        {notification && (
          <div style={{
            position: 'fixed',
            top: '20px',
            right: '25px',
            background: notification.type === 'error' ? '#ef4444' : '#10b981',
            color: '#ffffff',
            padding: '0.75rem 1.3rem',
            borderRadius: '8px',
            boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
            zIndex: 9999,
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontWeight: 600,
            fontSize: '0.86rem'
          }}>
            {notification.type === 'error' ? <AlertCircle size={16} /> : <CheckCircle2 size={16} />}
            <span>{notification.msg}</span>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 1: DATASET */}
        {/* ========================================================================= */}
        {activeTab === 'data' && (
          <div>
            {/* KPI Cards */}
            <div className="metrics-grid">
              <div className="metric-box">
                <div className="metric-label">
                  <span>Total Records</span>
                  <Database size={15} color="#38bdf8" />
                </div>
                <div className="metric-value">{summary?.rows ? summary.rows.toLocaleString() : 0}</div>
              </div>

              <div className="metric-box">
                <div className="metric-label">
                  <span>Features</span>
                  <Activity size={15} color="#34d399" />
                </div>
                <div className="metric-value">{summary?.cols || 0}</div>
              </div>

              <div className="metric-box">
                <div className="metric-label">
                  <span>Missing Values</span>
                  <CheckCircle2 size={15} color="#fbbf24" />
                </div>
                <div className="metric-value" style={{ color: summary?.total_missing === 0 ? '#34d399' : '#f59e0b' }}>
                  {summary?.total_missing ?? 0}
                </div>
              </div>

              <div className="metric-box">
                <div className="metric-label">
                  <span>Target Column</span>
                  <Target size={15} color="#818cf8" />
                </div>
                <div className="metric-value" style={{ fontSize: '1.2rem', color: '#60a5fa' }}>
                  {summary?.target_col || 'None'}
                </div>
              </div>
            </div>

            {/* Upload & Target Configuration */}
            <div className="card">
              <div className="card-title">
                <Upload size={17} color="#38bdf8" />
                <span>Upload Dataset & Select Target Column</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '1.5rem', alignItems: 'center' }}>
                <div>
                  <label className="form-label">Upload CSV or Excel File</label>
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={handleFileUpload}
                    className="form-input"
                    style={{ padding: '0.55rem' }}
                  />
                  <div style={{ fontSize: '0.74rem', color: '#94a3b8', marginTop: '0.4rem' }}>
                    Current: <b style={{ color: '#f8fafc' }}>{summary?.name}</b>
                  </div>
                </div>

                <div>
                  <label className="form-label">Target / Diagnosis Column</label>
                  <select
                    value={targetCol}
                    onChange={(e) => handleTargetChange(e.target.value)}
                    className="form-select"
                  >
                    {(summary?.all_columns || []).map((col) => (
                      <option key={col} value={col}>
                        {col} {col === summary?.target_col ? '(Detected)' : ''}
                      </option>
                    ))}
                  </select>
                  <div style={{ fontSize: '0.74rem', color: '#64748b', marginTop: '0.35rem' }}>
                    The models will be trained to predict this label.
                  </div>
                </div>
              </div>
            </div>

            {/* Target Class Distribution */}
            {eda?.target_distribution && eda.target_distribution.length > 0 && (
              <div className="card">
                <div className="card-title">
                  <BarChart3 size={17} color="#34d399" />
                  <span>Target Class Distribution</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {eda.target_distribution.map((item, idx) => (
                    <div key={idx}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.86rem', marginBottom: '0.3rem' }}>
                        <span style={{ fontWeight: 700, color: '#ffffff' }}>{item.class}</span>
                        <span style={{ color: '#94a3b8', fontFamily: 'JetBrains Mono' }}>
                          {item.count} cases ({item.percentage}%)
                        </span>
                      </div>
                      <div className="progress-bar-bg" style={{ height: '8px' }}>
                        <div
                          className="progress-bar-fill"
                          style={{
                            width: `${item.percentage}%`,
                            background: idx === 0 ? '#10b981' : '#ef4444'
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Data Preview Table */}
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div className="card-title" style={{ margin: 0 }}>
                  <FileSpreadsheet size={17} color="#38bdf8" />
                  <span>Data Preview (First 8 Rows)</span>
                </div>
                <div style={{ fontSize: '0.74rem', color: '#64748b' }}>
                  {summary?.rows} rows &bull; {summary?.cols} columns
                </div>
              </div>

              <div style={{ overflowX: 'auto', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      {summary?.preview && summary.preview.length > 0 &&
                        Object.keys(summary.preview[0]).slice(0, 8).map((col) => (
                          <th key={col}>{col}</th>
                        ))}
                    </tr>
                  </thead>
                  <tbody>
                    {summary?.preview &&
                      summary.preview.slice(0, 8).map((row, idx) => (
                        <tr key={idx}>
                          {Object.keys(row).slice(0, 8).map((col) => (
                            <td key={col} style={{ fontFamily: typeof row[col] === 'number' ? 'JetBrains Mono' : 'inherit' }}>
                              {typeof row[col] === 'number' ? row[col].toFixed(4) : String(row[col])}
                            </td>
                          ))}
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: PREPROCESSING */}
        {/* ========================================================================= */}
        {activeTab === 'preprocess' && (
          <div>
            <div className="card">
              <div className="card-title">
                <Layers size={17} color="#38bdf8" />
                <span>Preprocessing Configuration</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.4rem', marginBottom: '1.4rem' }}>
                <div>
                  <label className="form-label">Feature Scaling</label>
                  <select
                    value={prepConfig.scaling_strategy}
                    onChange={(e) => setPrepConfig({ ...prepConfig, scaling_strategy: e.target.value })}
                    className="form-select"
                  >
                    <option value="StandardScaler (Z-Score)">StandardScaler (Z-Score)</option>
                    <option value="MinMaxScaler (0 - 1)">MinMaxScaler (0 - 1)</option>
                    <option value="RobustScaler (Outlier-resistant)">RobustScaler (Outlier-resistant)</option>
                    <option value="None (No Scaling)">None (Raw Features)</option>
                  </select>
                </div>

                <div>
                  <label className="form-label">Missing Value Imputation</label>
                  <select
                    value={prepConfig.num_impute_strategy}
                    onChange={(e) => setPrepConfig({ ...prepConfig, num_impute_strategy: e.target.value })}
                    className="form-select"
                  >
                    <option value="median">Median</option>
                    <option value="mean">Mean</option>
                    <option value="mode">Mode</option>
                    <option value="constant (0)">Constant Zero</option>
                  </select>
                </div>

                <div>
                  <label className="form-label">Test Split Ratio: {Math.round(prepConfig.test_size * 100)}%</label>
                  <input
                    type="range"
                    min="10"
                    max="40"
                    step="5"
                    value={prepConfig.test_size * 100}
                    onChange={(e) => setPrepConfig({ ...prepConfig, test_size: Number(e.target.value) / 100 })}
                    style={{ width: '100%', marginTop: '0.5rem' }}
                  />
                </div>
              </div>

              <div style={{ display: 'flex', gap: '2rem', marginBottom: '1.5rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', cursor: 'pointer', fontSize: '0.86rem' }}>
                  <input
                    type="checkbox"
                    checked={prepConfig.stratify}
                    onChange={(e) => setPrepConfig({ ...prepConfig, stratify: e.target.checked })}
                  />
                  <span>Stratified Split (preserves class ratio)</span>
                </label>

                <label style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', cursor: 'pointer', fontSize: '0.86rem' }}>
                  <input
                    type="checkbox"
                    checked={prepConfig.drop_duplicates}
                    onChange={(e) => setPrepConfig({ ...prepConfig, drop_duplicates: e.target.checked })}
                  />
                  <span>Remove duplicate records</span>
                </label>
              </div>

              <button
                onClick={handlePreprocess}
                disabled={loading}
                className="btn btn-primary"
                style={{ width: '100%', padding: '0.8rem' }}
              >
                <Play size={16} />
                <span>Run Preprocessing Pipeline</span>
              </button>
            </div>

            {status?.preprocessed && (
              <div className="card">
                <div className="card-title">
                  <CheckCircle2 size={17} color="#10b981" />
                  <span>Preprocessed Partitions</span>
                </div>
                <div className="metrics-grid">
                  <div className="metric-box">
                    <div className="metric-label">Training Records</div>
                    <div className="metric-value">{status.train_size}</div>
                  </div>
                  <div className="metric-box">
                    <div className="metric-label">Testing Records</div>
                    <div className="metric-value">{status.test_size}</div>
                  </div>
                  <div className="metric-box">
                    <div className="metric-label">Processed Features</div>
                    <div className="metric-value">{status.feature_count}</div>
                  </div>
                  <div className="metric-box">
                    <div className="metric-label">Data Leakage Guard</div>
                    <div className="metric-value" style={{ fontSize: '1.2rem', color: '#10b981' }}>Active</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 3: MODEL BENCHMARK */}
        {/* ========================================================================= */}
        {activeTab === 'benchmark' && (
          <div>
            <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontWeight: 800, fontSize: '1.15rem', color: '#ffffff' }}>
                  Train & Compare Models
                </div>
                <div style={{ fontSize: '0.82rem', color: '#9ca3af', marginTop: '0.15rem' }}>
                  Trains 6 algorithms (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost)
                </div>
              </div>

              <button
                onClick={handleTrainAll}
                disabled={loading}
                className="btn btn-primary"
              >
                <Play size={16} />
                <span>Train All Models</span>
              </button>
            </div>

            {/* Leaderboard */}
            {modelResults?.leaderboard && modelResults.leaderboard.length > 0 && (
              <>
                <div className="card">
                  <div className="card-title">
                    <BarChart3 size={17} color="#38bdf8" />
                    <span>Model Performance Comparison</span>
                  </div>

                  <div style={{ overflowX: 'auto', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Model</th>
                          <th>Accuracy</th>
                          <th>Precision</th>
                          <th>Recall</th>
                          <th>F1 Score</th>
                          <th>5-Fold CV</th>
                          <th>Time (s)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {modelResults.leaderboard.map((row, idx) => (
                          <tr
                            key={idx}
                            onClick={() => setSelectedModel(row.Model)}
                            style={{
                              cursor: 'pointer',
                              background: selectedModel === row.Model ? 'rgba(37, 99, 235, 0.15)' : 'transparent',
                              borderLeft: selectedModel === row.Model ? '4px solid #38bdf8' : 'none'
                            }}
                          >
                            <td style={{ fontWeight: 700 }}>{idx + 1}</td>
                            <td style={{ fontWeight: 700, color: '#ffffff' }}>{row.Model}</td>
                            <td><span className="badge badge-green" style={{ fontFamily: 'JetBrains Mono' }}>{row['Accuracy (%)']}%</span></td>
                            <td style={{ fontFamily: 'JetBrains Mono' }}>{row['Precision (%)']}%</td>
                            <td style={{ fontWeight: 700, color: '#38bdf8', fontFamily: 'JetBrains Mono' }}>{row['Recall (%)']}%</td>
                            <td style={{ fontFamily: 'JetBrains Mono' }}>{row['F1 Score (%)']}%</td>
                            <td style={{ fontFamily: 'JetBrains Mono' }}>{row['CV Accuracy (%)'] ? `${row['CV Accuracy (%)']}%` : 'N/A'}</td>
                            <td style={{ fontFamily: 'JetBrains Mono', color: '#94a3b8' }}>{row['Training Time (s)']}s</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Model Details */}
                {currentModelData && (
                  <div className="card">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
                      <div className="card-title" style={{ margin: 0 }}>
                        <Cpu size={17} color="#818cf8" />
                        <span>Details: {selectedModel}</span>
                      </div>
                      <a
                        href={api.getExportUrl(selectedModel)}
                        download
                        className="btn btn-secondary"
                        style={{ fontSize: '0.78rem', padding: '0.4rem 0.8rem' }}
                      >
                        <Download size={14} />
                        <span>Export Model (.joblib)</span>
                      </a>
                    </div>

                    <div className="metrics-grid">
                      <div className="metric-box">
                        <div className="metric-label">Accuracy</div>
                        <div className="metric-value">{(currentModelData.accuracy * 100).toFixed(1)}%</div>
                      </div>
                      <div className="metric-box">
                        <div className="metric-label">Precision</div>
                        <div className="metric-value">{(currentModelData.precision * 100).toFixed(1)}%</div>
                      </div>
                      <div className="metric-box">
                        <div className="metric-label">Recall</div>
                        <div className="metric-value" style={{ color: '#34d399' }}>
                          {(currentModelData.recall * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="metric-box">
                        <div className="metric-label">ROC AUC</div>
                        <div className="metric-value" style={{ color: '#60a5fa' }}>
                          {currentModelData.roc_auc ? currentModelData.roc_auc.toFixed(3) : 'N/A'}
                        </div>
                      </div>
                    </div>

                    {/* Simple Confusion Matrix Table */}
                    {currentModelData.confusion_matrix && (
                      <div style={{ marginTop: '1.2rem' }}>
                        <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.6rem' }}>
                          Confusion Matrix
                        </div>
                        <table className="data-table" style={{ maxWidth: '350px' }}>
                          <thead>
                            <tr>
                              <th></th>
                              <th>Predicted Negative</th>
                              <th>Predicted Positive</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td style={{ fontWeight: 700 }}>Actual Negative</td>
                              <td style={{ color: '#34d399', fontWeight: 700, fontFamily: 'JetBrains Mono' }}>
                                {currentModelData.confusion_matrix[0]?.[0]} (TN)
                              </td>
                              <td style={{ color: '#fbbf24', fontFamily: 'JetBrains Mono' }}>
                                {currentModelData.confusion_matrix[0]?.[1]} (FP)
                              </td>
                            </tr>
                            <tr>
                              <td style={{ fontWeight: 700 }}>Actual Positive</td>
                              <td style={{ color: '#f87171', fontFamily: 'JetBrains Mono' }}>
                                {currentModelData.confusion_matrix[1]?.[0]} (FN)
                              </td>
                              <td style={{ color: '#38bdf8', fontWeight: 700, fontFamily: 'JetBrains Mono' }}>
                                {currentModelData.confusion_matrix[1]?.[1]} (TP)
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 4: PREDICTION & REPORT */}
        {/* ========================================================================= */}
        {activeTab === 'predict' && (
          <div>
            {clinicalReport ? (
              /* REPORT VIEW */
              <div className="report-paper">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', borderBottom: '1px solid rgba(255, 255, 255, 0.1)', paddingBottom: '1rem' }}>
                  <button
                    onClick={() => setClinicalReport(null)}
                    className="btn btn-secondary"
                  >
                    ← Edit Inputs / New Prediction
                  </button>

                  <div style={{ display: 'flex', gap: '0.6rem' }}>
                    <button
                      onClick={handleDownloadTextReport}
                      className="btn btn-secondary"
                    >
                      <Download size={15} />
                      <span>Download Text</span>
                    </button>
                    <button
                      onClick={() => window.print()}
                      className="btn btn-primary"
                    >
                      <Printer size={15} />
                      <span>Print / PDF</span>
                    </button>
                  </div>
                </div>

                {/* Report Header */}
                <div className="report-header">
                  <div>
                    <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#ffffff' }}>
                      Cancer Prediction Assessment Report
                    </div>
                    <div style={{ fontSize: '0.82rem', color: '#94a3b8', marginTop: '0.2rem' }}>
                      Educational Machine Learning Model Prediction
                    </div>
                  </div>

                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '0.68rem', color: '#64748b', fontWeight: 700, textTransform: 'uppercase' }}>
                      Report ID
                    </div>
                    <div style={{ fontFamily: 'JetBrains Mono', fontWeight: 700, color: '#38bdf8', fontSize: '1rem' }}>
                      {clinicalReport.report_id}
                    </div>
                    <div style={{ fontSize: '0.74rem', color: '#94a3b8' }}>
                      {clinicalReport.timestamp}
                    </div>
                  </div>
                </div>

                {/* Patient Information */}
                <div className="report-patient-grid">
                  <div className="report-patient-cell">
                    <div className="label">Patient Name</div>
                    <div className="val">{clinicalReport.patient_info?.patient_name || 'N/A'}</div>
                  </div>
                  <div className="report-patient-cell">
                    <div className="label">Patient ID</div>
                    <div className="val" style={{ fontFamily: 'JetBrains Mono' }}>{clinicalReport.patient_info?.patient_id || 'N/A'}</div>
                  </div>
                  <div className="report-patient-cell">
                    <div className="label">Age & Gender</div>
                    <div className="val">{clinicalReport.patient_info?.age} Yrs &bull; {clinicalReport.patient_info?.gender}</div>
                  </div>
                  <div className="report-patient-cell">
                    <div className="label">Physician</div>
                    <div className="val">{clinicalReport.patient_info?.referring_physician || 'N/A'}</div>
                  </div>
                </div>

                {/* Diagnosis Banner */}
                <div className={clinicalReport.is_malignant ? 'report-diagnosis-banner-malignant' : 'report-diagnosis-banner-benign'}>
                  <div style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 700, opacity: 0.85 }}>
                    Predicted Classification
                  </div>
                  <div style={{ fontSize: '2.2rem', fontWeight: 800, margin: '0.3rem 0' }}>
                    {clinicalReport.diagnosis}
                  </div>
                  <div style={{ fontSize: '0.95rem', fontWeight: 600 }}>
                    Confidence: <b>{clinicalReport.confidence}%</b> &bull; Risk Tier: <b>{clinicalReport.risk_tier}</b> &bull; Model: <b>{clinicalReport.model_used}</b>
                  </div>

                  {clinicalReport.probabilities && (
                    <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', marginTop: '1rem' }}>
                      {Object.entries(clinicalReport.probabilities).map(([cls, pct]) => (
                        <div key={cls} style={{ fontSize: '0.88rem' }}>
                          <span style={{ opacity: 0.85 }}>{cls}: </span>
                          <b style={{ fontFamily: 'JetBrains Mono', fontSize: '1rem' }}>{pct}%</b>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Narrative Summary */}
                <div style={{ marginBottom: '1.5rem' }}>
                  <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.5rem' }}>
                    Analysis Summary
                  </div>
                  <div className="report-narrative-box">
                    {clinicalReport.narrative}
                  </div>
                </div>

                {/* Abnormal Features Table */}
                {clinicalReport.abnormal_features && clinicalReport.abnormal_features.length > 0 && (
                  <div style={{ marginBottom: '1.5rem' }}>
                    <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.6rem' }}>
                      Notable Feature Values
                    </div>

                    <div style={{ overflowX: 'auto', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Feature</th>
                            <th>Patient Value</th>
                            <th>Benign Baseline</th>
                            <th>Deviation (Z-Score)</th>
                            <th>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {clinicalReport.abnormal_features.map((ab, i) => (
                            <tr key={i}>
                              <td style={{ fontWeight: 600, color: '#f8fafc' }}>{ab.feature}</td>
                              <td style={{ fontWeight: 700, color: '#ef4444', fontFamily: 'JetBrains Mono' }}>{ab.value}</td>
                              <td style={{ fontFamily: 'JetBrains Mono' }}>{ab.benign_baseline}</td>
                              <td style={{ fontFamily: 'JetBrains Mono', color: '#fbbf24', fontWeight: 600 }}>+{ab.deviation_z} σ</td>
                              <td><span className="badge badge-red">Elevated</span></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Recommendations */}
                <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '1.4rem' }}>
                  <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.7rem' }}>
                    Recommendations
                  </div>
                  <ul className="report-rec-list">
                    {(clinicalReport.recommendations || []).map((rec, i) => (
                      <li key={i}>
                        <span style={{ color: '#38bdf8', fontWeight: 700 }}>•</span>
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div style={{ marginTop: '2rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '1rem', fontSize: '0.72rem', color: '#64748b', textAlign: 'center' }}>
                  Educational Machine Learning demonstration only. Not for clinical diagnostic use.
                </div>
              </div>
            ) : (
              /* INPUT FORM */
              <div>
                <div className="card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
                    <div className="card-title" style={{ margin: 0 }}>
                      <Target size={18} color="#38bdf8" />
                      <span>Enter Patient Features</span>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <label style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Model:</label>
                      <select
                        value={predictModel}
                        onChange={(e) => setPredictModel(e.target.value)}
                        className="form-select"
                        style={{ width: 'auto', padding: '0.35rem 0.8rem', fontSize: '0.8rem' }}
                      >
                        {(status?.trained_models || ['Random Forest']).map((m) => (
                          <option key={m} value={m}>{m}</option>
                        ))}
                      </select>
                    </div>
                  </div>

                  {/* Patient Info */}
                  <div style={{
                    background: '#0b1120',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '10px',
                    padding: '1.1rem',
                    marginBottom: '1.4rem'
                  }}>
                    <div style={{ fontSize: '0.74rem', fontWeight: 700, color: '#94a3b8', marginBottom: '0.6rem', textTransform: 'uppercase' }}>
                      Patient Details (Optional)
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                      <div>
                        <label className="form-label" style={{ fontSize: '0.74rem' }}>Patient Name</label>
                        <input
                          type="text"
                          value={patientMeta.patient_name}
                          onChange={(e) => setPatientMeta({ ...patientMeta, patient_name: e.target.value })}
                          className="form-input"
                          style={{ padding: '0.45rem', fontSize: '0.82rem' }}
                        />
                      </div>
                      <div>
                        <label className="form-label" style={{ fontSize: '0.74rem' }}>Patient ID</label>
                        <input
                          type="text"
                          value={patientMeta.patient_id}
                          onChange={(e) => setPatientMeta({ ...patientMeta, patient_id: e.target.value })}
                          className="form-input"
                          style={{ padding: '0.45rem', fontSize: '0.82rem' }}
                        />
                      </div>
                      <div>
                        <label className="form-label" style={{ fontSize: '0.74rem' }}>Age</label>
                        <input
                          type="number"
                          value={patientMeta.age}
                          onChange={(e) => setPatientMeta({ ...patientMeta, age: e.target.value })}
                          className="form-input"
                          style={{ padding: '0.45rem', fontSize: '0.82rem' }}
                        />
                      </div>
                      <div>
                        <label className="form-label" style={{ fontSize: '0.74rem' }}>Physician</label>
                        <input
                          type="text"
                          value={patientMeta.physician}
                          onChange={(e) => setPatientMeta({ ...patientMeta, physician: e.target.value })}
                          className="form-input"
                          style={{ padding: '0.45rem', fontSize: '0.82rem' }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Preset Values */}
                  <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center', marginBottom: '1.2rem' }}>
                    <span style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 600 }}>Quick Presets:</span>
                    <button onClick={() => handleLoadPreset('benign')} className="btn btn-secondary" style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>
                      🟢 Benign Case
                    </button>
                    <button onClick={() => handleLoadPreset('malignant')} className="btn btn-secondary" style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>
                      🔴 Malignant Case
                    </button>
                    <button onClick={() => handleLoadPreset('mean')} className="btn btn-secondary" style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>
                      ⚪ Dataset Average
                    </button>
                  </div>

                  {/* Feature Inputs Grid */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '0.85rem' }}>
                    {predictFeatures.map((f) => (
                      <div key={f.name}>
                        <label className="form-label" style={{ fontSize: '0.74rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {f.name}
                        </label>
                        <input
                          type="number"
                          step="any"
                          value={patientInputs[f.name] ?? f.mean}
                          onChange={(e) => setPatientInputs({ ...patientInputs, [f.name]: parseFloat(e.target.value) || 0 })}
                          className="form-input"
                          style={{ padding: '0.45rem 0.65rem', fontSize: '0.84rem', fontFamily: 'JetBrains Mono' }}
                        />
                      </div>
                    ))}
                  </div>

                  <button
                    onClick={handleGenerateReport}
                    disabled={loading}
                    className="btn btn-primary"
                    style={{ width: '100%', marginTop: '1.8rem', padding: '0.85rem', fontSize: '0.95rem' }}
                  >
                    <Activity size={18} />
                    <span>Generate Prediction Report</span>
                  </button>
                </div>

                {/* Batch Upload */}
                <div className="card">
                  <div className="card-title">
                    <FileSpreadsheet size={17} color="#34d399" />
                    <span>Batch Prediction</span>
                  </div>
                  <p style={{ fontSize: '0.82rem', color: '#94a3b8', marginBottom: '1rem' }}>
                    Upload a CSV or Excel file containing patient records to predict all cases at once.
                  </p>

                  <input
                    type="file"
                    accept=".csv,.xlsx"
                    onChange={handleBatchUpload}
                    className="form-input"
                    style={{ padding: '0.55rem' }}
                  />

                  {batchResult && (
                    <div style={{ marginTop: '1.2rem' }}>
                      <div style={{ color: '#34d399', fontWeight: 700, marginBottom: '0.6rem', fontSize: '0.86rem' }}>
                        ✓ Evaluated {batchResult.total_records} records
                      </div>
                      <div style={{ overflowX: 'auto', maxHeight: '250px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                        <table className="data-table">
                          <thead>
                            <tr>
                              {batchResult.columns.slice(0, 7).map((c) => (
                                <th key={c}>{c}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {batchResult.preview.slice(0, 6).map((row, i) => (
                              <tr key={i}>
                                {batchResult.columns.slice(0, 7).map((c) => (
                                  <td key={c} style={{ fontFamily: typeof row[c] === 'number' ? 'JetBrains Mono' : 'inherit' }}>
                                    {String(row[c])}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
