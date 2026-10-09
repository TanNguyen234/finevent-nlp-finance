import React, { useState, useEffect } from 'react';

export default function App() {
  const [companies, setCompanies] = useState([
    { cik: '0000320193', company_name: 'Apple Inc.' },
    { cik: '0000789019', company_name: 'Microsoft Corporation' }
  ]);
  const [selectedCik, setSelectedCik] = useState('0000320193');
  const [fromDate, setFromDate] = useState('2020-01-01');
  const [toDate, setToDate] = useState('2026-12-31');
  const [changes, setChanges] = useState([]);
  const [events, setEvents] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);
  const [predictText, setPredictText] = useState(
    'For the second quarter of fiscal 2025, Apple expects gross margin to reach between 46.5 percent and 47.5 percent, an upward revision from previously modeled 45.5% to 46.0%.'
  );
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch changes and events
  useEffect(() => {
    fetchChanges();
    fetchEvents();
  }, [selectedCik, fromDate, toDate]);

  const fetchChanges = async () => {
    try {
      const res = await fetch(`http://localhost:8000/companies/${selectedCik}/changes?from=${fromDate}&to=${toDate}`);
      if (res.ok) {
        const data = await res.json();
        setChanges(data);
        if (data.length > 0) setSelectedItem(data[0]);
      }
    } catch (err) {
      console.warn('API offline or loading defaults');
    }
  };

  const fetchEvents = async () => {
    try {
      const res = await fetch(`http://localhost:8000/companies/${selectedCik}/events?limit=20`);
      if (res.ok) {
        const data = await res.json();
        setEvents(data);
      }
    } catch (err) {
      console.warn('API offline');
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/predict/event', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: predictText })
      });
      if (res.ok) {
        const data = await res.json();
        setPredictionResult(data);
      }
    } catch (err) {
      alert('Backend server not connected. Run: uvicorn app.main:app --port 8000');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Terminal Top Bar */}
      <header className="terminal-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ width: '12px', height: '12px', borderRadius: '3px', background: 'var(--accent-cyan)' }}></div>
          <div>
            <h1 style={{ fontSize: '1.15rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
              FinEvent <span style={{ color: 'var(--accent-cyan)' }}>Terminal</span>
            </h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>P01 SEC Guidance Revision & Event Intelligence</p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <div className="badge-live">
            <span className="pulse-dot"></span>
            FASTAPI V1 CONNECTED
          </div>
          <span className="mono" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>GPU: KAGGLE TESLA T4 READY</span>
        </div>
      </header>

      {/* Main Workspace */}
      <main style={{ padding: '1.5rem 2rem', flex: 1, display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {/* Controls Toolbar */}
        <section className="card" style={{ display: 'flex', gap: '1.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <div>
            <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.3rem' }}>
              TARGET REGISTRANT (CIK)
            </label>
            <select
              value={selectedCik}
              onChange={(e) => setSelectedCik(e.target.value)}
              className="mono"
              style={{
                background: '#0e1524',
                color: '#fff',
                border: '1px solid var(--border)',
                borderRadius: '6px',
                padding: '0.45rem 0.8rem',
                fontSize: '0.85rem'
              }}
            >
              <option value="0000320193">Apple Inc. (CIK: 0000320193)</option>
              <option value="0000789019">Microsoft Corp (CIK: 0000789019)</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.3rem' }}>
              HORIZON FROM
            </label>
            <input
              type="text"
              value={fromDate}
              onChange={(e) => setFromDate(e.target.value)}
              className="mono"
              style={{
                background: '#0e1524',
                color: '#fff',
                border: '1px solid var(--border)',
                borderRadius: '6px',
                padding: '0.45rem 0.8rem',
                fontSize: '0.85rem',
                width: '120px'
              }}
            />
          </div>

          <div>
            <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.3rem' }}>
              HORIZON TO
            </label>
            <input
              type="text"
              value={toDate}
              onChange={(e) => setToDate(e.target.value)}
              className="mono"
              style={{
                background: '#0e1524',
                color: '#fff',
                border: '1px solid var(--border)',
                borderRadius: '6px',
                padding: '0.45rem 0.8rem',
                fontSize: '0.85rem',
                width: '120px'
              }}
            />
          </div>

          <div style={{ marginLeft: 'auto', display: 'flex', gap: '1rem' }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>REVISIONS DETECTED</div>
              <div className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                {changes.length}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>HISTORICAL EVENTS</div>
              <div className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-green)' }}>
                {events.length}
              </div>
            </div>
          </div>
        </section>

        {/* 2-Column Split: Changes Table vs Evidence Inspector */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.8fr 1.2fr', gap: '1.5rem', flex: 1 }}>
          {/* Left Column: Guidance Changes Table */}
          <section className="card" style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ color: 'var(--accent-cyan)' }}>●</span> GUIDANCE REVISION TRACKER
            </h2>

            <div style={{ overflowX: 'auto', flex: 1 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                    <th style={{ padding: '0.6rem' }}>METRIC</th>
                    <th style={{ padding: '0.6rem' }}>PERIOD</th>
                    <th style={{ padding: '0.6rem' }}>PRIOR GUIDANCE</th>
                    <th style={{ padding: '0.6rem' }}>REVISED GUIDANCE</th>
                    <th style={{ padding: '0.6rem' }}>DELTA</th>
                    <th style={{ padding: '0.6rem' }}>DIRECTION</th>
                  </tr>
                </thead>
                <tbody>
                  {changes.length === 0 ? (
                    <tr>
                      <td colSpan="6" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                        No guidance changes found in selected date window.
                      </td>
                    </tr>
                  ) : (
                    changes.map((ch, idx) => {
                      const isSelected = selectedItem === ch;
                      return (
                        <tr
                          key={idx}
                          onClick={() => setSelectedItem(ch)}
                          style={{
                            borderBottom: '1px solid rgba(43, 56, 82, 0.4)',
                            cursor: 'pointer',
                            background: isSelected ? 'rgba(0, 240, 255, 0.08)' : 'transparent'
                          }}
                        >
                          <td style={{ padding: '0.75rem 0.6rem', fontWeight: 600 }}>{ch.metric}</td>
                          <td className="mono" style={{ padding: '0.75rem 0.6rem', color: 'var(--text-muted)' }}>
                            {ch.fiscal_period}
                          </td>
                          <td className="mono" style={{ padding: '0.75rem 0.6rem' }}>
                            {ch.old_low !== null ? `${ch.old_low} - ${ch.old_high}` : 'N/A'} {ch.unit}
                          </td>
                          <td className="mono" style={{ padding: '0.75rem 0.6rem', color: '#fff', fontWeight: 600 }}>
                            {ch.new_low} - {ch.new_high} {ch.unit}
                          </td>
                          <td className="mono" style={{ padding: '0.75rem 0.6rem' }}>
                            <span style={{ color: ch.delta_mid > 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                              {ch.delta_mid > 0 ? `+${ch.delta_mid}` : ch.delta_mid}
                            </span>
                          </td>
                          <td style={{ padding: '0.75rem 0.6rem' }}>
                            <span
                              className={
                                ch.direction === 'UPWARD_REVISION'
                                  ? 'tag-upward'
                                  : ch.direction === 'DOWNWARD_REVISION'
                                  ? 'tag-downward'
                                  : 'tag-reaffirmed'
                              }
                            >
                              {ch.direction}
                            </span>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </section>

          {/* Right Column: Verifiable Evidence Card */}
          <section className="card" style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ color: 'var(--accent-green)' }}>●</span> SEC EDGAR AUDIT EVIDENCE
            </h2>

            {selectedItem ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', flex: 1 }}>
                <div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>
                    VERBATIM DISCLOSURE EVIDENCE SPAN
                  </div>
                  <div
                    style={{
                      background: 'rgba(0, 0, 0, 0.4)',
                      padding: '1rem',
                      borderRadius: '8px',
                      borderLeft: '4px solid var(--accent-cyan)',
                      fontStyle: 'italic',
                      lineHeight: '1.5',
                      fontSize: '0.85rem'
                    }}
                  >
                    "{selectedItem.evidence}"
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem' }}>
                  <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.6rem', borderRadius: '6px' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>CONFIDENCE SCORE</div>
                    <div className="mono" style={{ fontSize: '1rem', color: 'var(--accent-green)', fontWeight: 700 }}>
                      {(selectedItem.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.6rem', borderRadius: '6px' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>FILING DATE</div>
                    <div className="mono" style={{ fontSize: '0.9rem' }}>
                      {selectedItem.filing_date_current || 'N/A'}
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid var(--border)' }}>
                  <a
                    href={selectedItem.source_url}
                    target="_blank"
                    rel="noreferrer"
                    style={{
                      display: 'block',
                      textAlign: 'center',
                      background: 'rgba(0, 240, 255, 0.1)',
                      color: 'var(--accent-cyan)',
                      border: '1px solid var(--accent-cyan)',
                      padding: '0.6rem',
                      borderRadius: '6px',
                      textDecoration: 'none',
                      fontWeight: 600,
                      fontSize: '0.85rem'
                    }}
                  >
                    ↗ Open SEC Filing Document (Archive URL)
                  </a>
                </div>
              </div>
            ) : (
              <div style={{ color: 'var(--text-muted)', textAlign: 'center', margin: 'auto' }}>
                Select a revision row on the left to inspect evidence
              </div>
            )}
          </section>
        </div>

        {/* Bottom Section: Live NLP Inference Playground */}
        <section className="card">
          <h2 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ color: 'var(--accent-orange)' }}>●</span> LIVE EVENT PREDICTOR & ARGUMENT EXTRACTOR
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem' }}>
            <div>
              <textarea
                value={predictText}
                onChange={(e) => setPredictText(e.target.value)}
                rows="4"
                style={{
                  width: '100%',
                  background: '#0e1524',
                  color: '#fff',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  padding: '0.8rem',
                  fontSize: '0.85rem',
                  fontFamily: 'inherit',
                  resize: 'vertical'
                }}
              />
              <button
                onClick={handlePredict}
                disabled={loading}
                style={{
                  marginTop: '0.6rem',
                  background: 'var(--accent-cyan)',
                  color: '#000',
                  fontWeight: 700,
                  border: 'none',
                  padding: '0.6rem 1.4rem',
                  borderRadius: '6px',
                  cursor: 'pointer'
                }}
              >
                {loading ? 'Analyzing...' : 'Run Extraction Engine'}
              </button>
            </div>

            <div style={{ background: '#0a0e17', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>MODEL PREDICTION</div>
              {predictionResult ? (
                <div style={{ fontSize: '0.85rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                  <div>
                    <strong>Class:</strong> <span className="tag-upward">{predictionResult.event_type}</span>
                  </div>
                  <div>
                    <strong>Confidence:</strong> {(predictionResult.confidence * 100).toFixed(1)}%
                  </div>
                  <div>
                    <strong>Metric:</strong> {predictionResult.arguments.metric}
                  </div>
                  <div>
                    <strong>Range:</strong> {predictionResult.arguments.new_low} - {predictionResult.arguments.new_high} {predictionResult.arguments.unit}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    Model: {predictionResult.model_version}
                  </div>
                </div>
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Click Run to view classification and parsed numbers.</div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
