import React from 'react';
import { GRADES, SUBJECTS_MAP } from '../utils/dummyData';
import { BookOpen, Calendar, ArrowRightLeft, ShieldCheck } from 'lucide-react';

export default function AcademicsModule() {
  const levels = ['Early Years', 'Lower Primary', 'Upper Primary', 'Junior School', 'Senior School'];

  return (
    <div className="page-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div>
          <h1 style={{ fontSize: '1.8rem' }}>Academic Framework</h1>
          <p style={{ color: 'var(--text-muted)' }}>CBE Curriculum & Subject Mapping</p>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button className="btn" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <Calendar size={18} />
            Academic Calendar
          </button>
          <button className="btn btn-primary">
            <ArrowRightLeft size={18} />
            Auto-Promotion Logic
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
            <div style={{ padding: '0.5rem', background: 'var(--bg)', borderRadius: '8px', color: 'var(--primary)' }}>
              <BookOpen size={20} />
            </div>
            <h3 style={{ margin: 0 }}>Subject Mapping (Per Level)</h3>
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {levels.map(level => (
              <details key={level} style={{ border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
                <summary style={{ padding: '1rem', cursor: 'pointer', background: 'white', fontWeight: 600, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  {level}
                  <span className="cbe-badge">CBE</span>
                </summary>
                <div style={{ padding: '1rem', background: 'var(--bg)', fontSize: '0.9rem' }}>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {level === 'Senior School' ? (
                      Object.keys(SUBJECTS_MAP[level]).map(pathway => (
                        <div key={pathway} style={{ width: '100%', marginTop: '0.5rem' }}>
                          <p style={{ fontWeight: 600, color: 'var(--secondary)', marginBottom: '0.25rem' }}>{pathway} Pathway:</p>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                            {SUBJECTS_MAP[level][pathway].map(sub => (
                              <span key={sub} style={{ padding: '4px 10px', background: 'white', border: '1px solid var(--border)', borderRadius: '4px' }}>{sub}</span>
                            ))}
                          </div>
                        </div>
                      ))
                    ) : (
                      SUBJECTS_MAP[level].map(sub => (
                        <span key={sub} style={{ padding: '4px 10px', background: 'white', border: '1px solid var(--border)', borderRadius: '4px' }}>{sub}</span>
                      ))
                    )}
                  </div>
                </div>
              </details>
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="card" style={{ background: 'var(--primary)', color: 'white' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
              <ShieldCheck size={24} style={{ color: 'var(--accent)' }} />
              <h3 style={{ color: 'white', margin: 0 }}>Automated Promotion Logic</h3>
            </div>
            <p style={{ fontSize: '0.9rem', opacity: 0.9, lineHeight: 1.6 }}>
              On <strong>January 1st</strong> of every academic year, Mzalendo Schools automatically transitions students:
            </p>
            <ul style={{ fontSize: '0.85rem', marginTop: '1rem', paddingLeft: '1.2rem', opacity: 0.9 }}>
              <li>PP2 students transition to Grade 1 (Lower Primary)</li>
              <li>Grade 6 students transition to Grade 7 (Junior School)</li>
              <li>Grade 9 students transition to Senior School based on Pathway Selection</li>
              <li>Grade 12 students are marked as Alumni/Graduated</li>
            </ul>
            <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}>
              <p style={{ fontSize: '0.8rem', fontWeight: 600 }}>Next Promotion Run: Jan 1, 2027</p>
            </div>
          </div>

          <div className="card">
            <h3>Grade Transitions</h3>
            <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {GRADES.slice(0, 5).map((g, i) => (
                <div key={g.id} style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '0.5rem', borderBottom: '1px solid var(--border)' }}>
                  <div style={{ width: '40px', height: '40px', background: 'var(--bg)', borderRadius: '50%', display: 'grid', placeItems: 'center', fontWeight: 'bold' }}>{i+1}</div>
                  <div>
                    <p style={{ margin: 0, fontWeight: 600 }}>{g.name}</p>
                    <p style={{ margin: 0, fontSize: '0.75rem', color: 'var(--text-muted)' }}>{g.level}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
