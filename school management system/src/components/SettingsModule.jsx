import React, { useState } from 'react';
import { Settings, Globe, Shield, CreditCard, RefreshCw, Database } from 'lucide-react';
import { SCHOOL_DATA } from '../utils/dummyData';
import { seedDatabase } from '../services/firebaseService';

export default function SettingsModule() {
  const [seeding, setSeeding] = useState(false);

  const handleSeed = async () => {
    setSeeding(true);
    try {
      await seedDatabase(SCHOOL_DATA.id);
      alert('Database seeded successfully!');
    } catch (e) {
      alert('Seeding failed. Check console for details.');
      console.error(e);
    } finally {
      setSeeding(false);
    }
  };
  return (
    <div className="page-container">
      <h1 style={{ marginBottom: '2rem' }}>System Settings</h1>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '2rem' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className="card" style={{ borderLeft: '4px solid var(--primary)', cursor: 'pointer' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Globe size={18} />
              <span>School Profile</span>
            </div>
          </div>
          <div className="card" style={{ cursor: 'pointer' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Shield size={18} />
              <span>User Permissions</span>
            </div>
          </div>
          <div className="card" style={{ cursor: 'pointer' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <CreditCard size={18} />
              <span>Billing & Subscription</span>
            </div>
          </div>
          <div className="card" style={{ cursor: 'pointer', background: 'var(--bg)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <RefreshCw size={18} />
              <span>Automation Rules</span>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 style={{ marginBottom: '1.5rem' }}>White-Labeling & Multi-tenancy</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div>
              <p style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.5rem' }}>Reseller ID / Tenant ID</p>
              <input 
                type="text" 
                defaultValue={SCHOOL_DATA.id}
                readOnly
                style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border)', background: 'var(--bg)' }}
              />
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Unique identifier for this school instance.</p>
            </div>

            <div>
              <p style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.5rem' }}>School Name</p>
              <input 
                type="text" 
                defaultValue={SCHOOL_DATA.name}
                style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border)' }}
              />
            </div>

            <div style={{ borderTop: '1px solid var(--border)', paddingTop: '1.5rem' }}>
              <h4 style={{ marginBottom: '1rem' }}>Data Management</h4>
              <button 
                onClick={handleSeed}
                disabled={seeding}
                className="btn" 
                style={{ background: 'var(--bg)', border: '1px solid var(--border)', width: '100%', justifyContent: 'center' }}
              >
                <Database size={18} />
                {seeding ? 'Seeding...' : 'Seed Data to Firestore'}
              </button>
              <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.5rem', textAlign: 'center' }}>
                Use this to push the initial prototype data to your live Firebase project once keys are added.
              </p>
            </div>

            <div style={{ borderTop: '1px solid var(--border)', paddingTop: '1.5rem' }}>
              <h4 style={{ marginBottom: '1rem' }}>Promotion Automation</h4>
              <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <div style={{ flex: 1 }}>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Auto-promotion Date</p>
                  <input type="date" defaultValue="2027-01-01" style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border)' }} />
                </div>
                <div style={{ flex: 1 }}>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Status</p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--secondary)', fontWeight: 600 }}>
                    <div style={{ width: '8px', height: '8px', background: 'var(--secondary)', borderRadius: '50%' }}></div>
                    Scheduled
                  </div>
                </div>
              </div>
            </div>

            <button className="btn btn-primary" style={{ alignSelf: 'flex-start', marginTop: '1rem' }}>
              Save Configuration
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
