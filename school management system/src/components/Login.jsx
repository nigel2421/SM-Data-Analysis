import React, { useState } from 'react';
import { auth } from '../firebase';
import { signInWithEmailAndPassword } from 'firebase/auth';
import { LogIn, ArrowRight, BookOpen, Wallet, Gavel, X, Menu } from 'lucide-react';

export default function Login() {
  const [showModal, setShowModal] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      await signInWithEmailAndPassword(auth, email, password);
    } catch (err) {
      setError('Invalid credentials. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const guestLogin = async () => {
    const demoEmail = 'admin@mzalendo.com';
    const demoPass = 'admin123';
    setEmail(demoEmail);
    setPassword(demoPass);
    setShowModal(true);
    setMobileMenuOpen(false);
    
    // Automatically trigger login for the demo experience
    setLoading(true);
    setError('');
    try {
      await signInWithEmailAndPassword(auth, demoEmail, demoPass);
    } catch (err) {
      setError('Demo user not found in Firebase. Please create "admin@mzalendo.com" with password "admin123" in your Firebase console to enable this shortcut.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="landing-page">
      {/* Navbar */}
      <nav className="landing-nav">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div className="sidebar-logo">M</div>
          <h2 style={{ color: 'white', margin: 0 }}>Mzalendo</h2>
        </div>
        
        <div className="hide-lp-mobile">
          <button className="btn btn-accent" onClick={() => setShowModal(true)}>
            <LogIn size={18} />
            Login to Dashboard
          </button>
        </div>

        <button className="landing-nav-btn" onClick={() => setMobileMenuOpen(true)}>
          <Menu size={28} />
        </button>
      </nav>

      {/* Mobile Menu Overlay */}
      <div className={`mobile-menu-overlay ${mobileMenuOpen ? 'open' : ''}`}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div className="sidebar-logo">M</div>
            <h2 style={{ color: 'white', margin: 0 }}>Mzalendo</h2>
          </div>
          <button className="landing-nav-btn" onClick={() => setMobileMenuOpen(false)}>
            <X size={28} />
          </button>
        </div>
        <div className="mobile-menu-content">
          <button className="btn btn-accent" style={{ padding: '1.25rem', justifyContent: 'center' }} onClick={() => { setShowModal(true); setMobileMenuOpen(false); }}>
            <LogIn size={20} />
            Login to Dashboard
          </button>
          <button className="btn" style={{ background: 'rgba(255,255,255,0.1)', color: 'white', padding: '1.25rem', justifyContent: 'center' }} onClick={guestLogin}>
            Explore System Demo
            <ArrowRight size={20} />
          </button>
        </div>
      </div>

      {/* Hero */}
      <section className="hero-section">
        <div className="hero-content animate-fade">
          <h1>Transforming Kenyan Schools</h1>
          <p>
            The comprehensive CBE-aligned management system for Mzalendo Schools. 
            Empower your staff, manage your finances, and track student success—all in one place.
          </p>
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'inherit' }} className="hero-actions">
            <button className="btn btn-accent hero-btn" style={{ padding: '1rem 2rem', fontSize: '1rem' }} onClick={guestLogin}>
              Explore System Demo
              <ArrowRight size={20} />
            </button>
          </div>
        </div>
        <img 
          src="/assets/hero.png" 
          alt="Modern School Campus" 
          className="hero-image"
        />
      </section>

      {/* Features */}
      <section className="feature-grid">
        <div className="feature-img-card animate-fade" style={{ animationDelay: '0.2s' }}>
          <img src="/assets/students.png" alt="Students" />
          <div className="feature-overlay">
            <BookOpen size={30} style={{ marginBottom: '1rem' }} />
            <h3>Academic Excellence</h3>
            <p style={{ fontSize: '0.85rem', opacity: 0.9 }}>CBE aligned tracking from Pre-Primary to Senior School.</p>
          </div>
        </div>
        <div className="feature-img-card animate-fade" style={{ animationDelay: '0.4s' }}>
          <img src="/assets/library.png" alt="Library" />
          <div className="feature-overlay">
            <Wallet size={30} style={{ marginBottom: '1rem' }} />
            <h3>Financial Transparency</h3>
            <p style={{ fontSize: '0.85rem', opacity: 0.9 }}>Automated fee management, budgets, and digital receipts.</p>
          </div>
        </div>
        <div className="feature-img-card animate-fade" style={{ animationDelay: '0.6s' }}>
          <div style={{ padding: '3rem', background: 'var(--primary)', height: '100%', color: 'white', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <Gavel size={40} style={{ color: 'var(--accent)', marginBottom: '1.5rem' }} />
            <h2>Robust Governance</h2>
            <p style={{ opacity: 0.8, marginTop: '1rem' }}>Manage BOM minutes, approvals, and school policies with ease.</p>
          </div>
        </div>
      </section>

      {/* Auth Modal */}
      {showModal && (
        <div className="auth-overlay">
          <div className="auth-modal">
            <button 
              onClick={() => setShowModal(false)}
              style={{ position: 'absolute', right: '1.5rem', top: '1.5rem', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
            >
              <X size={24} />
            </button>
            <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
              <div className="sidebar-logo" style={{ margin: '0 auto 1rem' }}>M</div>
              <h2 style={{ margin: 0 }}>Welcome Back</h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Enter your credentials to access the portal</p>
            </div>

            <form onSubmit={handleLogin}>
              <div className="input-group">
                <label>Email Address</label>
                <input 
                  type="email" 
                  placeholder="admin@mzalendo.com" 
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
              <div className="input-group">
                <label>Password</label>
                <input 
                  type="password" 
                  placeholder="••••••••" 
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>

              {error && <p style={{ color: 'red', fontSize: '0.8rem', marginBottom: '1rem', textAlign: 'center' }}>{error}</p>}

              <button 
                type="submit" 
                className="btn btn-primary" 
                style={{ width: '100%', padding: '1rem', justifyContent: 'center' }}
                disabled={loading}
              >
                {loading ? 'Verifying...' : 'Login to Portal'}
              </button>
            </form>
            
            <p style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Forgotten your password? Contact school administration.
            </p>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer style={{ background: '#00102e', color: 'white', padding: '4rem 10%', textAlign: 'center' }}>
        <p>© 2026 Mzalendo Schools Management System. All rights reserved.</p>
        <p style={{ opacity: 0.5, fontSize: '0.8rem', marginTop: '0.5rem' }}>Designed for the Kenya Competency-Based Curriculum.</p>
      </footer>
    </div>
  );
}
