import React, { useState, useEffect, Suspense, lazy } from 'react';
import { 
  LayoutDashboard, 
  Users, 
  GraduationCap, 
  Wallet, 
  FileText, 
  Package, 
  ShoppingCart, 
  Gavel, 
  Settings,
  Bell,
  Search,
  User,
  LogIn,
  Menu,
  X
} from 'lucide-react';

// --- Components ---

// eslint-disable-next-line react/prop-types
const SidebarItem = ({ icon: SidebarIcon, label, active, onClick }) => (
  <div className={`nav-item ${active ? 'active' : ''}`} onClick={onClick}>
    {/* eslint-disable-next-line no-unused-vars */}
    <SidebarIcon size={20} />
    <span>{label}</span>
  </div>
);

const Header = ({ title, onMenuClick }) => (
  <header className="header">
    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
      <button className="menu-toggle" onClick={onMenuClick}>
        <Menu size={24} />
      </button>
      <h2>{title}</h2>
    </div>
    <div className="header-actions" style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
      <div className="hide-mobile" style={{ position: 'relative' }}>
        <Search size={20} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
        <input 
          type="text" 
          placeholder="Search records..." 
          style={{ padding: '0.6rem 1rem 0.6rem 2.5rem', borderRadius: '8px', border: '1px solid var(--border)', outline: 'none' }}
        />
      </div>
      <div style={{ display: 'flex', gap: '1rem', color: 'var(--text-muted)' }}>
        <Bell size={22} style={{ cursor: 'pointer' }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', color: 'var(--text)' }}>
          <User size={22} />
          <span className="hide-mobile" style={{ fontWeight: 600 }}>Admin</span>
        </div>
      </div>
    </div>
  </header>
);

// --- Pages (Stubbed for now) ---
const Dashboard = ({ students, staff }) => (
  <div className="page-container">
    <div className="responsive-grid" style={{ marginBottom: '2rem' }}>
      <div className="card">
        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Total Students</p>
        <h1 style={{ fontSize: '2rem', marginTop: '0.5rem' }}>{students.length}</h1>
        <p style={{ color: 'var(--secondary)', fontSize: '0.75rem', marginTop: '0.5rem' }}>+12% from last term</p>
      </div>
      <div className="card">
        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Fees Collected (KES)</p>
        <h1 style={{ fontSize: '2rem', marginTop: '0.5rem' }}>8.4M</h1>
        <p style={{ color: 'var(--secondary)', fontSize: '0.75rem', marginTop: '0.5rem' }}>75% of target</p>
      </div>
      <div className="card">
        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Total Staff</p>
        <h1 style={{ fontSize: '2rem', marginTop: '0.5rem' }}>{staff.length}</h1>
        <p style={{ color: 'var(--accent)', fontSize: '0.75rem', marginTop: '0.5rem' }}>Steady</p>
      </div>
    </div>

    <div className="responsive-grid">
      <div className="card" style={{ minHeight: '300px' }}>
        <h3>Financial Overview</h3>
        <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>Monthly revenue vs expenditure</p>
        <div style={{ height: '200px', display: 'flex', alignItems: 'flex-end', gap: '1rem' }}>
          {[60, 45, 80, 50, 90, 70].map((h, i) => (
            <div key={i} style={{ flex: 1, height: `${h}%`, background: 'var(--primary)', borderRadius: '4px 4px 0 0' }} />
          ))}
        </div>
      </div>
      <div className="card">
        <h3>Upcoming BOM Meetings</h3>
        <div style={{ marginTop: '1rem' }}>
          {[1, 2].map(i => (
            <div key={i} style={{ padding: '1rem 0', borderBottom: '1px solid var(--border)' }}>
              <p style={{ fontWeight: 600 }}>Term 2 Budget Approval</p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>24th May, 2026 • 10:00 AM</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  </div>
);

const StudentModule = lazy(() => import('./components/StudentModule'));
const AcademicsModule = lazy(() => import('./components/AcademicsModule'));
const FinanceModule = lazy(() => import('./components/FinanceModule'));
const StaffModule = lazy(() => import('./components/StaffModule'));
const OperationsModule = lazy(() => import('./components/OperationsModule'));
const SettingsModule = lazy(() => import('./components/SettingsModule'));
const Login = lazy(() => import('./components/Login'));
import { fetchData } from './services/firebaseService';
import { SCHOOL_DATA } from './utils/dummyData';
import { auth } from './firebase';
import { onAuthStateChanged, signOut } from 'firebase/auth';

export default function App() {
  const [user, setUser] = useState(null);
  const [activeTab, setActiveTab] = useState('Dashboard');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [students, setStudents] = useState([]);
  const [staff, setStaff] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      if (currentUser) {
        loadData();
      } else {
        setLoading(false);
      }
    });

    return () => unsubscribe();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const studentData = await fetchData('students', SCHOOL_DATA.id);
      const staffData = await fetchData('staff', SCHOOL_DATA.id);
      setStudents(studentData);
      setStaff(staffData);
    } catch (e) {
      console.error("Failed to load data", e);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await signOut(auth);
    } catch (e) {
      console.error("Sign out failed", e);
    }
  };

  if (!user && !loading) {
    return <Login />;
  }

  const menuItems = [
    { id: 'Dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { id: 'Students', icon: GraduationCap, label: 'Students' },
    { id: 'Academics', icon: FileText, label: 'Academics' },
    { id: 'Finance', icon: Wallet, label: 'Finance' },
    { id: 'Staff', icon: Users, label: 'Staff & Payroll' },
    { id: 'Assets', icon: Package, label: 'Assets' },
    { id: 'Procurement', icon: ShoppingCart, label: 'Procurement' },
    { id: 'Governance', icon: Gavel, label: 'Governance' },
    { id: 'Settings', icon: Settings, label: 'Settings' },
  ];

  const renderModule = () => {
    if (loading) return (
      <div className="page-container" style={{ display: 'grid', placeItems: 'center', height: '60vh' }}>
        <h2 style={{ color: 'var(--text-muted)' }}>Loading records...</h2>
      </div>
    );

    return (
      <Suspense fallback={
        <div className="page-container" style={{ display: 'grid', placeItems: 'center', height: '60vh' }}>
          <div className="loading-spinner"></div>
          <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>Loading module...</p>
        </div>
      }>
        {(() => {
          switch (activeTab) {
            case 'Dashboard': return <Dashboard students={students} staff={staff} />;
            case 'Students': return <StudentModule students={students} />;
            case 'Academics': return <AcademicsModule />;
            case 'Finance': return <FinanceModule />;
            case 'Staff': return <StaffModule staff={staff} />;
            case 'Assets': 
            case 'Procurement': 
            case 'Governance': return <OperationsModule type={activeTab} />;
            case 'Settings': return <SettingsModule />;
            default: return (
              <div className="page-container">
                <div className="card" style={{ textAlign: 'center', padding: '5rem' }}>
                  <h2 style={{ color: 'var(--text-muted)' }}>{activeTab} Module Coming Soon</h2>
                  <p>We are currently building the {activeTab} section for the prototype.</p>
                </div>
              </div>
            );
          }
        })()}
      </Suspense>
    );
  };

  return (
    <div className="app-container">
      {isSidebarOpen && (
        <div 
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 950 }} 
          onClick={() => setIsSidebarOpen(false)} 
        />
      )}
      <aside className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-logo">M</div>
          <div style={{ flex: 1 }}>
            <h3 style={{ color: 'white', margin: 0, fontSize: '1.1rem' }}>Mzalendo</h3>
            <p style={{ color: 'var(--accent)', fontSize: '0.7rem', margin: 0 }}>SCHOOLS MGMT</p>
          </div>
          <button className="menu-toggle" onClick={() => setIsSidebarOpen(false)} style={{ color: 'white' }}>
            <X size={24} />
          </button>
        </div>
        
        <nav className="sidebar-nav">
          {menuItems.map(item => (
            <SidebarItem 
              key={item.id}
              icon={item.icon}
              label={item.label}
              active={activeTab === item.id}
              onClick={() => {
                setActiveTab(item.id);
                setIsSidebarOpen(false);
              }}
            />
          ))}
        </nav>

        <div style={{ padding: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div className="cbe-badge">CBE COMPLIANT</div>
          <button 
            onClick={handleLogout}
            style={{ padding: '0.8rem', background: 'rgba(239, 68, 68, 0.1)', border: 'none', borderRadius: '8px', color: '#EF4444', fontWeight: 600, cursor: 'pointer', textAlign: 'left', display: 'flex', gap: '0.75rem' }}
          >
            <LogIn size={18} style={{ transform: 'rotate(180deg)' }} />
            Sign Out
          </button>
        </div>
      </aside>

      <main className="main-content">
        <Header title={activeTab} onMenuClick={() => setIsSidebarOpen(true)} />
        {renderModule()}
      </main>
    </div>
  );
}
