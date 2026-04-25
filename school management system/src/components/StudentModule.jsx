import React, { useState } from 'react';
import { GRADES } from '../utils/dummyData';
import { 
  Plus, 
  Filter, 
  MoreVertical, 
  Search, 
  UserCheck, 
  BarChart3, 
  Users,
  CalendarDays,
  CheckCircle2,
  XCircle
} from 'lucide-react';

export default function StudentModule({ students }) {
  const [activeSubTab, setActiveSubTab] = useState('Directory');
  const [filter, setFilter] = useState('All');

  const filteredStudents = filter === 'All' 
    ? students 
    : students.filter(s => s.grade === filter);

  const renderContent = () => {
    switch (activeSubTab) {
      case 'Attendance':
        return (
          <div className="animate-fade">
            <div className="card card-header-actions" style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h3 style={{ margin: 0 }}>Daily School Attendance</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Full School Roll • {new Date().toLocaleDateString('en-GB', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>
              </div>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <button className="btn" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>Bulk Mark Present</button>
              </div>
            </div>
            <div className="card" style={{ padding: 0 }}>
              <div className="table-responsive">
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead style={{ background: 'var(--bg)' }}>
                    <tr>
                      <th style={{ padding: '1rem' }}>Student</th>
                      <th style={{ padding: '1rem' }}>Grade</th>
                      <th style={{ padding: '1rem' }}>Status</th>
                      <th style={{ padding: '1rem' }}>Remarks</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredStudents.map(student => (
                      <tr key={student.id} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td style={{ padding: '1rem' }}>
                          <div style={{ fontWeight: 600 }}>{student.name}</div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ID: {student.id}</div>
                        </td>
                        <td style={{ padding: '1rem' }}>{GRADES.find(g => g.id === student.grade)?.name}</td>
                        <td style={{ padding: '1rem' }}>
                          <div style={{ display: 'flex', gap: '0.5rem' }}>
                            <button style={{ border: 'none', background: '#DEF7EC', color: '#03543F', padding: '0.4rem 0.8rem', borderRadius: '4px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                              <CheckCircle2 size={14} /> Present
                            </button>
                            <button style={{ border: 'none', background: '#FDE8E8', color: '#9B1C1C', padding: '0.4rem 0.8rem', borderRadius: '4px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                              <XCircle size={14} /> Absent
                            </button>
                          </div>
                        </td>
                        <td style={{ padding: '1rem' }}>
                          <input type="text" placeholder="Note..." style={{ border: '1px solid var(--border)', padding: '0.3rem', borderRadius: '4px', fontSize: '0.85rem' }} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        );
      case 'Performance':
        return (
          <div className="animate-fade">
            <div className="responsive-grid" style={{ marginBottom: '1.5rem' }}>
              <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{ background: 'rgba(4, 99, 7, 0.1)', color: 'var(--secondary)', padding: '1rem', borderRadius: '12px' }}>
                  <BarChart3 size={24} />
                </div>
                <div>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', margin: 0 }}>Mean Grade</p>
                  <h3 style={{ margin: 0 }}>B+ (72.4)</h3>
                </div>
              </div>
              <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{ background: 'rgba(212, 175, 55, 0.1)', color: 'var(--accent)', padding: '1rem', borderRadius: '12px' }}>
                  <UserCheck size={24} />
                </div>
                <div>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', margin: 0 }}>Top Student</p>
                  <h3 style={{ margin: 0 }}>John Kamau (A)</h3>
                </div>
              </div>
            </div>
            <div className="card" style={{ padding: 0 }}>
              <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }} className="card-header-actions">
                <h3 style={{ margin: 0 }}>Termly Results</h3>
                <select style={{ padding: '0.5rem', borderRadius: '8px', border: '1px solid var(--border)' }}>
                  <option>Term 1, 2026</option>
                  <option>Term 3, 2025</option>
                </select>
              </div>
              <div className="table-responsive">
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead style={{ background: 'var(--bg)' }}>
                    <tr>
                      <th style={{ padding: '1rem' }}>Student</th>
                      <th style={{ padding: '1rem' }}>Math</th>
                      <th style={{ padding: '1rem' }}>English</th>
                      <th style={{ padding: '1rem' }}>Science</th>
                      <th style={{ padding: '1rem' }}>Total</th>
                      <th style={{ padding: '1rem' }}>Grade</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredStudents.map(student => (
                      <tr key={student.id} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td style={{ padding: '1rem', fontWeight: 600 }}>{student.name}</td>
                        <td style={{ padding: '1rem' }}>{Math.floor(Math.random() * 40) + 60}</td>
                        <td style={{ padding: '1rem' }}>{Math.floor(Math.random() * 40) + 60}</td>
                        <td style={{ padding: '1rem' }}>{Math.floor(Math.random() * 40) + 60}</td>
                        <td style={{ padding: '1rem', fontWeight: 700 }}>{Math.floor(Math.random() * 100) + 200}/300</td>
                        <td style={{ padding: '1rem' }}>
                          <span className="status-badge" style={{ background: 'var(--primary)', color: 'white' }}>A-</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        );
      default: // Directory
        return (
          <div className="animate-fade">
            <div className="card card-header-actions" style={{ marginBottom: '1.5rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <div style={{ position: 'relative', flex: 1, width: '100%' }}>
                <Search size={18} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                <input 
                  type="text" 
                  placeholder="Search by name or ID..." 
                  style={{ width: '100%', padding: '0.6rem 1rem 0.6rem 2.5rem', borderRadius: '8px', border: '1px solid var(--border)', outline: 'none' }}
                />
              </div>
              <div style={{ display: 'flex', gap: '0.5rem', width: '100%' }}>
                <select 
                  onChange={(e) => setFilter(e.target.value)}
                  style={{ flex: 1, padding: '0.6rem 1rem', borderRadius: '8px', border: '1px solid var(--border)', background: 'white' }}
                >
                  <option value="All">All Grades</option>
                  {GRADES.map(g => <option key={g.id} value={g.id}>{g.name}</option>)}
                </select>
                <button className="btn" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
                  <Filter size={18} />
                  <span className="hide-mobile">Filters</span>
                </button>
              </div>
            </div>

            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
              <div className="table-responsive">
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead style={{ background: 'var(--bg)', borderBottom: '1px solid var(--border)' }}>
                    <tr>
                      <th style={{ padding: '1rem' }}>Student Name</th>
                      <th style={{ padding: '1rem' }}>Grade</th>
                      <th style={{ padding: '1rem' }} className="hide-mobile">Stream</th>
                      <th style={{ padding: '1rem' }}>Balance</th>
                      <th style={{ padding: '1rem' }}>Status</th>
                      <th style={{ padding: '1rem' }}></th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredStudents.map(student => (
                      <tr key={student.id} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td style={{ padding: '1rem' }}>
                          <div style={{ fontWeight: 600 }}>{student.name}</div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ID: {student.id}</div>
                        </td>
                        <td style={{ padding: '1rem' }}>{GRADES.find(g => g.id === student.grade)?.name}</td>
                        <td style={{ padding: '1rem' }} className="hide-mobile">{student.stream}</td>
                        <td style={{ padding: '1rem' }}>
                          <span style={{ color: student.balance > 0 ? 'red' : 'green', fontWeight: 600 }}>
                            KES {student.balance.toLocaleString()}
                          </span>
                        </td>
                        <td style={{ padding: '1rem' }}>
                          <span className="status-badge" style={{ background: student.status === 'Active' ? '#DEF7EC' : '#FDE8E8', color: student.status === 'Active' ? '#03543F' : '#9B1C1C' }}>
                            {student.status}
                          </span>
                        </td>
                        <td style={{ padding: '1rem', textAlign: 'right' }}>
                          <MoreVertical size={18} style={{ cursor: 'pointer', color: 'var(--text-muted)' }} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="page-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }} className="card-header-actions">
        <div>
          <h1 style={{ fontSize: '1.8rem' }}>Student Records</h1>
          <p style={{ color: 'var(--text-muted)' }}>Admissions, Attendance & Performance Performance</p>
        </div>
        <button className="btn btn-primary">
          <Plus size={18} />
          New Admission
        </button>
      </div>

      <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.5rem', overflowX: 'auto' }}>
        {[
          { id: 'Directory', icon: Users, label: 'Directory' },
          { id: 'Attendance', icon: CalendarDays, label: 'Attendance' },
          { id: 'Performance', icon: BarChart3, label: 'Performance' },
        ].map(tab => (
          <button 
            key={tab.id}
            onClick={() => setActiveSubTab(tab.id)}
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem', 
              padding: '0.5rem 1rem', 
              border: 'none', 
              background: 'none', 
              cursor: 'pointer',
              color: activeSubTab === tab.id ? 'var(--primary)' : 'var(--text-muted)',
              borderBottom: activeSubTab === tab.id ? '2px solid var(--primary)' : '2px solid transparent',
              fontWeight: activeSubTab === tab.id ? 600 : 400,
              whiteSpace: 'nowrap'
            }}
          >
            <tab.icon size={18} />
            {tab.label}
          </button>
        ))}
      </div>

      {renderContent()}
    </div>
  );
}
