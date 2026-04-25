import React from 'react';
import { Users, DollarSign, Calendar, FileCheck } from 'lucide-react';

export default function StaffModule({ staff }) {
  return (
    <div className="page-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div>
          <h1 style={{ fontSize: '1.8rem' }}>Staff & Payroll</h1>
          <p style={{ color: 'var(--text-muted)' }}>Employee records and monthly salary processing</p>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button className="btn" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <Calendar size={18} />
            Duty Roster
          </button>
          <button className="btn btn-primary">
            <DollarSign size={18} />
            Run Payroll
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '1.5rem' }}>
        <div className="card">
          <h3 style={{ marginBottom: '1rem' }}>Payroll Summary (May)</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem', background: 'var(--bg)', borderRadius: '8px' }}>
              <span>Gross Salaries</span>
              <span style={{ fontWeight: 600 }}>KES 1.2M</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem', background: 'var(--bg)', borderRadius: '8px' }}>
              <span>Deductions (PAYE/NHIF)</span>
              <span style={{ fontWeight: 600, color: '#E02424' }}>KES 185,000</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem', borderTop: '2px dashed var(--border)', marginTop: '0.5rem' }}>
              <span style={{ fontWeight: 700 }}>Net Pay</span>
              <span style={{ fontWeight: 700, color: 'var(--secondary)' }}>KES 1,015,000</span>
            </div>
          </div>
          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '1.5rem', fontStyle: 'italic' }}>
            * Automated KRA/PAYE calculations integrated.
          </p>
        </div>

        <div className="card" style={{ padding: 0 }}>
          <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border)' }}>
            <h3 style={{ margin: 0 }}>Employee Register</h3>
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead style={{ background: 'var(--bg)' }}>
              <tr>
                <th style={{ padding: '1rem' }}>Employee Name</th>
                <th style={{ padding: '1rem' }}>Role</th>
                <th style={{ padding: '1rem' }}>Base Salary</th>
                <th style={{ padding: '1rem' }}>Joined</th>
                <th style={{ padding: '1rem' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {staff.map(member => (
                <tr key={member.id} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '1rem' }}>
                    <div style={{ fontWeight: 600 }}>{member.name}</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>ID: {member.id}</div>
                  </td>
                  <td style={{ padding: '1rem' }}>{member.role}</td>
                  <td style={{ padding: '1rem', fontWeight: 600 }}>KES {member.salary.toLocaleString()}</td>
                  <td style={{ padding: '1rem' }}>{member.joined}</td>
                  <td style={{ padding: '1rem' }}>
                    <FileCheck size={18} style={{ cursor: 'pointer', color: 'var(--primary)' }} />
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
