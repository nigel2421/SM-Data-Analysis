import React, { useState } from 'react';
import { 
  Wallet, 
  TrendingUp, 
  TrendingDown, 
  FileText, 
  Download, 
  PieChart, 
  FileCheck,
  Calendar,
  Layers
} from 'lucide-react';

export default function FinanceModule() {
  const [subTab, setSubTab] = useState('Cashbook');

  const transactions = [
    { id: 'tx001', type: 'Income', category: 'Tuition Fees', amount: 45000, date: '2026-04-10', status: 'Paid', doc: 'Receipt' },
    { id: 'tx002', type: 'Expense', category: 'Electricity', amount: 12000, date: '2026-04-12', status: 'Pending', doc: 'Invoice' },
    { id: 'tx003', type: 'Expense', category: 'Stationery', amount: 8000, date: '2026-04-14', status: 'Paid', doc: 'Voucher' },
    { id: 'tx004', type: 'Income', category: 'Bus Fees', amount: 15000, date: '2026-04-15', status: 'Paid', doc: 'Receipt' },
  ];

  const budgets = [
    { id: 'B26-01', department: 'Academics', allocation: 5000000, spent: 3200000, per: 64 },
    { id: 'B26-02', department: 'Operations', allocation: 2000000, spent: 1800000, per: 90 },
    { id: 'B26-03', department: 'Sports', allocation: 800000, spent: 150000, per: 18 },
  ];

  const renderContent = () => {
    switch (subTab) {
      case 'Budgets':
        return (
          <div className="animate-fade">
            <div className="responsive-grid" style={{ marginBottom: '2rem' }}>
              {budgets.map(b => (
                <div key={b.id} className="card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                    <h4 style={{ margin: 0 }}>{b.department}</h4>
                    <span className="cbe-badge">2026 Full Year</span>
                  </div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Allocation: KES {b.allocation.toLocaleString()}</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, margin: '0.5rem 0' }}>Spent: KES {b.spent.toLocaleString()}</div>
                  <div style={{ height: '8px', background: 'var(--bg)', borderRadius: '4px', overflow: 'hidden', marginTop: '1rem' }}>
                    <div style={{ width: `${b.per}%`, height: '100%', background: b.per > 85 ? '#E02424' : 'var(--secondary)', transition: 'width 0.5s' }} />
                  </div>
                  <div style={{ textAlign: 'right', fontSize: '0.75rem', marginTop: '0.25rem', color: 'var(--text-muted)' }}>{b.per}% utilized</div>
                </div>
              ))}
            </div>
            <div className="card">
              <h3>Budget Approvals</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Pending BOM approvals for budget adjustments.</p>
              <div style={{ marginTop: '1.5rem', padding: '1rem', border: '1px dashed var(--border)', borderRadius: '8px', textAlign: 'center' }}>
                No pending budget requests.
              </div>
            </div>
          </div>
        );
      case 'SupportingDocs':
        return (
          <div className="animate-fade">
            <div className="card" style={{ padding: 0 }}>
              <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }} className="card-header-actions">
                <h3 style={{ margin: 0 }}>Document Repository</h3>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button className="btn btn-primary" style={{ fontSize: '0.75rem' }}>Upload New</button>
                  <button className="btn" style={{ fontSize: '0.75rem', background: 'var(--bg)', border: '1px solid var(--border)' }}>Filter</button>
                </div>
              </div>
              <div className="table-responsive">
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead style={{ background: 'var(--bg)' }}>
                    <tr>
                      <th style={{ padding: '1rem' }}>Doc Title</th>
                      <th style={{ padding: '1rem' }}>Type</th>
                      <th style={{ padding: '1rem' }}>Reference</th>
                      <th style={{ padding: '1rem' }}>Date</th>
                      <th style={{ padding: '1rem' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {transactions.map(tx => (
                      <tr key={tx.id} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td style={{ padding: '1rem', fontWeight: 600 }}>{tx.doc} for {tx.category}</td>
                        <td style={{ padding: '1rem' }}><span className="status-badge" style={{ background: 'var(--bg)', color: 'var(--primary)' }}>{tx.doc}</span></td>
                        <td style={{ padding: '1rem' }}>REF-{tx.id.toUpperCase()}</td>
                        <td style={{ padding: '1rem' }}>{tx.date}</td>
                        <td style={{ padding: '1rem' }}>
                          <div style={{ display: 'flex', gap: '0.5rem' }}>
                            <Download size={16} style={{ cursor: 'pointer', color: 'var(--text-muted)' }} />
                            <FileCheck size={16} style={{ cursor: 'pointer', color: 'var(--secondary)' }} />
                          </div>
                        </td>
                      </tr>
                    ))}
                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '1rem', fontWeight: 600 }}>KCB Bank Statement - March</td>
                      <td style={{ padding: '1rem' }}><span className="status-badge" style={{ background: 'var(--bg)', color: 'var(--primary)' }}>Statement</span></td>
                      <td style={{ padding: '1rem' }}>STMT-001</td>
                      <td style={{ padding: '1rem' }}>2026-03-31</td>
                      <td style={{ padding: '1rem' }}><Download size={16} style={{ cursor: 'pointer', color: 'var(--text-muted)' }} /></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        );
      default: // Cashbook
        return (
          <div className="animate-fade">
             <div className="responsive-grid" style={{ marginBottom: '2rem' }}>
                <div className="card" style={{ borderLeft: '5px solid var(--secondary)' }}>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Total Revenue (Term 2)</p>
                  <h2 style={{ margin: '0.5rem 0' }}>KES 12,450,000</h2>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--secondary)', fontSize: '0.75rem' }}>
                    <TrendingUp size={14} /> <span>12% Increase</span>
                  </div>
                </div>
                <div className="card" style={{ borderLeft: '5px solid #E02424' }}>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Total Expenditure</p>
                  <h2 style={{ margin: '0.5rem 0' }}>KES 4,200,500</h2>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#E02424', fontSize: '0.75rem' }}>
                    <TrendingDown size={14} /> <span>Within Budget</span>
                  </div>
                </div>
                <div className="card" style={{ borderLeft: '5px solid var(--accent)' }}>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Fee Arrears</p>
                  <h2 style={{ margin: '0.5rem 0' }}>KES 1,840,000</h2>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>42 Students Outstanding</p>
                </div>
              </div>

              <div className="card" style={{ padding: 0 }}>
                <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }} className="card-header-actions">
                  <h3 style={{ margin: 0 }}>Daily Cashbook / Transactions</h3>
                  <button className="btn" style={{ fontSize: '0.75rem', background: 'var(--bg)', border: '1px solid var(--border)' }}>Export CSV</button>
                </div>
                <div className="table-responsive">
                  <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                    <thead style={{ background: 'var(--bg)' }}>
                      <tr>
                        <th style={{ padding: '1rem' }}>Transaction ID</th>
                        <th style={{ padding: '1rem' }}>Category</th>
                        <th style={{ padding: '1rem' }}>Date</th>
                        <th style={{ padding: '1rem' }}>Amount</th>
                        <th style={{ padding: '1rem' }}>Status</th>
                        <th style={{ padding: '1rem' }}>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {transactions.map(tx => (
                        <tr key={tx.id} style={{ borderBottom: '1px solid var(--border)' }}>
                          <td style={{ padding: '1rem', fontWeight: 600 }}>{tx.id}</td>
                          <td style={{ padding: '1rem' }}>
                            <div style={{ fontWeight: 500 }}>{tx.category}</div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{tx.type}</div>
                          </td>
                          <td style={{ padding: '1rem' }}>{tx.date}</td>
                          <td style={{ padding: '1rem', color: tx.type === 'Income' ? 'var(--secondary)' : '#E02424', fontWeight: 600 }}>
                            {tx.type === 'Income' ? '+' : '-'} KES {tx.amount.toLocaleString()}
                          </td>
                          <td style={{ padding: '1rem' }}>
                            <span className="status-badge" style={{ background: tx.status === 'Paid' ? '#DEF7EC' : '#FEF3C7', color: tx.status === 'Paid' ? '#03543F' : '#92400E' }}>
                              {tx.status}
                            </span>
                          </td>
                          <td style={{ padding: '1rem' }}>
                            <Download size={16} style={{ cursor: 'pointer', color: 'var(--text-muted)' }} />
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
          <h1 style={{ fontSize: '1.8rem' }}>Financial Accounts</h1>
          <p style={{ color: 'var(--text-muted)' }}>Fees, Cashbooks, Budgets & Supporting Docs</p>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button className="btn" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <FileText size={18} />
            <span className="hide-mobile">Report</span>
          </button>
          <button className="btn btn-primary">
            <TrendingUp size={18} />
            Record Payment
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.5rem', overflowX: 'auto' }}>
        {[
          { id: 'Cashbook', icon: Wallet, label: 'Fees & Cashbook' },
          { id: 'Budgets', icon: PieChart, label: 'Budgets' },
          { id: 'SupportingDocs', icon: Layers, label: 'Supporting Docs' },
        ].map(tab => (
          <button 
            key={tab.id}
            onClick={() => setSubTab(tab.id)}
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem', 
              padding: '0.5rem 1rem', 
              border: 'none', 
              background: 'none', 
              cursor: 'pointer',
              color: subTab === tab.id ? 'var(--primary)' : 'var(--text-muted)',
              borderBottom: subTab === tab.id ? '2px solid var(--primary)' : '2px solid transparent',
              fontWeight: subTab === tab.id ? 600 : 400,
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
