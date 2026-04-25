import React, { useState } from 'react';
import { Package, ShoppingCart, Gavel, FileText, CheckCircle, AlertTriangle, Boxes, ClipboardList } from 'lucide-react';

const BOMTemplate = () => (
  <div className="card animate-fade" style={{ background: '#fff', border: '1px solid var(--border)', padding: '2rem', fontFamily: 'serif', maxWidth: '800px', margin: '0 auto' }}>
    <div style={{ textAlign: 'center', borderBottom: '2px solid var(--primary)', paddingBottom: '1rem', marginBottom: '2rem' }}>
      <h2 style={{ margin: 0, color: 'var(--primary)', textTransform: 'uppercase' }}>MZALENDO SCHOOLS</h2>
      <h3 style={{ margin: '5px 0', fontWeight: 600 }}>BOARD OF MANAGEMENT (BOM) MINUTES</h3>
    </div>
    
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem', fontSize: '0.9rem' }}>
      <div>
        <p><strong>DATE:</strong> ____________________</p>
        <p><strong>TIME:</strong> ____________________</p>
      </div>
      <div>
        <p><strong>VENUE:</strong> ____________________</p>
        <p><strong>CHAIR:</strong> ____________________</p>
      </div>
    </div>

    <div style={{ marginBottom: '1.5rem' }}>
      <h4 style={{ borderBottom: '1px solid #ccc', paddingBottom: '5px' }}>1. MEMBERS PRESENT</h4>
      <p style={{ minHeight: '60px', border: '1px dashed #eee', marginTop: '10px' }}></p>
    </div>

    <div style={{ marginBottom: '1.5rem' }}>
      <h4 style={{ borderBottom: '1px solid #ccc', paddingBottom: '5px' }}>2. AGENDA</h4>
      <ol style={{ marginTop: '10px' }}>
        <li>Preliminaries</li>
        <li>Confirmation of Previous Minutes</li>
        <li>Financial Report & Budget Approval</li>
        <li>Infrastructural Projects Update</li>
        <li>A.O.B</li>
      </ol>
    </div>

    <div style={{ marginBottom: '1.5rem' }}>
      <h4 style={{ borderBottom: '1px solid #ccc', paddingBottom: '5px' }}>3. MINUTE 001/2026: FINANCIAL APPROVALS</h4>
      <p style={{ minHeight: '80px', border: '1px dashed #eee', marginTop: '10px' }}></p>
    </div>

    <div style={{ marginTop: '3rem', display: 'flex', justifyContent: 'space-between' }}>
      <div style={{ textAlign: 'center' }}>
        <p>____________________</p>
        <p style={{ fontSize: '0.8rem' }}>CHAIRMAN</p>
      </div>
      <div style={{ textAlign: 'center' }}>
        <p>____________________</p>
        <p style={{ fontSize: '0.8rem' }}>SECRETARY / PRINCIPAL</p>
      </div>
    </div>
  </div>
);

export default function OperationsModule({ type }) {
  const [activeTab, setActiveTab] = useState('Overview');

  const contentMap = {
    Assets: {
      title: 'Asset Register',
      subtitle: 'Track school property and fixed assets',
      icon: <Package />,
      stats: [
        { label: 'Total Assets', value: '452' },
        { label: 'Under Maintenance', value: '18', color: '#92400E' },
        { label: 'Asset Value', value: 'KES 145M', color: 'var(--primary)' },
      ],
      tabs: ['Overview', 'Insurance', 'Service Log'],
      table: [
        { id: 'AS001', item: 'School Bus (KCB 001X)', cat: 'Transport', status: 'Operational', value: '5.2M' },
        { id: 'AS042', item: 'Computer Lab PCs', cat: 'IT', status: 'Service Due', value: '2.4M' },
        { id: 'AS105', item: 'Dining Hall Tables', cat: 'Furniture', status: 'Operational', value: '850K' },
      ]
    },
    Procurement: {
      title: 'Procurement & Tenders',
      subtitle: 'Manage tenders, contracts and purchases',
      icon: <ShoppingCart />,
      stats: [
        { label: 'Active Contracts', value: '8' },
        { label: 'Pending Tenders', value: '3', color: 'var(--accent)' },
        { label: 'Open LPOs', value: '5' },
      ],
      tabs: ['Overview', 'Tenders', 'LPOs'],
      table: [
        { id: 'TND-2026-04', item: 'New Science Lab Construction', cat: 'Tender', status: 'Evaluating', value: '12M' },
        { id: 'LPO-00842', item: 'Term 2 Textbooks', cat: 'Purchase Order', status: 'Delivered', value: '450K' },
        { id: 'CTR-0012', item: 'Security Services Contract', cat: 'Contract', status: 'Active', value: '120K/Mo' },
      ]
    },
    Governance: {
      title: 'Governance & BOM',
      subtitle: 'Board of Management minutes and approvals',
      icon: <Gavel />,
      stats: [
        { label: 'Minutes Recorded', value: '24' },
        { label: 'Policy Approvals', value: '12' },
        { label: 'Next Meeting', value: '24 May', color: 'var(--primary)' },
      ],
      tabs: ['Meeting Minutes', 'Policy Register', 'Approvals'],
      table: [
        { id: 'BOM104', item: 'Term 1 Strategy Meeting', cat: 'Meeting Minutes', status: 'Signed', date: '2026-03-10' },
        { id: 'APP022', item: 'Annual Budget 2026/27', cat: 'BOM Approval', status: 'Approved', date: '2026-04-02' },
        { id: 'MTG056', item: 'Infrastructure Development Review', cat: 'Meeting Agenda', status: 'Draft', date: '2026-05-24' },
      ]
    },
    Stock: {
      title: 'Stock & Inventory',
      subtitle: 'Track consumable items and supplies',
      icon: <Boxes />,
      stats: [
        { label: 'Low Stock Items', value: '14', color: '#E02424' },
        { label: 'Stock Value', value: 'KES 2.8M' },
        { label: 'Active Orders', value: '4' },
      ],
      tabs: ['Current Stock', 'Requisitions', 'Vendors'],
      table: [
        { id: 'STK001', item: 'A4 Printing Paper', cat: 'Stationery', status: 'Low Stock', qty: '12 Reams' },
        { id: 'STK042', item: 'Laboratory Chemicals', cat: 'Science Labs', status: 'In Stock', qty: '45 Units' },
        { id: 'STK105', item: 'Cleaning Detergents', cat: 'Sanitation', status: 'In Stock', qty: '200L' },
      ]
    }
  };

  const current = contentMap[type] || (type === 'Assets' ? contentMap.Assets : contentMap[type]);
  // Handle the 'Assets' case vs 'Stock' distinction from sidebar if needed
  // In App.jsx, 'Assets' is the activeTab, but user wants 'Stock Records' too.
  // I will show them as sub-sections if type is Assets.
  
  const displayData = type === 'Assets' && activeTab === 'Stock' ? contentMap.Stock : current;

  return (
    <div className="page-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }} className="card-header-actions">
        <div>
          <h1 style={{ fontSize: '1.8rem' }}>{displayData.title}</h1>
          <p style={{ color: 'var(--text-muted)' }}>{displayData.subtitle}</p>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
           <button className="btn" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <ClipboardList size={18} />
            <span className="hide-mobile">Full Report</span>
          </button>
          <button className="btn btn-primary">
            <FileText size={18} />
            New Record
          </button>
        </div>
      </div>

      {type === 'Assets' || type === 'Governance' ? (
        <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.5rem', overflowX: 'auto' }}>
          {(type === 'Assets' ? ['Overview', 'Stock', 'Insurance', 'History'] : ['Overview', 'Minutes Template', 'Policy Register']).map(tab => (
            <button 
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{ 
                border: 'none', 
                background: 'none', 
                padding: '0.5rem 1rem', 
                cursor: 'pointer',
                color: activeTab === tab ? 'var(--primary)' : 'var(--text-muted)',
                borderBottom: activeTab === tab ? '2px solid var(--primary)' : '2px solid transparent',
                fontWeight: activeTab === tab ? 600 : 400,
                whiteSpace: 'nowrap'
              }}
            >
              {tab}
            </button>
          ))}
        </div>
      ) : null}

      {activeTab === 'Minutes Template' && type === 'Governance' ? (
        <BOMTemplate />
      ) : (
        <>
          <div className="responsive-grid" style={{ marginBottom: '2rem' }}>
            {displayData.stats.map((s, i) => (
              <div key={i} className="card">
                <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>{s.label}</p>
                <h2 style={{ margin: '0.25rem 0', color: s.color || 'var(--text)' }}>{s.value}</h2>
              </div>
            ))}
          </div>

          <div className="card" style={{ padding: 0 }}>
            <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }} className="card-header-actions">
               <h3 style={{ margin: 0 }}>Records List</h3>
               <div style={{ position: 'relative' }} className="hide-mobile">
                  <input type="text" placeholder="Search records..." style={{ padding: '0.5rem 1rem', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '0.85rem' }} />
               </div>
            </div>
            <div className="table-responsive">
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead style={{ background: 'var(--bg)' }}>
                  <tr>
                    <th style={{ padding: '1rem' }}>ID</th>
                    <th style={{ padding: '1rem' }}>Description / Item</th>
                    <th style={{ padding: '1rem' }}>Category</th>
                    <th style={{ padding: '1rem' }}>{displayData.title.includes('Stock') ? 'Quantity' : 'Value/Date'}</th>
                    <th style={{ padding: '1rem' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {displayData.table.map(row => (
                    <tr key={row.id} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '1rem', fontWeight: 600 }}>{row.id}</td>
                      <td style={{ padding: '1rem' }}>{row.item}</td>
                      <td style={{ padding: '1rem' }}>{row.cat}</td>
                      <td style={{ padding: '1rem' }}>{row.value || row.qty || row.date}</td>
                      <td style={{ padding: '1rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                          {row.status.includes('Opera') || row.status.includes('Sign') || row.status.includes('Approv') || row.status.includes('Deliv') || row.status.includes('Stock') && !row.status.includes('Low')
                            ? <CheckCircle size={14} color="var(--secondary)" /> 
                            : <AlertTriangle size={14} color={row.status.includes('Low') || row.status.includes('Due') ? '#E02424' : 'var(--accent)'} />
                          }
                          <span style={{ color: row.status.includes('Low') || row.status.includes('Due') ? '#E02424' : 'inherit' }}>{row.status}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
