import React, { useState } from 'react';
import { Activity, Home, Calendar, Users, TrendingUp, Search, Bell } from 'lucide-react';
import { Sidebar } from '../components/Sidebar';
import { ReceiptGenerator } from '../components/ReceiptGenerator';

const stats = [
  { label: 'Total Units', value: '24', icon: Home, trend: '+2 this month' },
  { label: 'Occupancy', value: '88%', icon: Activity, trend: '+5% vs last month' },
  { label: 'Revenue', value: '$42,500', icon: TrendingUp, trend: '+12% growth' },
  { label: 'Active Tasks', value: '8', icon: Users, trend: '4 due today' },
];

export const Dashboard = () => {
  const [selectedReceipt, setSelectedReceipt] = useState<any>(null);
  return (
    <div className="flex min-h-screen bg-background text-white font-sans">
      <Sidebar />
      
      <main className="flex-1 ml-64 p-8">
        {/* Top Navbar */}
        <header className="flex justify-between items-center mb-10">
          <div>
            <h1 className="text-3xl font-bold mb-2">Command Center</h1>
            <p className="text-slate-400">Welcome back, Nigel. Here's your property overview.</p>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="relative group">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-accent transition-colors" size={18} />
              <input 
                type="text" 
                placeholder="Search units..." 
                className="bg-white/5 border border-white/10 rounded-xl py-2.5 pl-10 pr-4 outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition-all w-64"
              />
            </div>
            <button className="p-2.5 glass rounded-xl hover:bg-white/10 transition-colors relative">
              <Bell size={20} />
              <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full border-2 border-background" />
            </button>
            <button className="bg-accent hover:bg-accent/90 text-white px-6 py-2.5 rounded-xl font-semibold transition-all shadow-lg shadow-accent/20">
              + New Unit
            </button>
          </div>
        </header>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
          {stats.map((stat) => (
            <div key={stat.label} className="glass p-6 rounded-2xl hover:border-accent/30 transition-colors">
              <div className="flex justify-between items-start mb-4">
                <div className="p-3 bg-white/5 rounded-xl">
                  <stat.icon size={24} className="text-accent" />
                </div>
                <span className="text-xs font-medium text-accent bg-accent/10 px-2 py-1 rounded-lg">
                  {stat.trend}
                </span>
              </div>
              <p className="text-slate-400 text-sm font-medium">{stat.label}</p>
              <h3 className="text-2xl font-bold mt-1 uppercase tracking-tight">{stat.value}</h3>
            </div>
          ))}
        </div>

        {/* Active Units Table */}
        <div className="glass rounded-2xl overflow-hidden border border-white/5">
          <div className="p-6 border-b border-white/5 flex justify-between items-center bg-white/[0.02]">
            <h2 className="text-xl font-bold">Property Status Monitor</h2>
            <div className="flex gap-2">
              <button className="px-3 py-1 text-xs font-semibold rounded-lg bg-white/5 text-slate-400 hover:text-white transition-colors">All</button>
              <button className="px-3 py-1 text-xs font-semibold rounded-lg text-accent bg-accent/10 transition-colors">Occupied</button>
              <button className="px-3 py-1 text-xs font-semibold rounded-lg bg-white/5 text-slate-400 hover:text-white transition-colors">Maintenance</button>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead className="bg-white/5 text-slate-400 text-sm font-medium">
                <tr>
                  <th className="px-6 py-4 uppercase tracking-wider text-[10px] font-bold">Unit Name</th>
                  <th className="px-6 py-4 uppercase tracking-wider text-[10px] font-bold">Status</th>
                  <th className="px-6 py-4 uppercase tracking-wider text-[10px] font-bold">Guest</th>
                  <th className="px-6 py-4 uppercase tracking-wider text-[10px] font-bold">Pricing</th>
                  <th className="px-6 py-4 uppercase tracking-wider text-[10px] font-bold">Revenue</th>
                  <th className="px-6 py-4 uppercase tracking-wider text-[10px] font-bold text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {[
                  { name: 'Unit 4B', status: 'Occupied', color: 'bg-green-500', guest: 'John Doe', price: 'Dynamic', revenue: '$1,200', active: true },
                  { name: 'Studio C', status: 'Maintenance', color: 'bg-red-500', guest: '-', price: 'Fixed', revenue: '$850' },
                  { name: 'Villa Rose', status: 'Turnover', color: 'bg-yellow-500', guest: 'Sarah Lee', price: 'Dynamic', revenue: '$2,100' },
                  { name: 'Apartment 12', status: 'Occupied', color: 'bg-green-500', guest: 'Mike Ross', price: 'Dynamic', revenue: '$1,450' },
                ].map((row) => (
                  <tr key={row.name} className="hover:bg-white/[0.03] transition-colors cursor-pointer group">
                    <td className="px-6 py-4 font-semibold group-hover:text-accent transition-colors">{row.name}</td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${row.color} shadow-[0_0_8px_rgba(0,0,0,0.5)] shadow-${row.color.split('-')[1]}-500/50`} />
                        <span className="text-sm font-medium">{row.status}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-slate-400 text-sm">{row.guest}</td>
                    <td className="px-6 py-4">
                      <span className="text-[10px] px-2 py-1 bg-white/5 rounded-lg border border-white/10 uppercase font-black tracking-widest text-slate-300">
                        {row.price}
                      </span>
                    </td>
                    <td className="px-6 py-4 font-mono font-bold text-accent">{row.revenue}</td>
                    <td className="px-6 py-4 text-right">
                      <button 
                        onClick={() => setSelectedReceipt({
                          unitName: row.name,
                          guestName: row.guest === '-' ? 'N/A' : row.guest,
                          amount: row.revenue,
                          date: '2026-03-24',
                          receiptId: `REC-${row.name.replace(' ', '')}-001`
                        })}
                        className="text-xs font-bold text-slate-400 hover:text-accent transition-colors p-2 hover:bg-accent/10 rounded-lg"
                      >
                        Receipt
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </main>

      {/* Receipt Modal */}
      {selectedReceipt && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="relative w-full max-w-2xl transform animate-in slide-in-from-bottom-8 duration-300">
            <button 
              onClick={() => setSelectedReceipt(null)}
              className="absolute -top-12 right-0 text-white/50 hover:text-white flex items-center gap-2 font-bold mb-4"
            >
              Close [ESC]
            </button>
            <ReceiptGenerator {...selectedReceipt} />
          </div>
        </div>
      )}
    </div>
  );
};
