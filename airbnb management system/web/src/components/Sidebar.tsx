import React from 'react';
import { LayoutDashboard, Home, Calendar, Users, FileText, Settings } from 'lucide-react';
import { cn } from '../lib/utils';

const navItems = [
  { icon: LayoutDashboard, label: 'Dashboard', active: true },
  { icon: Home, label: 'Properties' },
  { icon: Calendar, label: 'Bookings' },
  { icon: Users, label: 'Admins' },
  { icon: FileText, label: 'Receipts' },
  { icon: Settings, label: 'Settings' },
];

export const Sidebar = () => {
  return (
    <div className="w-64 h-screen glass border-r border-white/5 flex flex-col p-6 fixed left-0 top-0">
      <div className="flex items-center gap-2 mb-10 px-2">
        <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center font-bold text-white">M</div>
        <span className="text-xl font-bold tracking-tight text-white">MogulPMS</span>
      </div>
      
      <nav className="flex-1 space-y-2">
        {navItems.map((item) => (
          <a
            key={item.label}
            href="#"
            className={cn(
              "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group text-white",
              item.active 
                ? "bg-accent/10 text-accent border border-accent/20" 
                : "text-slate-400 hover:bg-white/5 hover:text-white"
            )}
          >
            <item.icon size={20} className={cn(item.active ? "text-accent" : "group-hover:text-white")} />
            <span className="font-medium">{item.label}</span>
          </a>
        ))}
      </nav>
      
      <div className="mt-auto px-2 py-4 border-t border-white/5">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-accent to-blue-500" />
          <div className="text-white">
            <p className="text-sm font-semibold">Nigel Mogul</p>
            <p className="text-xs text-slate-500">Super Admin</p>
          </div>
        </div>
      </div>
    </div>
  );
};
