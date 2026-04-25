import React from 'react';
import { Mail, ArrowLeft, ShieldCheck } from 'lucide-react';
import { Link } from 'react-router-dom';

export const SignUpPage = () => {
  return (
    <div className="min-h-screen bg-background text-white font-sans flex flex-col items-center justify-center px-6 py-12">
      {/* Back to Home */}
      <Link to="/" className="absolute top-8 left-8 flex items-center gap-2 text-slate-400 hover:text-white transition-colors font-bold text-sm">
        <ArrowLeft size={16} /> Back
      </Link>

      <div className="w-full max-w-sm space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
        <div className="text-center">
          <div className="w-16 h-16 bg-accent rounded-3xl mx-auto mb-6 flex items-center justify-center shadow-lg shadow-accent/20 rotate-12">
            <ShieldCheck size={32} />
          </div>
          <h1 className="text-3xl font-black mb-2 uppercase tracking-tighter">Join the Elite</h1>
          <p className="text-slate-400 text-sm">Enter your property email to start your 14-day premium trial.</p>
        </div>

        <form className="space-y-4" onSubmit={(e) => e.preventDefault()}>
          <div className="space-y-2">
            <label className="text-xs font-black uppercase tracking-widest text-slate-500 px-1">Email Address</label>
            <div className="relative group">
              <Mail className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-accent transition-colors" size={18} />
              <input 
                type="email" 
                placeholder="mogul@luxury-stays.com" 
                className="w-full bg-white/5 border border-white/10 rounded-2xl py-4 pl-12 pr-4 outline-none focus:ring-4 focus:ring-accent/10 focus:border-accent transition-all text-lg"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-black uppercase tracking-widest text-slate-500 px-1">Business Name</label>
            <input 
              type="text" 
              placeholder="e.g. Prestige Rentals" 
              className="w-full bg-white/5 border border-white/10 rounded-2xl py-4 px-6 outline-none focus:ring-4 focus:ring-accent/10 focus:border-accent transition-all text-lg"
            />
          </div>

          <button className="w-full bg-accent hover:bg-accent/90 text-white py-5 rounded-2xl font-black text-lg transition-all shadow-xl shadow-accent/20 active:scale-95 mt-4">
            Create Mogul Account
          </button>
        </form>

        <p className="text-center text-xs text-slate-500 font-medium">
          By joining, you agree to our <span className="text-slate-300 underline">Terms of Management</span> and <span className="text-slate-300 underline">Privacy Policy</span>.
        </p>

        <div className="p-4 glass rounded-2xl border-accent/20 text-center">
          <p className="text-[10px] font-black uppercase tracking-widest text-accent mb-1 italic">Flash Sale</p>
          <p className="text-sm font-bold">50% Legacy Discount if you join today.</p>
        </div>
      </div>
    </div>
  );
};
