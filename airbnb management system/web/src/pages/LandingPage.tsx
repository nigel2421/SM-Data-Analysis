import React from 'react';
import { Shield, Zap, BarChart3, ArrowRight, UserPlus } from 'lucide-react';
import { Link } from 'react-router-dom';

export const LandingPage = () => {
  return (
    <div className="min-h-screen bg-background text-white font-sans overflow-x-hidden">
      {/* Navbar */}
      <nav className="p-6 flex justify-between items-center glass sticky top-0 z-50">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center font-bold">M</div>
          <span className="text-xl font-bold">MogulPMS</span>
        </div>
        <Link to="/signup" className="bg-accent px-5 py-2 rounded-xl text-sm font-bold shadow-lg shadow-accent/20 hover:bg-accent/90 transition-all">
          Join Now
        </Link>
      </nav>

      {/* Hero Section */}
      <header className="px-6 py-20 text-center max-w-4xl mx-auto">
        <div className="inline-flex items-center gap-2 bg-accent/10 text-accent px-4 py-1.5 rounded-full text-xs font-bold mb-6 border border-accent/20">
          <Zap size={14} /> 2026 SaaS of the Year nominee
        </div>
        <h1 className="text-5xl md:text-7xl font-black mb-6 leading-tight tracking-tight">
          Turn Your Airbnb into a <span className="text-accent underline decoration-4 underline-offset-8">Mogul Empire</span>
        </h1>
        <p className="text-slate-400 text-lg md:text-xl mb-10 max-w-2xl mx-auto leading-relaxed">
          The only property management system that automates turnover, pricing, and guest experience using high-end logistics agents. Build for owners, by owners.
        </p>
        <div className="flex flex-col md:flex-row gap-4 justify-center">
          <Link to="/signup" className="bg-white text-background px-8 py-4 rounded-2xl font-bold flex items-center justify-center gap-2 hover:bg-slate-200 transition-all group lg:text-lg">
            Start Free Trial <ArrowRight className="group-hover:translate-x-1 transition-transform" />
          </Link>
          <button className="glass px-8 py-4 rounded-2xl font-bold hover:bg-white/10 transition-all lg:text-lg">
            Watch Demo
          </button>
        </div>
      </header>

      {/* Value Props Grid */}
      <section className="px-6 py-20 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
        {[
          { icon: Shield, title: 'Zero-Stress Turnover', desc: 'Our Logistics Agent alerts cleaning crews automatically upon guest checkout.' },
          { icon: Zap, title: 'Instant Sign-up', desc: 'Start managing your properties in under 2 minutes with our seamless onboarding.' },
          { icon: BarChart3, title: 'Revenue Optimization', desc: 'Dynamic pricing engine that out-performs competitors by 22% on average.' },
        ].map((prop, i) => (
          <div key={i} className="glass p-8 rounded-3xl group hover:border-accent/30 transition-all">
            <div className="p-4 bg-accent/10 text-accent rounded-2xl w-fit mb-6 group-hover:scale-110 transition-transform">
              <prop.icon size={28} />
            </div>
            <h3 className="text-xl font-bold mb-3">{prop.title}</h3>
            <p className="text-slate-400 leading-relaxed text-sm">{prop.desc}</p>
          </div>
        ))}
      </section>

      {/* Social Proof Section */}
      <section className="px-6 py-10 text-center border-y border-white/5 bg-white/[0.02]">
        <div className="flex flex-wrap justify-center gap-12 opacity-40 grayscale pointer-events-none">
          {['Airbnb Management', 'Realtor Pro', 'Elite Stays', 'Superhost Central'].map(brand => (
            <span key={brand} className="text-lg font-black tracking-widest">{brand}</span>
          ))}
        </div>
      </section>

      {/* Footer CTA */}
      <section className="px-6 py-32 text-center bg-gradient-to-b from-transparent to-accent/5">
        <div className="bg-accent/10 p-4 w-12 h-12 rounded-2xl mx-auto mb-8 flex items-center justify-center">
          <UserPlus className="text-accent" />
        </div>
        <h2 className="text-3xl md:text-4xl font-bold mb-6">Ready to Scale Your Portfolio?</h2>
        <p className="text-slate-400 mb-10">Limited beta slots available for property moguls.</p>
        <Link to="/signup" className="bg-accent px-10 py-4 rounded-2xl font-bold text-lg inline-block shadow-xl shadow-accent/20 hover:scale-105 transition-transform">
          Register with Email
        </Link>
      </section>
    </div>
  );
};
