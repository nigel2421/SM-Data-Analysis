import { useParams, Link, Navigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, CheckCircle2, MessageSquare, ArrowRight } from "lucide-react";
import { servicesData } from "../data/servicesData";
import ServiceCarousel from "../components/ServiceCarousel";

export default function ServiceDetail() {
  const { slug } = useParams();
  const service = servicesData.find((s) => s.slug === slug);

  if (!service) {
    return <Navigate to="/services" replace />;
  }

  const Icon = service.icon;
  const waNumber = "254700000000"; // Real agency number
  const waMessage = encodeURIComponent(`Hello Ela Creatives! I'm interested in your ${service.title} services. Could you share more details about the options for ${service.features[0].title}?`);
  const waLink = `https://wa.me/${waNumber}?text=${waMessage}`;

  return (
    <div className="pt-32 pb-24 px-6 min-h-screen">
      <div className="max-w-7xl mx-auto">
        {/* Back Button */}
        <Link 
          to="/services" 
          className="inline-flex items-center gap-2 text-brand-gold font-bold mb-12 hover:translate-x-1 transition-transform"
        >
          <ArrowLeft size={20} /> Back to Services
        </Link>

        {/* Hero Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center mb-32">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <div className="w-16 h-16 bg-brand-gold/10 rounded-2xl flex items-center justify-center mb-8 border border-brand-gold/20">
              <Icon size={32} className="text-brand-gold" />
            </div>
            <h1 className="text-5xl md:text-7xl mb-6 font-display font-black leading-tight text-brand-black dark:text-brand-white-off transition-colors">
              {service.title}
            </h1>
            <h4 className="text-brand-gold font-bold text-xl mb-8 uppercase tracking-[0.2em]">
              {service.subtitle}
            </h4>
            <p className="text-xl text-brand-black/60 dark:text-brand-white-off/60 leading-relaxed mb-10 transition-colors">
              {service.longDescription}
            </p>
            <div className="flex flex-wrap gap-4">
              <Link to="/contact" className="btn-gold group">
                Start a Project
                <ArrowRight size={20} className="inline-block ml-2 group-hover:translate-x-1 transition-transform" />
              </Link>
              <a 
                href={waLink} 
                target="_blank" 
                rel="noreferrer" 
                className="btn-outline flex items-center gap-2"
              >
                <MessageSquare size={20} /> WhatsApp Expert
              </a>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="relative"
          >
            <div className="rounded-[2.5rem] overflow-hidden border border-brand-white/10 shadow-2xl shadow-brand-gold/10 aspect-square lg:aspect-[4/5]">
              <img 
                src={service.image} 
                alt={service.title} 
                className="w-full h-full object-cover"
              />
            </div>
            {/* Decorative background glow */}
            <div className="absolute -inset-10 bg-brand-gold/5 blur-3xl -z-10 rounded-full" />
          </motion.div>
        </div>

        {/* Features Grid */}
        <section className="mb-32">
          <h2 className="text-3xl md:text-5xl mb-16 text-center text-brand-black dark:text-brand-white-off transition-colors">Core <span className="text-gradient-gold">Capabilities</span></h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {service.features.map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="glass-card p-8 flex gap-6"
              >
                <div className="shrink-0 w-12 h-12 rounded-full border border-brand-gold/20 flex items-center justify-center bg-brand-gold/5">
                  <CheckCircle2 size={24} className="text-brand-gold" />
                </div>
                <div>
                  <h3 className="text-2xl mb-2 text-brand-black dark:text-brand-white-off transition-colors">{feature.title}</h3>
                  <p className="text-brand-black/60 dark:text-brand-white-off/60 leading-relaxed transition-colors">{feature.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Showcase Gallery */}
        <section className="mb-32">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-5xl mb-4 text-brand-black dark:text-brand-white-off transition-colors">The <span className="text-gradient-gold">Showcase</span></h2>
            <p className="text-brand-black/40 dark:text-brand-white-off/40 font-bold uppercase tracking-widest text-sm transition-colors">Real projects for real clients</p>
          </div>
          
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <ServiceCarousel images={service.gallery} />
          </motion.div>
        </section>

        {/* CTA Banner */}
        <section className="bg-gold-gradient p-[1px] rounded-3xl overflow-hidden shadow-2xl shadow-brand-gold/10">
          <div className="bg-brand-black dark:bg-brand-black p-12 md:p-20 text-center flex flex-col items-center">
            <h2 className="text-3xl md:text-5xl mb-8 text-brand-white-off transition-colors">Ready to start your <span className="text-gradient-gold">{service.title}</span> project?</h2>
            <p className="text-brand-white-off/60 mb-12 max-w-2xl text-lg transition-colors">
              Partner with the industry's branding guardians to elevate your presence with authoritative design and physical execution.
            </p>
            <div className="flex flex-wrap justify-center gap-6">
              <Link to="/contact" className="btn-gold !px-12">
                Get a Quote
              </Link>
              <a href={waLink} className="btn-outline !px-12">
                Chat with Us
              </a>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
