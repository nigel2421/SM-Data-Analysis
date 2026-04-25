import { motion } from "framer-motion";
import { CheckCircle2, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { servicesData } from "../data/servicesData";

export default function Services() {
  return (
    <div className="pt-32 pb-24 px-6 min-h-screen overflow-hidden">
      <div className="max-w-7xl mx-auto">
        <header className="mb-24 text-center max-w-4xl mx-auto text-brand-black dark:text-brand-white-off transition-colors">
          <motion.h1 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-6xl md:text-8xl mb-8 font-display font-black text-brand-black dark:text-brand-white-off"
          >
            Brand <span className="text-gradient-gold">Capabilities</span>
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-2xl text-brand-black/60 dark:text-brand-white-off/60 leading-relaxed"
          >
            We bridge the gap between digital vision and physical reality. Our services are tailored for brands that demand authority and elegance.
          </motion.p>
        </header>

        <div className="space-y-32">
          {servicesData.map((service, i) => {
            const Icon = service.icon;
            return (
              <motion.section
                key={service.id}
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                viewport={{ once: true, margin: "-100px" }}
                className={`flex flex-col ${i % 2 === 1 ? 'lg:flex-row-reverse' : 'lg:flex-row'} items-center gap-16 lg:gap-24`}
              >
                {/* Content */}
                <div className="flex-1">
                  <div className="w-16 h-16 bg-brand-gold/10 rounded-2xl flex items-center justify-center mb-8 border border-brand-gold/20">
                    <Icon size={32} className="text-brand-gold" />
                  </div>
                  <h2 className="text-4xl md:text-5xl mb-4 font-display text-brand-black dark:text-brand-white-off transition-colors">{service.title}</h2>
                  <h4 className="text-brand-gold font-bold text-lg mb-6 uppercase tracking-widest">{service.subtitle}</h4>
                  <p className="text-xl text-brand-black/60 dark:text-brand-white-off/60 mb-10 leading-relaxed">
                    {service.description}
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-10">
                    {service.features.slice(0, 4).map((feature, idx) => (
                      <div key={idx} className="flex items-center gap-3 text-brand-black/80 dark:text-brand-white-off/80 font-medium transition-colors">
                        <CheckCircle2 size={20} className="text-brand-gold shrink-0" />
                        {feature.title}
                      </div>
                    ))}
                  </div>
                  <Link to={`/services/${service.slug}`} className="btn-gold group inline-flex items-center">
                    Explore Service <ArrowRight size={20} className="ml-2 group-hover:translate-x-1 transition-transform" />
                  </Link>
                </div>

                {/* Visuals */}
                <div className="flex-1 w-full relative group">
                  <div className="relative z-10 rounded-[2.5rem] overflow-hidden shadow-2xl shadow-brand-gold/10 border border-brand-black/10 dark:border-brand-white/10 aspect-[4/3]">
                    <img 
                      src={service.image} 
                      alt={service.title} 
                      className="w-full h-full object-cover transform hover:scale-105 transition-transform duration-1000" 
                    />
                    <div className="absolute inset-0 bg-gradient-to-tr from-brand-black/40 via-transparent to-transparent pointer-events-none" />
                  </div>
                  <div className="absolute -top-10 -left-10 w-full h-full bg-brand-gold/5 rounded-[2.5rem] -rotate-3 -z-10 group-hover:rotate-0 transition-transform duration-500" />
                </div>
              </motion.section>
            );
          })}
        </div>
      </div>
    </div>
  );
}
