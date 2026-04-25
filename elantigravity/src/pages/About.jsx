import { motion } from "framer-motion";
import { Award, Users, Heart, Target, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";

export default function About() {
  return (
    <div className="pt-32 pb-24 px-6 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-20 items-center mb-32">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h1 className="text-5xl md:text-7xl mb-8 font-display text-brand-black dark:text-brand-white-off transition-colors">The <span className="text-gradient-gold">Force</span> Behind The Brand</h1>
            <p className="text-xl text-brand-black/60 dark:text-brand-white-off/60 mb-6 leading-relaxed transition-colors">
              Ela Creatives was founded on a simple principle: branding should be cinematic. It should command attention, inspire trust, and radiate authority.
            </p>
            <p className="text-lg text-brand-black/40 dark:text-brand-white-off/40 mb-10 leading-relaxed transition-colors">
              We started as a small design studio in 2014, focus on high-end physical branding. Today, we are a full-service agency that specializes in bridging the gap between digital strategy and tangible impact.
            </p>
            <div className="flex gap-12">
              <div>
                <div className="text-4xl font-display font-black text-brand-gold mb-1">10+</div>
                <div className="text-xs text-brand-black/40 dark:text-brand-white-off/40 uppercase tracking-widest font-bold transition-colors">Years of Craft</div>
              </div>
              <div>
                <div className="text-4xl font-display font-black text-brand-gold mb-1">2k+</div>
                <div className="text-xs text-brand-black/40 dark:text-brand-white-off/40 uppercase tracking-widest font-bold transition-colors">Projects Completed</div>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="relative"
          >
            <div className="rounded-3xl shadow-2xl shadow-brand-gold/10 overflow-hidden border border-brand-white/10 aspect-square lg:aspect-auto h-[600px]">
              <img src="/assets/hero-merch.png" alt="Agency Studio" className="w-full h-full object-cover grayscale hover:grayscale-0 transition-all duration-1000" />
            </div>
            <div className="absolute -bottom-6 -right-6 bg-brand-gold p-8 rounded-2xl text-brand-black shadow-xl shadow-brand-gold/20 max-w-[280px]">
              <Heart className="mb-4" size={32} />
              <p className="font-bold text-lg leading-tight">Committed to premium quality in every single thread and pixel.</p>
            </div>
          </motion.div>
        </div>

        {/* Values Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-32">
          {[
            { 
              title: "Cinematic Aesthetic", 
              desc: "We don't do 'simple'. We do visual poetry that elevates your brand to an authoritative position.", 
              icon: Target 
            },
            { 
              title: "Physical Excellence", 
              desc: "Quality isn't optional. From material selection to printing technique, we demand the best.", 
              icon: Award 
            },
            { 
              title: "Client-First Strategy", 
              desc: "We are your partners in growth. Your success is the only metric that matters to us.", 
              icon: Users 
            }
          ].map((val, i) => (
            <motion.div
              key={i}
              whileHover={{ y: -10 }}
              className="glass-card p-10 flex flex-col items-center text-center"
            >
              <div className="w-14 h-14 bg-brand-gold/10 rounded-full flex items-center justify-center mb-6 text-brand-gold border border-brand-gold/20">
                <val.icon size={28} />
              </div>
              <h3 className="text-2xl mb-4 font-display text-brand-black dark:text-brand-white-off transition-colors">{val.title}</h3>
              <p className="text-brand-black/60 dark:text-brand-white-off/60 leading-relaxed transition-colors">{val.desc}</p>
            </motion.div>
          ))}
        </div>

        {/* Contact CTA */}
        <div className="bg-brand-gold/5 dark:bg-brand-black-subtle/50 rounded-3xl p-12 md:p-20 text-center border border-brand-black/5 dark:border-brand-white/5 transition-colors">
          <h2 className="text-4xl md:text-5xl mb-8 text-brand-black dark:text-brand-white-off transition-colors">Ready to <span className="text-gradient-gold">Collaborate?</span></h2>
          <p className="text-xl text-brand-black/60 dark:text-brand-white-off/60 mb-12 max-w-2xl mx-auto leading-relaxed transition-colors">
            Let's build something exceptional together. Our team is ready to take your brand to the next level.
          </p>
          <Link to="/contact" className="btn-gold !py-4 !px-12 text-lg">
            Connect With Us <ArrowRight size={20} className="inline-block ml-2" />
          </Link>
        </div>
      </div>
    </div>
  );
}
