import { motion } from "framer-motion";
import { ArrowRight, Star, Shield, Zap } from "lucide-react";
import { Link } from "react-router-dom";
import HeroSlider from "../components/HeroSlider";

const servicesPreview = [
  {
    title: "Outdoor Advertising",
    description: "High-impact billboards and large-format signage that commands attention in any environment.",
    icon: Shield,
    image: "/assets/AAS Ambulance 2.png"
  },
  {
    title: "Apparel & Merchandise",
    description: "Premium custom clothing and branded goods with retail-quality finishes.",
    icon: Zap,
    image: "/assets/Captain Evan cap.png"
  },
  {
    title: "Corporate Identity",
    description: "Complete stationery sets and internal branding that defines professional excellence.",
    icon: Star,
    image: "/assets/Rivergate Business cards.png"
  }
];

export default function Home() {
  return (
    <div className="overflow-hidden">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center pt-24 px-6 md:pt-32">
        {/* Background Texture/Gradient */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(212,175,55,0.05),transparent_50%)]" />
        <div className="absolute top-0 right-0 w-1/2 h-full bg-gradient-to-l from-brand-gold/5 to-transparent pointer-events-none" />

        <div className="max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-2 gap-16 items-center relative z-10">
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-brand-gold/20 bg-brand-gold/5 text-brand-gold text-sm font-semibold mb-8">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-gold opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-gold"></span>
              </span>
              Your Brand's Guardian
            </div>
            <h1 className="text-5xl md:text-7xl lg:text-8xl mb-6 leading-[1.1] text-brand-black dark:text-brand-white-off transition-colors">
              Elevating Brands to <span className="text-gradient-gold">Authoritative</span> Reality
            </h1>
            <p className="text-xl text-brand-black/60 dark:text-brand-white-off/60 mb-10 max-w-xl leading-relaxed transition-colors">
              We create cinematic branding experiences through high-impact outdoor signage, premium merchandise, and world-class corporate identity.
            </p>
            <div className="flex flex-wrap gap-4">
              <Link to="/portfolio" className="btn-gold group">
                View Showcase 
                <ArrowRight size={20} className="inline-block ml-2 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link to="/contact" className="btn-outline">
                Request a Quote
              </Link>
            </div>
          </motion.div>

          {/* Hero Slider */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
            className="w-full"
          >
            <HeroSlider />
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-24 border-y border-brand-black/5 dark:border-brand-white/5 bg-brand-gold/5 dark:bg-brand-black-subtle/30 transition-colors">
        <div className="max-w-7xl mx-auto px-6 grid grid-cols-2 md:grid-cols-4 gap-12 text-center">
          {[
            { label: "Projects Delivered", value: "500+" },
            { label: "Global Clients", value: "120+" },
            { label: "Design Awards", value: "15" },
            { label: "Years Experience", value: "12" }
          ].map((stat, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              viewport={{ once: true }}
            >
              <div className="text-4xl md:text-5xl font-display font-black text-brand-gold mb-2">{stat.value}</div>
              <div className="text-brand-black/40 dark:text-brand-white-off/40 uppercase tracking-widest text-xs font-bold transition-colors">{stat.label}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Services Preview Grid */}
      <section className="py-32 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-20">
            <h2 className="text-4xl md:text-5xl mb-6 text-brand-black dark:text-brand-white-off transition-colors">Our <span className="text-gradient-gold">Services</span></h2>
            <p className="text-brand-black/60 dark:text-brand-white-off/60 max-w-2xl mx-auto text-lg leading-relaxed transition-colors">
              From concept to physical execution, we provide a full spectrum of branding solutions that resonate with authority.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {servicesPreview.map((service, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.2 }}
                viewport={{ once: true }}
                className="glass-card group"
              >
                <div className="h-64 overflow-hidden relative">
                  <img src={service.image} alt={service.title} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" />
                  <div className="absolute inset-0 bg-gradient-to-t from-brand-black to-transparent opacity-60" />
                </div>
                <div className="p-8">
                  <div className="w-12 h-12 bg-brand-gold/10 rounded-lg flex items-center justify-center mb-6 border border-brand-gold/20">
                    <service.icon size={24} className="text-brand-gold" />
                  </div>
                  <h3 className="text-2xl mb-4 text-brand-black dark:text-brand-white-off transition-colors">{service.title}</h3>
                  <p className="text-brand-black/60 dark:text-brand-white-off/60 leading-relaxed mb-6 transition-colors">
                    {service.description}
                  </p>
                  <Link to="/services" className="text-brand-gold font-bold inline-flex items-center group-hover:translate-x-2 transition-transform">
                    Learn More <ArrowRight size={18} className="ml-2" />
                  </Link>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Recent Work Showcase */}
      <section className="py-32 px-6 border-t border-brand-white/5 bg-brand-black-subtle/20">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row items-end justify-between mb-20 gap-8">
            <div className="max-w-xl">
              <h2 className="text-4xl md:text-5xl mb-6 text-brand-black dark:text-brand-white-off transition-colors">Recent <span className="text-gradient-gold">Signature</span> Work</h2>
              <p className="text-brand-black/60 dark:text-brand-white-off/60 text-lg transition-colors">
                Discover how we transformed branding for Alpha Ambulance, Nabila Scented Candles, and more.
              </p>
            </div>
            <Link to="/portfolio" className="btn-outline group">
              View All Projects
              <ArrowRight size={18} className="ml-2 inline-block group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { img: "/assets/AAS Ambulance 3.png", client: "Alpha Ambulance", type: "Outdoor" },
              { img: "/assets/Nabila candle jars.png", client: "Nabila Candles", type: "Lifestyle" },
              { img: "/assets/Gio-01.png", client: "Gio's Kitchen", type: "Corporate" },
              { img: "/assets/MAU MAU BOOKS.png", client: "Books", type: "Editorial" }
            ].map((work, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                className="relative aspect-[3/4] rounded-2xl overflow-hidden group border border-brand-white/5"
              >
                <img src={work.img} alt={work.client} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700" />
                <div className="absolute inset-0 bg-gradient-to-t from-brand-black via-transparent to-transparent opacity-80" />
                <div className="absolute bottom-6 left-6">
                  <span className="text-brand-gold font-bold text-xs uppercase tracking-widest mb-1 block opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    {work.type}
                  </span>
                  <p className="text-xl font-display font-bold text-brand-white-off">
                    {work.client}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-32 px-6">
        <div className="max-w-5xl mx-auto bg-gold-gradient p-[1px] rounded-3xl overflow-hidden shadow-2xl shadow-brand-gold/20">
          <div className="bg-brand-black dark:bg-brand-black px-8 py-20 md:p-20 flex flex-col items-center text-center">
            <h2 className="text-4xl md:text-6xl mb-8 leading-tight text-brand-white-off dark:text-brand-white-off transition-colors">Ready to <span className="text-gradient-gold">Define</span> Your Brand Authority?</h2>
            <p className="text-xl text-brand-white-off/60 dark:text-brand-white-off/60 mb-12 max-w-2xl leading-relaxed transition-colors">
              Join elite brands that have transformed their market presence with our cinematic branding strategies.
            </p>
            <div className="flex flex-wrap justify-center gap-6">
              <Link to="/contact" className="btn-gold !py-4 !px-12 text-lg">
                Start a Project
              </Link>
              <Link to="/portfolio" className="btn-outline !py-4 !px-12 text-lg">
                Explore Work
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
