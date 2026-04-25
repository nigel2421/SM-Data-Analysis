import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeft, ChevronRight } from "lucide-react";

const slides = [
  {
    image: "/assets/AAS Ambulance 1.png",
    title: "Alpha Ambulance Services",
    subtitle: "High-impact emergency branding"
  },
  {
    image: "/assets/Billboards.png",
    title: "Urban Impact",
    subtitle: "Commanding city-wide attention"
  },
  {
    image: "/assets/Nabila candle_with_box.png",
    title: "Nabila Luxury Candles",
    subtitle: "Premium artisanal packaging"
  },
  {
    id: "book-1",
    image: "/assets/MAU MAU BOOKS.png",
    title: "Historical Chronicles",
    subtitle: "Authoritative book cover design"
  }
];

export default function HeroSlider() {
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrent((prev) => (prev + 1) % slides.length);
    }, 6000);
    return () => clearInterval(timer);
  }, []);

  const next = () => setCurrent((prev) => (prev + 1) % slides.length);
  const prev = () => setCurrent((prev) => (prev - 1 + slides.length) % slides.length);

  return (
    <div className="relative w-full aspect-[4/3] lg:aspect-video rounded-3xl overflow-hidden shadow-2xl shadow-brand-gold/10 border border-brand-white/10 group">
      <AnimatePresence mode="wait">
        <motion.div
          key={current}
          initial={{ opacity: 0, scale: 1.1 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
          className="absolute inset-0"
        >
          <img 
            src={slides[current].image} 
            alt={slides[current].title}
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-brand-black via-brand-black/20 to-transparent" />
          
          {/* Slide Caption */}
          <div className="absolute bottom-8 left-8 right-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <h3 className="text-2xl md:text-3xl font-display font-bold text-brand-gold mb-1">
                {slides[current].title}
              </h3>
              <p className="text-brand-white-off/60 uppercase tracking-widest text-xs font-bold">
                {slides[current].subtitle}
              </p>
            </motion.div>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Controls */}
      <div className="absolute inset-y-0 left-4 flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
        <button onClick={prev} className="w-10 h-10 rounded-full bg-brand-black/50 backdrop-blur-md border border-brand-white/10 flex items-center justify-center text-brand-white-off hover:bg-brand-gold hover:text-brand-black transition-all">
          <ChevronLeft size={24} />
        </button>
      </div>
      <div className="absolute inset-y-0 right-4 flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
        <button onClick={next} className="w-10 h-10 rounded-full bg-brand-black/50 backdrop-blur-md border border-brand-white/10 flex items-center justify-center text-brand-white-off hover:bg-brand-gold hover:text-brand-black transition-all">
          <ChevronRight size={24} />
        </button>
      </div>

      {/* Indicators */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2">
        {slides.map((_, i) => (
          <button
            key={i}
            onClick={() => setCurrent(i)}
            className={`h-1 transition-all duration-300 rounded-full ${i === current ? 'w-8 bg-brand-gold' : 'w-4 bg-brand-white/20'}`}
          />
        ))}
      </div>
    </div>
  );
}
