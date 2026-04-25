import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeft, ChevronRight } from "lucide-react";

export default function ServiceCarousel({ images }) {
  const [current, setCurrent] = useState(0);

  const next = () => setCurrent((prev) => (prev + 1) % images.length);
  const prev = () => setCurrent((prev) => (prev - 1 + images.length) % images.length);

  return (
    <div className="relative w-full aspect-video rounded-3xl overflow-hidden shadow-2xl shadow-brand-gold/10 border border-brand-black/5 dark:border-brand-white/10 group">
      <AnimatePresence mode="wait">
        <motion.div
          key={current}
          initial={{ opacity: 0, x: 100 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -100 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="absolute inset-0"
        >
          <img 
            src={images[current]} 
            alt={`Showcase item ${current + 1}`} 
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-brand-black/40 to-transparent" />
        </motion.div>
      </AnimatePresence>

      {/* Navigation Arrows */}
      <div className="absolute inset-y-0 left-4 flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
        <button 
          onClick={prev} 
          className="w-12 h-12 rounded-full bg-brand-white/10 dark:bg-brand-black/50 backdrop-blur-md border border-brand-black/5 dark:border-brand-white/10 flex items-center justify-center text-brand-black dark:text-brand-white-off hover:bg-brand-gold hover:text-brand-black transition-all"
        >
          <ChevronLeft size={24} />
        </button>
      </div>
      <div className="absolute inset-y-0 right-4 flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
        <button 
          onClick={next} 
          className="w-12 h-12 rounded-full bg-brand-white/10 dark:bg-brand-black/50 backdrop-blur-md border border-brand-black/5 dark:border-brand-white/10 flex items-center justify-center text-brand-black dark:text-brand-white-off hover:bg-brand-gold hover:text-brand-black transition-all"
        >
          <ChevronRight size={24} />
        </button>
      </div>

      {/* Slides Indicator */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex gap-3">
        {images.map((_, i) => (
          <button
            key={i}
            onClick={() => setCurrent(i)}
            className={`h-1.5 transition-all duration-300 rounded-full ${
              i === current ? 'w-10 bg-brand-gold' : 'w-4 bg-white/20 hover:bg-white/40'
            }`}
          />
        ))}
      </div>
    </div>
  );
}
