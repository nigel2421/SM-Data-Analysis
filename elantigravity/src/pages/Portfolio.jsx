import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowUpRight, Search, Filter } from "lucide-react";
import { cn } from "../lib/utils";
import { portfolioProjects, categories } from "../data/portfolioData";

export default function Portfolio() {
  const [activeCategory, setActiveCategory] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");

  const filteredProjects = portfolioProjects.filter((p) => {
    const matchesCategory = activeCategory === "All" || p.category === activeCategory;
    const matchesSearch = p.title.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          p.client.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  return (
    <div className="pt-32 pb-24 px-6 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <header className="mb-16">
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-5xl md:text-7xl mb-6 font-display text-brand-black dark:text-brand-white-off transition-colors"
          >
            Our <span className="text-gradient-gold">Showcase</span>
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-xl text-brand-black/60 dark:text-brand-white-off/60 max-w-2xl leading-relaxed transition-colors"
          >
            A curated gallery of our most impactful branding projects, from street-side billboards to board-room stationery.
          </motion.p>
        </header>

        {/* Filters */}
        <div className="flex flex-col md:flex-row items-center justify-between gap-8 mb-12">
          <div className="flex flex-wrap gap-4">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat)}
                className={cn(
                  "px-8 py-3 rounded-full border transition-all duration-300 font-bold",
                  activeCategory === cat 
                    ? "bg-brand-gold border-brand-gold text-brand-black shadow-lg shadow-brand-gold/20" 
                    : "border-brand-black/10 dark:border-brand-white/10 text-brand-black/60 dark:text-brand-white-off/60 hover:border-brand-gold/40 transition-colors"
                )}
              >
                {cat}
              </button>
            ))}
          </div>
          <div className="relative group w-full md:w-auto">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-brand-black/40 dark:text-brand-white-off/40 group-focus-within:text-brand-gold transition-colors" size={20} />
            <input 
              type="text" 
              placeholder="Search projects or clients..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-brand-gold/5 dark:bg-brand-black-subtle/40 border border-brand-black/10 dark:border-brand-white/10 rounded-full pl-12 pr-6 py-3 text-brand-black dark:text-brand-white-off placeholder:text-brand-black/40 dark:placeholder:text-brand-white-off/40 focus:outline-none focus:border-brand-gold transition-all duration-300 md:w-80"
            />
          </div>
        </div>

        {/* Grid */}
        <motion.div 
          layout
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
        >
          <AnimatePresence mode="popLayout">
            {filteredProjects.map((project) => (
              <motion.div
                key={project.id}
                layout
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.4 }}
                className="glass-card group cursor-pointer"
              >
                <div className="h-80 overflow-hidden relative">
                  <img src={project.image} alt={project.title} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700" />
                  <div className="absolute inset-0 bg-gradient-to-t from-brand-black via-transparent to-transparent opacity-0 group-hover:opacity-90 transition-opacity duration-300 flex flex-col justify-end p-8">
                    <div className="w-12 h-12 bg-brand-gold rounded-full flex items-center justify-center text-brand-black ml-auto mb-4 -translate-y-4 opacity-0 group-hover:translate-y-0 group-hover:opacity-100 transition-all duration-300 delay-100">
                      <ArrowUpRight size={24} />
                    </div>
                  </div>
                </div>
                <div className="p-8">
                  <span className="text-brand-gold font-bold text-sm tracking-widest uppercase block mb-2">{project.category}</span>
                  <h3 className="text-2xl mb-2 text-brand-black dark:text-brand-white-off transition-colors">{project.title}</h3>
                  <p className="text-brand-black/60 dark:text-brand-white-off/60 text-sm leading-relaxed transition-colors">{project.description}</p>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </motion.div>

        {/* Empty State */}
        {filteredProjects.length === 0 && (
          <div className="py-32 text-center text-brand-black/40 dark:text-brand-white-off/40 transition-colors">
            <Filter size={48} className="mx-auto mb-6 opacity-20" />
            <h3 className="text-2xl text-brand-black/60 dark:text-brand-white-off/60">No projects found in this category</h3>
          </div>
        )}
      </div>
    </div>
  );
}
