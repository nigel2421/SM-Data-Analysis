import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X, ChevronRight } from "lucide-react";
import { cn } from "../lib/utils";
import ThemeToggle from "./ThemeToggle";

const navLinks = [
  { name: "Home", path: "/" },
  { name: "Services", path: "/services" },
  { name: "Portfolio", path: "/portfolio" },
  { name: "About", path: "/about" },
  { name: "Contact", path: "/contact" },
];

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <nav
      className={cn(
        "fixed top-0 w-full z-50 transition-all duration-300 px-6 py-4",
        scrolled ? "glass-nav py-3" : "bg-transparent"
      )}
    >
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center group">
          <img 
            src="/assets/Ela Creatives_Logo with Slogan_Final-01.png" 
            alt="Ela Creatives Logo" 
            className="h-16 md:h-20 w-auto group-hover:scale-105 transition-transform"
          />
        </Link>

        {/* Desktop Nav */}
        <div className="hidden md:flex items-center gap-8">
          {navLinks.map((link) => (
            <Link
              key={link.name}
              to={link.path}
              className={cn(
                "font-medium transition-colors hover:text-brand-gold",
                location.pathname === link.path ? "text-brand-gold" : "text-brand-black/70 dark:text-brand-white-off/70"
              )}
            >
              {link.name}
            </Link>
          ))}
          <div className="flex items-center gap-4">
            <ThemeToggle />
            <Link to="/contact" className="btn-gold !py-2 !px-6 text-sm">
              Get a Quote
            </Link>
          </div>
        </div>

        {/* Mobile Toggle */}
        <button
          className="md:hidden text-brand-black dark:text-brand-white-off transition-colors ml-4"
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? <X size={28} /> : <Menu size={28} />}
        </button>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="absolute top-full left-0 w-full bg-brand-white/95 dark:bg-brand-black/95 backdrop-blur-xl border-b border-brand-black/5 dark:border-brand-white/10 md:hidden overflow-hidden transition-colors"
          >
            <div className="flex flex-col p-6 gap-4">
              <div className="flex items-center justify-between mb-4 pb-4 border-b border-brand-black/5 dark:border-brand-white/5 transition-colors">
                <span className="text-brand-black/60 dark:text-brand-white-off/60 font-bold uppercase tracking-widest text-xs">Switch Theme</span>
                <ThemeToggle />
              </div>
              {navLinks.map((link) => (
                <Link
                  key={link.name}
                  to={link.path}
                  onClick={() => setIsOpen(false)}
                  className={cn(
                    "text-xl font-display py-2 flex items-center justify-between group transition-colors",
                    location.pathname === link.path ? "text-brand-gold" : "text-brand-black/70 dark:text-brand-white-off/70"
                  )}
                >
                  {link.name}
                  <ChevronRight size={20} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                </Link>
              ))}
              <Link
                to="/contact"
                onClick={() => setIsOpen(false)}
                className="btn-gold mt-4 text-center"
              >
                Instant Quote
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
