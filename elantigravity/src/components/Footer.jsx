import { Link } from "react-router-dom";
import { Facebook, Instagram, Twitter, Linkedin, Mail, MapPin, Phone } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-brand-gold/5 dark:bg-brand-black-subtle border-t border-brand-black/5 dark:border-brand-white/5 pt-16 pb-8 px-6 transition-colors duration-500">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-16">
          {/* Brand Info */}
          <div className="col-span-1 md:col-span-1">
            <Link to="/" className="flex items-center group mb-6">
              <img 
                src="/assets/Ela Creatives_Logo with Slogan_Final-01.png" 
                alt="Ela Creatives Logo" 
                className="h-16 w-auto group-hover:scale-105 transition-transform"
              />
            </Link>
            <p className="text-brand-black/60 dark:text-brand-white-off/60 leading-relaxed max-w-xs transition-colors">
              Elevating brands through cinematic design and world-class marketing. We turn vision into authoritative reality.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-display font-bold text-lg mb-6 text-brand-gold">Agency</h4>
            <ul className="space-y-4">
              <li><Link to="/services" className="text-brand-black/60 dark:text-brand-white-off/60 hover:text-brand-gold transition-colors">Services</Link></li>
              <li><Link to="/portfolio" className="text-brand-black/60 dark:text-brand-white-off/60 hover:text-brand-gold transition-colors">Portfolio</Link></li>
              <li><Link to="/about" className="text-brand-black/60 dark:text-brand-white-off/60 hover:text-brand-gold transition-colors">Our Story</Link></li>
              <li><Link to="/contact" className="text-brand-black/60 dark:text-brand-white-off/60 hover:text-brand-gold transition-colors">Contact</Link></li>
            </ul>
          </div>

          {/* Services */}
          <div>
            <h4 className="font-display font-bold text-lg mb-6 text-brand-gold">Expertise</h4>
            <ul className="space-y-4">
              <li className="text-brand-black/60 dark:text-brand-white-off/60 hover:text-brand-gold transition-colors cursor-pointer">Outdoor Advertising</li>
              <li className="text-brand-black/60 dark:text-brand-white-off/60 hover:text-brand-gold transition-colors cursor-pointer">Corporate Identity</li>
              <li className="text-brand-black/60 dark:text-brand-white-off/60 hover:text-brand-gold transition-colors cursor-pointer">Custom Apparel</li>
              <li className="text-brand-black/60 dark:text-brand-white-off/60 hover:text-brand-gold transition-colors cursor-pointer">Digital Strategy</li>
            </ul>
          </div>

          {/* Contact Info */}
          <div>
            <h4 className="font-display font-bold text-lg mb-6 text-brand-gold">Connect</h4>
            <div className="space-y-4 mb-8">
              <div className="flex items-center gap-3 text-brand-black/60 dark:text-brand-white-off/60 transition-colors">
                <MapPin size={18} className="text-brand-gold" />
                <span>123 Agency Drive, Luxury District</span>
              </div>
              <div className="flex items-center gap-3 text-brand-black/60 dark:text-brand-white-off/60 transition-colors">
                <Phone size={18} className="text-brand-gold" />
                <span>+1 234 567 890</span>
              </div>
              <div className="flex items-center gap-3 text-brand-black/60 dark:text-brand-white-off/60 transition-colors">
                <Mail size={18} className="text-brand-gold" />
                <span>hello@elacreatives.com</span>
              </div>
            </div>
            <div className="flex gap-4">
              {[Facebook, Instagram, Twitter, Linkedin].map((Icon, i) => (
                <a key={i} href="#" className="w-10 h-10 rounded-full border border-brand-black/10 dark:border-brand-white/10 flex items-center justify-center text-brand-black/60 dark:text-brand-white-off/60 hover:border-brand-gold hover:text-brand-gold transition-all duration-300">
                  <Icon size={18} />
                </a>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-brand-white/5 pt-8 flex flex-col md:row items-center justify-between gap-4">
          <p className="text-brand-white-off/40 text-sm">
            © {new Date().getFullYear()} Ela Creatives Branding Agency. All rights reserved.
          </p>
          <div className="flex gap-8 text-xs text-brand-white-off/40 uppercase tracking-widest">
            <a href="#" className="hover:text-brand-gold">Privacy Policy</a>
            <a href="#" className="hover:text-brand-gold">Terms of Service</a>
          </div>
        </div>
      </div>
    </footer>
  );
}
