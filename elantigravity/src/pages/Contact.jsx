import { motion } from "framer-motion";
import { Mail, Phone, MapPin, Globe, MessageCircle } from "lucide-react";
import QuoteBuilder from "../components/QuoteBuilder";

export default function Contact() {
  return (
    <div className="pt-32 pb-24 px-6 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <header className="mb-20 text-center max-w-3xl mx-auto">
          <motion.h1 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-5xl md:text-7xl mb-8 font-display text-brand-black dark:text-brand-white-off transition-colors"
          >
            Let's Start a <span className="text-gradient-gold">Conversation</span>
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-xl text-brand-black/60 dark:text-brand-white-off/60 leading-relaxed transition-colors"
          >
            Ready to elevate your brand presence? Fill out our luxury quote builder or reach out through our official channels.
          </motion.p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-16 items-start">
          {/* Quote Builder */}
          <div className="lg:col-span-3">
            <QuoteBuilder />
          </div>

          {/* Contact Info */}
          <div className="lg:col-span-2 space-y-12">
            <div>
              <h3 className="text-2xl mb-8 font-display">Contact Details</h3>
              <div className="space-y-8">
                <div className="flex gap-6">
                  <div className="w-12 h-12 bg-brand-gold/10 rounded-xl flex items-center justify-center text-brand-gold border border-brand-gold/20 shrink-0">
                    <Mail size={24} />
                  </div>
                  <div>
                    <p className="text-sm text-brand-black/40 dark:text-brand-white-off/40 uppercase tracking-widest font-bold mb-1 transition-colors">Email Us</p>
                    <p className="text-xl font-medium text-brand-black dark:text-brand-white-off transition-colors">hello@elacreatives.com</p>
                    <p className="text-sm text-brand-black/30 dark:text-brand-white-off/30 mt-1 transition-colors">Direct inquiries only</p>
                  </div>
                </div>
                <div className="flex gap-6">
                  <div className="w-12 h-12 bg-brand-gold/10 rounded-xl flex items-center justify-center text-brand-gold border border-brand-gold/20 shrink-0">
                    <Phone size={24} />
                  </div>
                  <div>
                    <p className="text-sm text-brand-black/40 dark:text-brand-white-off/40 uppercase tracking-widest font-bold mb-1 transition-colors">Call Us</p>
                    <p className="text-xl font-medium text-brand-black dark:text-brand-white-off transition-colors">+1 234 567 890</p>
                    <p className="text-sm text-brand-black/30 dark:text-brand-white-off/30 mt-1 transition-colors">Mon - Fri, 9am - 6pm</p>
                  </div>
                </div>
                <div className="flex gap-6">
                  <div className="w-12 h-12 bg-brand-gold/10 rounded-xl flex items-center justify-center text-brand-gold border border-brand-white/20 shrink-0">
                    <MapPin size={24} />
                  </div>
                  <div>
                    <p className="text-sm text-brand-black/40 dark:text-brand-white-off/40 uppercase tracking-widest font-bold mb-1 transition-colors">Visit Studio</p>
                    <p className="text-xl font-medium text-brand-black dark:text-brand-white-off transition-colors">123 Agency Drive, Suite 500</p>
                    <p className="text-sm text-brand-black/30 dark:text-brand-white-off/30 mt-1 transition-colors">Luxury Design District, NY</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-brand-gold/5 border border-brand-gold/20 p-8 rounded-3xl">
              <div className="flex items-center gap-4 mb-6">
                <MessageCircle size={32} className="text-brand-gold" />
                <h4 className="text-xl font-display">Quick Integration</h4>
              </div>
              <p className="text-brand-black/60 dark:text-brand-white-off/60 mb-8 leading-relaxed transition-colors">
                Prefer a direct conversation? Message us on WhatsApp for an immediate response regarding project timelines and basic pricing.
              </p>
              <a 
                href="https://wa.me/1234567890" 
                target="_blank" 
                rel="noopener noreferrer"
                className="btn-gold w-full text-center block"
              >
                Chat on WhatsApp
              </a>
            </div>

            {/* Map Placeholder */}
            <div className="h-64 rounded-3xl overflow-hidden grayscale opacity-50 border border-brand-white/10 relative">
              <div className="absolute inset-0 bg-brand-black/20" />
              <img src="/assets/billboard.png" alt="Map Location" className="w-full h-full object-cover" />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="bg-brand-black/80 px-6 py-3 rounded-full border border-brand-gold/30 backdrop-blur-sm text-sm font-bold uppercase tracking-widest text-brand-gold">
                  Studio Location
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
