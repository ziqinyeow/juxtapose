import Link from "next/link";
import localFont from "next/font/local";
import { useEffect, useState } from "react";

const mistrully = localFont({
  src: "../public/font/mistrully.ttf",
  display: "swap",
  variable: "--font-mistrully",
});

const Logo = () => {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);
  return (
    mounted && (
      <Link
        href="/"
        title="Home"
        className={`hover:opacity-75 ${mistrully.className}`}
      >
        <span className="text-2xl font-bold">Juxtapose</span>
      </Link>
    )
  );
};

export default Logo;
