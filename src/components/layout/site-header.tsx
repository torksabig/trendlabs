import Link from "next/link";

const links = [
  { href: "/", label: "Dashboard" },
  { href: "/markets", label: "Markets" },
  { href: "/agents", label: "Agents" },
  { href: "/research", label: "Research" },
  { href: "/trades", label: "Trades" },
];

export function SiteHeader() {
  return (
    <header className="border-b border-ink/10 bg-paper/80 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-6 px-4 py-4 sm:px-6">
        <Link href="/" className="group flex items-baseline gap-3">
          <span className="font-display text-3xl tracking-tight text-ink sm:text-4xl">
            TrendLab
          </span>
          <span className="hidden text-xs uppercase tracking-[0.22em] text-ink/45 sm:inline">
            Paper research
          </span>
        </Link>
        <nav className="flex flex-wrap items-center gap-1 text-sm">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="rounded-md px-3 py-2 text-ink/70 transition hover:bg-ink/5 hover:text-ink"
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
